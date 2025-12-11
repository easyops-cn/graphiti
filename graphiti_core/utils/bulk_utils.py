"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Any

from graphiti_core.driver.driver import (
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_bulk_query_by_type,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


def _get_entity_type_label(node: EntityNode) -> str:
    """Get the most specific entity type label (non-'Entity') from a node."""
    for label in node.labels:
        if label != 'Entity':
            return label
    return 'Entity'


async def _resolve_batch_with_llm(
    llm_client: LLMClient,
    unresolved_nodes: list[EntityNode],
    canonical_candidates: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None,
) -> list[tuple[EntityNode, EntityNode]]:
    """Use LLM to resolve semantic duplicates that deterministic matching missed.

    EasyOps customization: Enables batch-internal deduplication of semantic synonyms
    like 'EasyITSM' vs 'IT服务中心' that cannot be matched by exact string or MinHash.

    Returns:
        List of (source_node, canonical_node) pairs for detected duplicates.
    """
    if not unresolved_nodes or not canonical_candidates:
        return []

    entity_types_dict = entity_types or {}

    # Build context for extracted nodes (unresolved)
    extracted_nodes_context = []
    for i, node in enumerate(unresolved_nodes):
        entity_type_label = _get_entity_type_label(node)
        type_model = entity_types_dict.get(entity_type_label)
        extracted_nodes_context.append({
            'id': i,
            'name': node.name,
            'summary': node.summary or '',
            'entity_type': node.labels,
            'entity_type_description': type_model.__doc__ if type_model else 'Default Entity Type',
        })

    # Build context for existing candidates (canonical pool)
    existing_nodes_context = [
        {
            'idx': i,
            'name': candidate.name,
            'summary': candidate.summary or '',
            'entity_types': candidate.labels,
            **candidate.attributes,
        }
        for i, candidate in enumerate(canonical_candidates)
    ]

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': '',
        'previous_episodes': [],
    }

    logger.info(
        '[batch_dedup_llm] Sending %d nodes to LLM against %d candidates',
        len(unresolved_nodes),
        len(canonical_candidates),
    )

    try:
        llm_response = await llm_client.generate_response(
            prompt_library.dedupe_nodes.nodes(context),
            response_model=NodeResolutions,
            prompt_name='dedupe_nodes.nodes_batch',
        )

        node_resolutions = NodeResolutions(**llm_response).entity_resolutions
        duplicate_pairs: list[tuple[EntityNode, EntityNode]] = []

        for resolution in node_resolutions:
            node_id = resolution.id
            duplicate_idx = resolution.duplicate_idx

            if node_id < 0 or node_id >= len(unresolved_nodes):
                continue
            if duplicate_idx == -1:
                continue
            if duplicate_idx < 0 or duplicate_idx >= len(canonical_candidates):
                continue

            source_node = unresolved_nodes[node_id]
            canonical_node = canonical_candidates[duplicate_idx]

            if source_node.uuid != canonical_node.uuid:
                logger.info(
                    '[batch_dedup_llm] Duplicate found: "%s" -> "%s"',
                    source_node.name,
                    canonical_node.name,
                )
                duplicate_pairs.append((source_node, canonical_node))

        return duplicate_pairs

    except Exception as e:
        logger.error('[batch_dedup_llm] LLM dedup failed: %s', e)
        return []


def _merge_node_into_canonical(source: EntityNode, canonical: EntityNode) -> None:
    """Merge source node's summary and attributes into canonical node.

    EasyOps customization: When LLM identifies duplicates, merge their information
    to preserve all extracted knowledge.
    """
    # Merge summary: concatenate if both exist, prefer non-empty
    if source.summary and canonical.summary:
        if source.summary not in canonical.summary:
            canonical.summary = f"{canonical.summary} {source.summary}"
    elif source.summary and not canonical.summary:
        canonical.summary = source.summary

    # Merge attributes: source attributes fill in missing canonical attributes
    for key, value in source.attributes.items():
        if key not in canonical.attributes or not canonical.attributes[key]:
            canonical.attributes[key] = value

    # EasyOps: Record synonyms (space-separated string for BM25 full-text index)
    if source.name and source.name != canonical.name:
        existing_synonyms = canonical.attributes.get('synonyms', '')
        synonym_list = existing_synonyms.split() if existing_synonyms else []
        if source.name not in synonym_list:
            synonym_list.append(source.name)
            canonical.attributes['synonyms'] = ' '.join(synonym_list)


async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityNode]:
    """Perform LLM-based semantic deduplication on hydrated nodes.

    EasyOps customization: Run after extract_attributes_from_nodes so that
    summary and attributes are available for LLM to make better decisions.

    Groups nodes by entity type and uses LLM to identify semantic duplicates
    (e.g., 'EasyITSM' vs 'IT服务中心') that deterministic matching cannot handle.

    Returns:
        Deduplicated list of nodes with merged summaries and attributes.
    """
    if len(nodes) < 2:
        return nodes

    # Group nodes by entity type
    nodes_by_type: dict[str, list[EntityNode]] = {}
    for node in nodes:
        entity_type = _get_entity_type_label(node)
        if entity_type not in nodes_by_type:
            nodes_by_type[entity_type] = []
        nodes_by_type[entity_type].append(node)

    # Track which nodes are duplicates (uuid -> canonical_uuid)
    duplicate_map: dict[str, str] = {}
    nodes_by_uuid: dict[str, EntityNode] = {node.uuid: node for node in nodes}

    # For each entity type with multiple nodes, check for semantic duplicates
    for entity_type, type_nodes in nodes_by_type.items():
        if len(type_nodes) < 2:
            continue

        logger.info(
            '[semantic_dedup] Checking %d %s entities for semantic duplicates',
            len(type_nodes), entity_type,
        )

        # Check each node against others of same type
        for i, node in enumerate(type_nodes):
            if node.uuid in duplicate_map:
                continue  # Already identified as duplicate

            candidates = [n for n in type_nodes[i+1:] if n.uuid not in duplicate_map]
            if not candidates:
                continue

            llm_pairs = await _resolve_batch_with_llm(
                clients.llm_client, [node], candidates, entity_types
            )

            for source, canonical in llm_pairs:
                # Merge source into canonical
                _merge_node_into_canonical(source, canonical)
                duplicate_map[source.uuid] = canonical.uuid
                logger.info(
                    '[semantic_dedup] Merged "%s" into "%s"',
                    source.name, canonical.name,
                )

    # Return deduplicated nodes
    deduped_nodes = [node for node in nodes if node.uuid not in duplicate_map]
    return deduped_nodes


def _sanitize_string_for_falkordb(value: str) -> str:
    """Sanitize string content for FalkorDB query parameters.

    FalkorDB's stringify_param_value only escapes backslashes and double quotes,
    but control characters (newlines, carriage returns, tabs, etc.) can break
    the Cypher query parsing. This function escapes these characters.
    """
    if not isinstance(value, str):
        return value
    # Escape control characters that can break FalkorDB query parsing
    # Note: backslash must be escaped first to avoid double-escaping
    value = value.replace('\\', '\\\\')
    value = value.replace('\n', '\\n')
    value = value.replace('\r', '\\r')
    value = value.replace('\t', '\\t')
    value = value.replace('\0', '')  # Remove null bytes entirely
    return value


def _sanitize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize attributes to only include primitive types that FalkorDB supports.

    FalkorDB only supports primitive types (str, int, float, bool) or arrays of primitive types
    as property values. This function filters out any non-primitive values.
    """
    if not attributes:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Only keep arrays of primitive types
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                sanitized[key] = value
            else:
                logger.warning(
                    f'Skipping attribute {key}: array contains non-primitive types'
                )
        else:
            logger.warning(
                f'Skipping attribute {key}: value type {type(value).__name__} is not supported by FalkorDB'
            )
    return sanitized


def _build_directed_uuid_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Collapse alias -> canonical chains while preserving direction.

    The incoming pairs represent directed mappings discovered during node dedupe. We use a simple
    union-find with iterative path compression to ensure every source UUID resolves to its ultimate
    canonical target, even if aliases appear lexicographically smaller than the canonical UUID.
    """

    parent: dict[str, str] = {}

    def find(uuid: str) -> str:
        """Directed union-find lookup using iterative path compression."""
        parent.setdefault(uuid, uuid)
        root = uuid
        while parent[root] != root:
            root = parent[root]

        while parent[uuid] != root:
            next_uuid = parent[uuid]
            parent[uuid] = root
            uuid = next_uuid

        return root

    for source_uuid, target_uuid in pairs:
        parent.setdefault(source_uuid, source_uuid)
        parent.setdefault(target_uuid, target_uuid)
        parent[find(source_uuid)] = find(target_uuid)

    return {uuid: find(uuid) for uuid in parent}


class RawEpisode(BaseModel):
    name: str
    uuid: str | None = Field(default=None)
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id]
            )
            for episode in episodes
        ]
    )
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]] = [
        (episode, previous_episodes_list[i]) for i, episode in enumerate(episodes)
    ]

    return episode_tuples


async def add_nodes_and_edges_bulk(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
):
    session = driver.session()
    try:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
    finally:
        await session.close()


async def add_nodes_and_edges_bulk_tx(
    tx: GraphDriverSession,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)
        # Sanitize string fields to prevent FalkorDB query parsing issues
        if driver.provider == GraphProvider.FALKORDB:
            if 'content' in episode and episode['content']:
                episode['content'] = _sanitize_string_for_falkordb(episode['content'])
            if 'name' in episode and episode['name']:
                episode['name'] = _sanitize_string_for_falkordb(episode['name'])
            if 'source_description' in episode and episode['source_description']:
                episode['source_description'] = _sanitize_string_for_falkordb(episode['source_description'])

    nodes = []

    for node in entity_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(embedder)

        # Sanitize string fields for FalkorDB
        name = node.name
        summary = node.summary
        reasoning = node.reasoning
        if driver.provider == GraphProvider.FALKORDB:
            name = _sanitize_string_for_falkordb(name) if name else name
            summary = _sanitize_string_for_falkordb(summary) if summary else summary
            reasoning = _sanitize_string_for_falkordb(reasoning) if reasoning else reasoning

        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': name,
            'group_id': node.group_id,
            'summary': summary,
            'created_at': node.created_at,
            'name_embedding': node.name_embedding,
            'labels': list(set(node.labels + ['Entity'])),
            'reasoning': reasoning,
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
            entity_data['attributes'] = json.dumps(attributes)
        else:
            # Sanitize attributes to only include primitive types (FalkorDB requirement)
            entity_data.update(_sanitize_attributes(node.attributes))

        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        if edge.fact_embedding is None:
            await edge.generate_embedding(embedder)

        # Sanitize string fields for FalkorDB
        edge_name = edge.name
        edge_fact = edge.fact
        if driver.provider == GraphProvider.FALKORDB:
            edge_name = _sanitize_string_for_falkordb(edge_name) if edge_name else edge_name
            edge_fact = _sanitize_string_for_falkordb(edge_fact) if edge_fact else edge_fact

        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge_name,
            'fact': edge_fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'fact_embedding': edge.fact_embedding,
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(edge.attributes) if edge.attributes else {}
            edge_data['attributes'] = json.dumps(attributes)
        else:
            # Sanitize attributes to only include primitive types (FalkorDB requirement)
            edge_data.update(_sanitize_attributes(edge.attributes))

        edges.append(edge_data)

    if driver.graph_operations_interface:
        await driver.graph_operations_interface.episodic_node_save_bulk(None, driver, tx, episodes)
        await driver.graph_operations_interface.node_save_bulk(None, driver, tx, nodes)
        await driver.graph_operations_interface.episodic_edge_save_bulk(
            None, driver, tx, [edge.model_dump() for edge in episodic_edges]
        )
        await driver.graph_operations_interface.edge_save_bulk(None, driver, tx, edges)

    elif driver.provider == GraphProvider.KUZU:
        # FIXME: Kuzu's UNWIND does not currently support STRUCT[] type properly, so we insert the data one by one instead for now.
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        for episode in episodes:
            await tx.run(episode_query, **episode)
        entity_node_query = get_entity_node_save_bulk_query(driver.provider, nodes)
        for node in nodes:
            await tx.run(entity_node_query, **node)
        entity_edge_query = get_entity_edge_save_bulk_query(driver.provider)
        for edge in edges:
            await tx.run(entity_edge_query, **edge)
        episodic_edge_query = get_episodic_edge_save_bulk_query(driver.provider)
        for edge in episodic_edges:
            await tx.run(episodic_edge_query, **edge.model_dump())
    else:
        # Log bulk save operation details for debugging
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        logger.info(f'[bulk_save] Saving {len(episodes)} episodes, query_len={len(episode_query)}, first_episode_uuid={episodes[0]["uuid"] if episodes else "none"}')
        if not episode_query or not episode_query.strip():
            logger.error(f'[bulk_save] Empty episode query! provider={driver.provider}, episodes_count={len(episodes)}')
        await tx.run(episode_query, episodes=episodes)
        await tx.run(
            get_entity_node_save_bulk_query(driver.provider, nodes),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        # Group edges by type and save each group with the correct relationship type
        # This is necessary because Cypher doesn't support dynamic relationship types
        edges_by_type: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            edge_type = edge.get('name', 'RELATES_TO')
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)

        for edge_type, typed_edges in edges_by_type.items():
            query = get_entity_edge_save_bulk_query_by_type(driver.provider, edge_type)
            logger.info(f'[bulk_save] Saving {len(typed_edges)} edges of type {edge_type}')
            await tx.run(query, entity_edges=typed_edges)


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]]]:
    extracted_nodes_bulk: list[list[EntityNode]] = await semaphore_gather(
        *[
            extract_nodes(clients, episode, previous_episodes, entity_types, excluded_entity_types)
            for episode, previous_episodes in episode_tuples
        ]
    )

    extracted_edges_bulk: list[list[EntityEdge]] = await semaphore_gather(
        *[
            extract_edges(
                clients,
                episode,
                extracted_nodes_bulk[i],
                previous_episodes,
                edge_type_map=edge_type_map,
                group_id=episode.group_id,
                edge_types=edge_types,
            )
            for i, (episode, previous_episodes) in enumerate(episode_tuples)
        ]
    )

    return extracted_nodes_bulk, extracted_edges_bulk


async def dedupe_nodes_bulk(
    clients: GraphitiClients,
    extracted_nodes: list[list[EntityNode]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    """Resolve entity duplicates across an in-memory batch using a two-pass strategy.

    1. Run :func:`resolve_extracted_nodes` for every episode in parallel so each batch item is
       reconciled against the live graph just like the non-batch flow.
    2. Re-run the deterministic similarity heuristics across the union of resolved nodes to catch
       duplicates that only co-occur inside this batch, emitting a canonical UUID map that callers
       can apply to edges and persistence.
    """

    first_pass_results = await semaphore_gather(
        *[
            resolve_extracted_nodes(
                clients,
                nodes,
                episode_tuples[i][0],
                episode_tuples[i][1],
                entity_types,
            )
            for i, nodes in enumerate(extracted_nodes)
        ]
    )

    episode_resolutions: list[tuple[str, list[EntityNode]]] = []
    per_episode_uuid_maps: list[dict[str, str]] = []
    duplicate_pairs: list[tuple[str, str]] = []

    for (resolved_nodes, uuid_map, duplicates), (episode, _) in zip(
        first_pass_results, episode_tuples, strict=True
    ):
        episode_resolutions.append((episode.uuid, resolved_nodes))
        per_episode_uuid_maps.append(uuid_map)
        duplicate_pairs.extend((source.uuid, target.uuid) for source, target in duplicates)

    canonical_nodes: dict[str, EntityNode] = {}
    for _, resolved_nodes in episode_resolutions:
        for node in resolved_nodes:
            # NOTE: this loop is O(n^2) in the number of nodes inside the batch because we rebuild
            # the MinHash index for the accumulated canonical pool each time. The LRU-backed
            # shingle cache keeps the constant factors low for typical batch sizes (≤ CHUNK_SIZE),
            # but if batches grow significantly we should switch to an incremental index or chunked
            # processing.
            if not canonical_nodes:
                canonical_nodes[node.uuid] = node
                continue

            existing_candidates = list(canonical_nodes.values())
            normalized = _normalize_string_exact(node.name)
            exact_match = next(
                (
                    candidate
                    for candidate in existing_candidates
                    if _normalize_string_exact(candidate.name) == normalized
                ),
                None,
            )
            if exact_match is not None:
                if exact_match.uuid != node.uuid:
                    duplicate_pairs.append((node.uuid, exact_match.uuid))
                continue

            indexes = _build_candidate_indexes(existing_candidates)
            state = DedupResolutionState(
                resolved_nodes=[None],
                uuid_map={},
                unresolved_indices=[],
            )
            _resolve_with_similarity([node], indexes, state)

            resolved = state.resolved_nodes[0]
            if resolved is None:
                canonical_nodes[node.uuid] = node
                continue

            canonical_uuid = resolved.uuid
            canonical_nodes.setdefault(canonical_uuid, resolved)
            if canonical_uuid != node.uuid:
                duplicate_pairs.append((node.uuid, canonical_uuid))

    union_pairs: list[tuple[str, str]] = []
    for uuid_map in per_episode_uuid_maps:
        union_pairs.extend(uuid_map.items())
    union_pairs.extend(duplicate_pairs)

    compressed_map: dict[str, str] = _build_directed_uuid_map(union_pairs)

    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for episode_uuid, resolved_nodes in episode_resolutions:
        deduped_nodes: list[EntityNode] = []
        seen: set[str] = set()
        for node in resolved_nodes:
            canonical_uuid = compressed_map.get(node.uuid, node.uuid)
            if canonical_uuid in seen:
                continue
            seen.add(canonical_uuid)
            canonical_node = canonical_nodes.get(canonical_uuid)
            if canonical_node is None:
                logger.error(
                    'Canonical node %s missing during batch dedupe; falling back to %s',
                    canonical_uuid,
                    node.uuid,
                )
                canonical_node = node
            deduped_nodes.append(canonical_node)

        nodes_by_episode[episode_uuid] = deduped_nodes

    return nodes_by_episode, compressed_map


async def dedupe_edges_bulk(
    clients: GraphitiClients,
    extracted_edges: list[list[EntityEdge]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    _entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    _edge_type_map: dict[tuple[str, str], list[str]],
) -> dict[str, list[EntityEdge]]:
    embedder = clients.embedder
    min_score = 0.6

    # generate embeddings
    await semaphore_gather(
        *[create_entity_edge_embeddings(embedder, edges) for edges in extracted_edges]
    )

    # Find similar results
    dedupe_tuples: list[tuple[EpisodicNode, EntityEdge, list[EntityEdge]]] = []
    for i, edges_i in enumerate(extracted_edges):
        existing_edges: list[EntityEdge] = []
        for edges_j in extracted_edges:
            existing_edges += edges_j

        for edge in edges_i:
            candidates: list[EntityEdge] = []
            for existing_edge in existing_edges:
                # Skip self-comparison
                if edge.uuid == existing_edge.uuid:
                    continue
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                if (
                    edge.source_node_uuid != existing_edge.source_node_uuid
                    or edge.target_node_uuid != existing_edge.target_node_uuid
                ):
                    continue

                edge_words = set(edge.fact.lower().split())
                existing_edge_words = set(existing_edge.fact.lower().split())
                has_overlap = not edge_words.isdisjoint(existing_edge_words)
                if has_overlap:
                    candidates.append(existing_edge)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(edge.fact_embedding or []),
                    normalize_l2(existing_edge.fact_embedding or []),
                )
                if similarity >= min_score:
                    candidates.append(existing_edge)

            dedupe_tuples.append((episode_tuples[i][0], edge, candidates))

    bulk_edge_resolutions: list[
        tuple[EntityEdge, EntityEdge, list[EntityEdge]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_edge(
                clients.llm_client,
                edge,
                candidates,
                candidates,
                episode,
                edge_types,
                set(edge_types),
            )
            for episode, edge, candidates in dedupe_tuples
        ]
    )

    # For now we won't track edge invalidation
    duplicate_pairs: list[tuple[str, str]] = []
    for i, (_, _, duplicates) in enumerate(bulk_edge_resolutions):
        episode, edge, candidates = dedupe_tuples[i]
        for duplicate in duplicates:
            duplicate_pairs.append((edge.uuid, duplicate.uuid))

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_pairs)

    edge_uuid_map: dict[str, EntityEdge] = {
        edge.uuid: edge for edges in extracted_edges for edge in edges
    }

    edges_by_episode: dict[str, list[EntityEdge]] = {}
    for i, edges in enumerate(extracted_edges):
        episode = episode_tuples[i][0]

        edges_by_episode[episode.uuid] = [
            edge_uuid_map[compressed_map.get(edge.uuid, edge.uuid)] for edge in edges
        ]

    return edges_by_episode


class UnionFind:
    def __init__(self, elements):
        # start each element in its own set
        self.parent = {e: e for e in elements}

    def find(self, x):
        # path‐compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # attach the lexicographically larger root under the smaller
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def compress_uuid_map(duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    all_ids: iterable of all entity IDs (strings)
    duplicate_pairs: iterable of (id1, id2) pairs
    returns: dict mapping each id -> lexicographically smallest id in its duplicate set
    """
    all_uuids = set()
    for pair in duplicate_pairs:
        all_uuids.add(pair[0])
        all_uuids.add(pair[1])

    uf = UnionFind(all_uuids)
    for a, b in duplicate_pairs:
        uf.union(a, b)
    # ensure full path‐compression before mapping
    return {uuid: uf.find(uuid) for uuid in all_uuids}


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges
