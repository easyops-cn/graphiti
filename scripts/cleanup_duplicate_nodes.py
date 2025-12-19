#!/usr/bin/env python3
"""
Offline cleanup script for deduplicating existing database nodes.

This script identifies and merges duplicate entities in the database using LLM-based
semantic deduplication. It's designed to clean up legacy data that was imported before
the Map-Reduce deduplication was implemented.

Usage:
    # Dry run (no changes)
    python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --dry-run

    # Execute cleanup for specific entity type
    python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --entity-type ProductModule

    # Execute cleanup for all entity types
    python scripts/cleanup_duplicate_nodes.py --group-id easyops_support
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.utils.bulk_utils import _get_entity_type_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_entity_node_from_record(record: dict[str, Any], provider: GraphProvider) -> EntityNode:
    """Convert a database record to EntityNode."""
    labels = record.get('labels', ['Entity'])
    if isinstance(labels, str):
        labels = [labels]

    # Parse type_scores from JSON string if present
    type_scores = None
    if record.get('type_scores'):
        try:
            type_scores = json.loads(record['type_scores'])
        except (json.JSONDecodeError, TypeError):
            pass

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        labels=labels,
        summary=record.get('summary', ''),
        attributes=record.get('attributes', {}),
        reasoning=record.get('reasoning'),
        type_scores=type_scores,
        type_confidence=record.get('type_confidence'),
        created_at=record.get('created_at'),
    )


async def get_all_nodes_by_type(
    driver: GraphDriver,
    group_id: str,
    entity_type: str | None = None,
) -> dict[str, list[EntityNode]]:
    """Query all entity nodes from database, grouped by entity type."""

    if entity_type:
        # Query specific entity type
        query = f"""
            MATCH (n:{entity_type})
            WHERE n.group_id = $group_id
            RETURN n.uuid AS uuid,
                   n.name AS name,
                   n.group_id AS group_id,
                   labels(n) AS labels,
                   n.summary AS summary,
                   properties(n) AS attributes,
                   n.reasoning AS reasoning,
                   n.type_scores AS type_scores,
                   n.type_confidence AS type_confidence,
                   n.created_at AS created_at
        """
    else:
        # Query all entity nodes
        query = """
            MATCH (n:Entity)
            WHERE n.group_id = $group_id
            RETURN n.uuid AS uuid,
                   n.name AS name,
                   n.group_id AS group_id,
                   labels(n) AS labels,
                   n.summary AS summary,
                   properties(n) AS attributes,
                   n.reasoning AS reasoning,
                   n.type_scores AS type_scores,
                   n.type_confidence AS type_confidence,
                   n.created_at AS created_at
        """

    records, _, _ = await driver.execute_query(query, group_id=group_id)

    nodes_by_type: dict[str, list[EntityNode]] = {}
    for record in records:
        node = get_entity_node_from_record(dict(record), driver.provider)
        node_type = _get_entity_type_label(node)
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)

    return nodes_by_type


async def identify_duplicates_with_llm(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[tuple[EntityNode, EntityNode]]:
    """Use LLM to identify duplicate pairs among existing nodes.

    Returns list of (source_node, canonical_node) tuples where source should be merged into canonical.
    """
    if len(nodes) < 2:
        return []

    entity_types_dict = entity_types or {}

    # Build context for LLM - nodes as both ENTITIES and EXISTING ENTITIES
    # We compare each node against all subsequent nodes (O(n^2) but for cleanup this is acceptable)
    all_duplicate_pairs: list[tuple[EntityNode, EntityNode]] = []

    for i, node in enumerate(nodes):
        candidates = nodes[i + 1:]  # Only compare with subsequent nodes to avoid duplicate pairs
        if not candidates:
            continue

        # Build context
        entity_type_label = _get_entity_type_label(node)
        type_model = entity_types_dict.get(entity_type_label)

        extracted_nodes_context = [{
            'id': 0,
            'name': node.name,
            'summary': node.summary or '',
            'entity_type': node.labels,
            'entity_type_description': type_model.__doc__ if type_model else 'Default Entity Type',
            **node.attributes,
        }]

        existing_nodes_context = [
            {
                'idx': j,
                'name': candidate.name,
                'summary': candidate.summary or '',
                'entity_types': candidate.labels,
                **candidate.attributes,
            }
            for j, candidate in enumerate(candidates)
        ]

        context = {
            'extracted_nodes': extracted_nodes_context,
            'existing_nodes': existing_nodes_context,
            'episode_content': '',
            'previous_episodes': [],
        }

        try:
            llm_response = await llm_client.generate_response(
                prompt_library.dedupe_nodes.nodes(context),
                response_model=NodeResolutions,
                prompt_name='cleanup_dedupe_nodes',
            )

            node_resolutions = NodeResolutions(**llm_response).entity_resolutions
            for resolution in node_resolutions:
                if resolution.duplicate_idx == -1:
                    continue
                if resolution.duplicate_idx < 0 or resolution.duplicate_idx >= len(candidates):
                    continue

                canonical_node = candidates[resolution.duplicate_idx]
                if node.uuid != canonical_node.uuid:
                    logger.info(
                        '[cleanup_dedup] Duplicate found: "%s" -> "%s", reasoning: %s',
                        node.name,
                        canonical_node.name,
                        resolution.reasoning or 'no reasoning',
                    )
                    all_duplicate_pairs.append((node, canonical_node))

        except Exception as e:
            logger.error('[cleanup_dedup] LLM dedup failed for node "%s": %s', node.name, e)

    return all_duplicate_pairs


async def merge_duplicate_nodes(
    driver: GraphDriver,
    source_node: EntityNode,
    canonical_node: EntityNode,
    dry_run: bool = True,
) -> bool:
    """Merge source node into canonical node.

    1. Update all edges pointing to/from source to point to canonical
    2. Merge source's summary and attributes into canonical
    3. Delete source node

    Returns True if successful.
    """
    logger.info(
        '[cleanup_merge] %s "%s" (uuid=%s) -> "%s" (uuid=%s)',
        'Would merge' if dry_run else 'Merging',
        source_node.name,
        source_node.uuid[:8],
        canonical_node.name,
        canonical_node.uuid[:8],
    )

    if dry_run:
        return True

    try:
        # Step 1: Update edges - redirect source edges to canonical
        # Update outgoing edges (source is the source_node_uuid)
        await driver.execute_query(
            """
            MATCH (source:Entity {uuid: $source_uuid})-[r]->(target:Entity)
            WHERE NOT target.uuid = $canonical_uuid
            MATCH (canonical:Entity {uuid: $canonical_uuid})
            CREATE (canonical)-[r2:RELATES_TO]->(target)
            SET r2 = properties(r)
            DELETE r
            """,
            source_uuid=source_node.uuid,
            canonical_uuid=canonical_node.uuid,
        )

        # Update incoming edges (source is the target_node_uuid)
        await driver.execute_query(
            """
            MATCH (origin:Entity)-[r]->(source:Entity {uuid: $source_uuid})
            WHERE NOT origin.uuid = $canonical_uuid
            MATCH (canonical:Entity {uuid: $canonical_uuid})
            CREATE (origin)-[r2:RELATES_TO]->(canonical)
            SET r2 = properties(r)
            DELETE r
            """,
            source_uuid=source_node.uuid,
            canonical_uuid=canonical_node.uuid,
        )

        # Step 2: Merge summary and attributes into canonical
        merged_summary = canonical_node.summary or ''
        if source_node.summary and source_node.summary not in merged_summary:
            merged_summary = f"{merged_summary} {source_node.summary}".strip()

        merged_attributes = dict(canonical_node.attributes)
        for key, value in source_node.attributes.items():
            if key not in merged_attributes or not merged_attributes[key]:
                merged_attributes[key] = value

        # Update canonical node
        await driver.execute_query(
            """
            MATCH (n:Entity {uuid: $uuid})
            SET n.summary = $summary
            """,
            uuid=canonical_node.uuid,
            summary=merged_summary,
        )

        # Step 3: Delete source node
        await driver.execute_query(
            """
            MATCH (n:Entity {uuid: $uuid})
            DETACH DELETE n
            """,
            uuid=source_node.uuid,
        )

        logger.info('[cleanup_merge] Successfully merged "%s" into "%s"', source_node.name, canonical_node.name)
        return True

    except Exception as e:
        logger.error('[cleanup_merge] Failed to merge "%s": %s', source_node.name, e)
        return False


async def cleanup_duplicates(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_id: str,
    entity_type: str | None = None,
    dry_run: bool = True,
):
    """Main cleanup function."""
    logger.info('[cleanup] Starting cleanup for group_id=%s, entity_type=%s, dry_run=%s',
                group_id, entity_type or 'all', dry_run)

    # Step 1: Query all nodes
    nodes_by_type = await get_all_nodes_by_type(driver, group_id, entity_type)
    total_nodes = sum(len(nodes) for nodes in nodes_by_type.values())
    logger.info('[cleanup] Found %d nodes across %d entity types', total_nodes, len(nodes_by_type))

    for node_type, type_info in nodes_by_type.items():
        logger.info('[cleanup]   - %s: %d nodes', node_type, len(type_info))

    # Step 2: Identify duplicates for each entity type
    all_duplicate_pairs: list[tuple[EntityNode, EntityNode]] = []

    for node_type, nodes in nodes_by_type.items():
        if len(nodes) < 2:
            continue

        logger.info('[cleanup] Checking %d %s nodes for duplicates...', len(nodes), node_type)
        duplicate_pairs = await identify_duplicates_with_llm(llm_client, nodes)
        all_duplicate_pairs.extend(duplicate_pairs)
        logger.info('[cleanup] Found %d duplicate pairs for %s', len(duplicate_pairs), node_type)

    if not all_duplicate_pairs:
        logger.info('[cleanup] No duplicates found!')
        return

    logger.info('[cleanup] Total duplicate pairs to merge: %d', len(all_duplicate_pairs))

    # Step 3: Merge duplicates
    success_count = 0
    for source_node, canonical_node in all_duplicate_pairs:
        if await merge_duplicate_nodes(driver, source_node, canonical_node, dry_run):
            success_count += 1

    logger.info('[cleanup] %s %d/%d duplicate pairs',
                'Would merge' if dry_run else 'Merged',
                success_count, len(all_duplicate_pairs))


async def main():
    parser = argparse.ArgumentParser(description='Cleanup duplicate nodes in the database')
    parser.add_argument('--group-id', required=True, help='Group ID to cleanup')
    parser.add_argument('--entity-type', help='Specific entity type to cleanup (e.g., ProductModule)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--falkordb-host', default=os.getenv('FALKORDB_HOST', 'localhost'))
    parser.add_argument('--falkordb-port', type=int, default=int(os.getenv('FALKORDB_PORT', '6379')))
    parser.add_argument('--falkordb-database', default=os.getenv('FALKORDB_DATABASE', 'elevo_memory'))

    args = parser.parse_args()

    # Initialize FalkorDB driver
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = FalkorDriver(
        host=args.falkordb_host,
        port=args.falkordb_port,
        database=args.falkordb_database,
    )

    # Initialize LLM client
    llm_config = LLMConfig(
        api_key=os.getenv('OPENAI_API_KEY'),
        model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
    llm_client = OpenAIGenericClient(llm_config)

    try:
        await cleanup_duplicates(
            driver=driver,
            llm_client=llm_client,
            group_id=args.group_id,
            entity_type=args.entity_type,
            dry_run=args.dry_run,
        )
    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
