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

import re
from typing import Any

from graphiti_core.driver.driver import GraphProvider


def get_episode_node_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.NEPTUNE:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: join([x IN coalesce($entity_edges, []) | toString(x) ], '|'), created_at: $created_at, valid_at: $valid_at, user_id: $user_id}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges,
                    n.user_id = $user_id
                RETURN n.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            # Use SET directly with vecf32() for vector properties
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n.uuid = $uuid, n.name = $name, n.group_id = $group_id,
                    n.source_description = $source_description, n.source = $source,
                    n.content = $content, n.entity_edges = $entity_edges,
                    n.created_at = $created_at, n.valid_at = $valid_at,
                    n.summary = $summary, n.tags = $tags,
                    n.content_embedding = CASE
                        WHEN $content_embedding IS NOT NULL AND size($content_embedding) > 0
                        THEN vecf32($content_embedding)
                        ELSE NULL
                    END,
                    n.content_hash = $content_hash,
                    n.content_storage_type = $content_storage_type,
                    n.content_file_path = $content_file_path,
                    n.content_file_size = $content_file_size,
                    n.user_id = $user_id
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at, user_id: $user_id}
                RETURN n.uuid AS uuid
            """


def get_episode_node_save_bulk_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.NEPTUNE:
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description,
                    source: episode.source, content: episode.content,
                entity_edges: join([x IN coalesce(episode.entity_edges, []) | toString(x) ], '|'), created_at: episode.created_at, valid_at: episode.valid_at, user_id: episode.user_id}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges,
                    n.user_id = $user_id
                RETURN n.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n.uuid = episode.uuid, n.name = episode.name, n.group_id = episode.group_id,
                    n.source_description = episode.source_description, n.source = episode.source,
                    n.content = episode.content, n.entity_edges = episode.entity_edges,
                    n.created_at = episode.created_at, n.valid_at = episode.valid_at,
                    n.summary = episode.summary, n.tags = episode.tags,
                    n.content_hash = episode.content_hash,
                    n.content_storage_type = episode.content_storage_type,
                    n.content_file_path = episode.content_file_path,
                    n.content_file_size = episode.content_file_size,
                    n.user_id = episode.user_id
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content,
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at, user_id: episode.user_id}
                RETURN n.uuid AS uuid
            """


EPISODIC_NODE_RETURN = """
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
    e.entity_edges AS entity_edges,
    e.summary AS summary,
    e.tags AS tags,
    e.content_hash AS content_hash,
    e.content_storage_type AS content_storage_type,
    e.content_file_path AS content_file_path,
    e.content_file_size AS content_file_size,
    e.user_id AS user_id
"""

EPISODIC_NODE_RETURN_NEPTUNE = """
    e.content AS content,
    e.created_at AS created_at,
    e.valid_at AS valid_at,
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.source_description AS source_description,
    e.source AS source,
    split(e.entity_edges, ",") AS entity_edges,
    e.summary AS summary,
    split(e.tags, "|") AS tags,
    e.user_id AS user_id
"""


def get_entity_node_save_query(
    provider: GraphProvider,
    labels: str,
    has_aoss: bool = False,
    *,
    maintain_revision: bool = False,
) -> str:
    if maintain_revision:
        if provider != GraphProvider.FALKORDB:
            raise NotImplementedError('Entity revision maintenance requires FalkorDB')
        return f"""
            OPTIONAL MATCH (candidate {{uuid: $entity_data.uuid}})
            WITH collect(candidate) AS matches
            WITH matches, size(matches) AS match_count
            WITH match_count, CASE WHEN match_count = 1 THEN matches[0] ELSE NULL END AS existing
            WITH match_count, existing, existing._revision AS current_revision,
                 existing.group_id AS current_group_id
            WHERE match_count = 0 OR
                  (match_count = 1 AND 'Entity' IN labels(existing) AND
                   current_group_id = $entity_data.group_id AND
                   (current_revision IS NULL OR
                    (current_revision >= 1 AND
                     toString(current_revision) = toString(toInteger(current_revision)))))
            WITH existing, current_revision,
                 CASE WHEN existing IS NULL THEN true ELSE NOT (
                     all(key IN [key IN keys(existing) WHERE key <> '_revision' AND
                         key <> 'name_embedding' AND key <> 'summary_embedding']
                         WHERE key IN keys($entity_data) AND $entity_data[key] IS NOT NULL AND
                               existing[key] = $entity_data[key]) AND
                     all(key IN [key IN keys($entity_data) WHERE key <> 'name_embedding' AND
                         key <> 'summary_embedding' AND $entity_data[key] IS NOT NULL]
                         WHERE key IN keys(existing)) AND
                     all(label IN {labels.split(':')!r} WHERE label IN labels(existing)) AND
                     (CASE WHEN $entity_data.name_embedding IS NULL
                           THEN existing.name_embedding IS NULL
                           ELSE coalesce(existing.name_embedding =
                                vecf32($entity_data.name_embedding), false) END) AND
                     (CASE WHEN $entity_data.summary_embedding IS NULL
                           THEN existing.summary_embedding IS NULL
                           ELSE coalesce(existing.summary_embedding =
                                vecf32($entity_data.summary_embedding), false) END)
                 ) END AS changed
            MERGE (n:Entity {{uuid: $entity_data.uuid}})
            FOREACH (write_marker IN CASE WHEN changed THEN [1] ELSE [] END |
                SET n:{labels}
                SET n = $entity_data,
                    n.name_embedding = vecf32($entity_data.name_embedding),
                    n.summary_embedding = CASE WHEN $entity_data.summary_embedding IS NOT NULL
                        THEN vecf32($entity_data.summary_embedding) ELSE NULL END,
                    n._revision = CASE WHEN existing IS NULL THEN 1
                        ELSE coalesce(current_revision, 1) + 1 END)
            RETURN n.uuid AS uuid, n._revision AS revision, changed AS changed
        """
    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                SET n.name_embedding = vecf32($entity_data.name_embedding)
                SET n.summary_embedding = CASE WHEN $entity_data.summary_embedding IS NOT NULL THEN vecf32($entity_data.summary_embedding) ELSE NULL END
                RETURN n.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                WITH n
                RETURN n.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            label_subquery = ''
            for label in labels.split(':'):
                label_subquery += f' SET n:{label}\n'
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                {label_subquery}
                SET n = removeKeyFromMap(removeKeyFromMap($entity_data, "labels"), "name_embedding")
                SET n.name_embedding = join([x IN coalesce($entity_data.name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            """
        case _:
            save_embedding_query = (
                'WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)'
                if not has_aoss
                else ''
            )
            return (
                f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                """
                + save_embedding_query
                + """
                RETURN n.uuid AS uuid
            """
            )


def get_entity_node_save_bulk_query(
    provider: GraphProvider,
    nodes: list[dict],
    has_aoss: bool = False,
    *,
    maintain_revision: bool = False,
) -> str | Any:
    if maintain_revision:
        if provider != GraphProvider.FALKORDB:
            raise NotImplementedError('Bulk entity revision maintenance requires FalkorDB')
        queries = []
        core_properties = {
            'uuid', 'name', 'group_id', 'summary', 'created_at', 'labels',
            'reasoning', 'type_scores', 'type_confidence', 'synonyms',
            'name_embedding', 'summary_embedding', '_revision',
        }
        for original_node in nodes:
            node = dict(original_node)
            node.pop('_revision', None)
            node.setdefault('synonyms', None)
            labels = list(dict.fromkeys(node['labels']))
            custom_properties = [key for key in node if key not in core_properties]
            identifiers = labels + custom_properties
            if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', identifier) for identifier in identifiers):
                raise ValueError('Invalid entity label or property name')
            label_clause = ':'.join(labels)
            scalar_properties = [
                'name', 'group_id', 'summary', 'created_at', 'reasoning',
                'type_scores', 'type_confidence', 'synonyms',
            ] + custom_properties
            scalar_same = [
                f'(coalesce(existing.{key} = $node.{key}, false) OR '
                f'(existing.{key} IS NULL AND $node.{key} IS NULL))'
                for key in scalar_properties
            ]
            scalar_sets = ',\n                    '.join(
                f'n.{key} = $node.{key}' for key in scalar_properties
            )
            same_conditions = ' AND\n                    '.join(
                scalar_same + [
                    'all(label IN $node.labels WHERE label IN labels(existing))',
                    'all(label IN $node.labels WHERE label IN coalesce(existing.labels, []))',
                    '(CASE WHEN $node.name_embedding IS NULL '
                    'THEN existing.name_embedding IS NULL '
                    'ELSE coalesce(existing.name_embedding = '
                    'vecf32($node.name_embedding), false) END)',
                    '(CASE WHEN $node.summary_embedding IS NULL '
                    'THEN existing.summary_embedding IS NULL '
                    'ELSE coalesce(existing.summary_embedding = '
                    'vecf32($node.summary_embedding), false) END)',
                ]
            )
            queries.append((f"""
                OPTIONAL MATCH (candidate {{uuid: $node.uuid}})
                WITH collect(candidate) AS matches
                WITH matches, size(matches) AS match_count
                WITH match_count,
                     CASE WHEN match_count = 1 THEN matches[0] ELSE NULL END AS existing
                WITH match_count, existing, existing._revision AS prior_revision
                WHERE match_count = 0 OR
                      (match_count = 1 AND 'Entity' IN labels(existing) AND
                       existing.group_id = $node.group_id AND
                       (prior_revision IS NULL OR
                        (prior_revision >= 1 AND
                         toString(prior_revision) = toString(toInteger(prior_revision)))))
                WITH match_count, prior_revision, existing,
                     coalesce(existing.labels, []) AS stored_labels,
                     CASE WHEN match_count = 0 THEN [] ELSE labels(existing) END AS physical_labels
                WITH match_count, prior_revision, existing,
                     stored_labels + [label IN physical_labels WHERE NOT label IN stored_labels]
                     AS known_labels
                WITH match_count, prior_revision, existing,
                     known_labels + [label IN $node.labels WHERE NOT label IN known_labels]
                     AS labels_to_save
                WITH match_count, prior_revision, existing, labels_to_save,
                     CASE WHEN match_count = 0 THEN true ELSE NOT (
                         {same_conditions}
                     ) END AS changed
                MERGE (n:Entity {{uuid: $node.uuid}})
                FOREACH (write_marker IN CASE WHEN changed THEN [1] ELSE [] END |
                    SET n:{label_clause}
                    SET {scalar_sets},
                        n.labels = labels_to_save,
                        n.name_embedding = vecf32($node.name_embedding),
                        n.summary_embedding = CASE WHEN $node.summary_embedding IS NOT NULL
                            THEN vecf32($node.summary_embedding) ELSE NULL END,
                        n._revision = CASE WHEN match_count = 0 THEN 1
                            ELSE coalesce(prior_revision, 1) + 1 END
                )
                RETURN n.uuid AS uuid, n._revision AS revision, changed AS changed
            """, {'node': node}))
        return queries
    # Core properties that are always present (excluding embeddings which need special handling)
    CORE_PROPERTIES = {
        'uuid', 'name', 'group_id', 'summary', 'created_at', 'labels',
        'reasoning', 'type_scores', 'type_confidence',
        'name_embedding', 'summary_embedding',  # These are handled separately
        'synonyms',  # EasyOps: top-level field for BM25 + semantic search
    }

    match provider:
        case GraphProvider.FALKORDB:
            queries = []
            for node in nodes:
                # Find custom attributes (properties not in CORE_PROPERTIES)
                custom_attrs = [k for k in node.keys() if k not in CORE_PROPERTIES]

                # Build custom attributes SET clause
                custom_set_clause = ''
                if custom_attrs:
                    custom_set_parts = [f'n.{attr} = node.{attr}' for attr in custom_attrs]
                    custom_set_clause = ',\n                                ' + ',\n                                '.join(custom_set_parts)

                for label in node['labels']:
                    # EasyOps fix: Set properties individually to avoid List type for embeddings
                    # Using SET n = node would set embeddings as List, then vecf32() would fail
                    # because FalkorDB doesn't allow overwriting List with Vectorf32
                    queries.append(
                        (
                            f"""
                            UNWIND $nodes AS node
                            MERGE (n:Entity {{uuid: node.uuid}})
                            SET n:{label}
                            SET n.uuid = node.uuid,
                                n.name = node.name,
                                n.group_id = node.group_id,
                                n.summary = node.summary,
                                n.created_at = node.created_at,
                                n.labels = node.labels,
                                n.reasoning = node.reasoning,
                                n.type_scores = node.type_scores,
                                n.type_confidence = node.type_confidence,
                                n.synonyms = node.synonyms{custom_set_clause}
                            WITH n, node
                            SET n.name_embedding = vecf32(node.name_embedding)
                            SET n.summary_embedding = CASE WHEN node.summary_embedding IS NOT NULL THEN vecf32(node.summary_embedding) ELSE NULL END
                            RETURN n.uuid AS uuid
                            """,
                            {'nodes': [node]},
                        )
                    )
            return queries
        case GraphProvider.NEPTUNE:
            queries = []
            for node in nodes:
                labels = ''
                for label in node['labels']:
                    labels += f' SET n:{label}\n'
                queries.append(
                    f"""
                        UNWIND $nodes AS node
                        MERGE (n:Entity {{uuid: node.uuid}})
                        {labels}
                        SET n = removeKeyFromMap(removeKeyFromMap(node, "labels"), "name_embedding")
                        SET n.name_embedding = join([x IN coalesce(node.name_embedding, []) | toString(x) ], ",")
                        RETURN n.uuid AS uuid
                    """
                )
            return queries
        case GraphProvider.KUZU:
            return """
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            save_embedding_query = (
                'WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)'
                if not has_aoss
                else ''
            )
            return (
                """
                    UNWIND $nodes AS node
                    MERGE (n:Entity {uuid: node.uuid})
                    SET n:$(node.labels)
                    SET n = node
                    """
                + save_embedding_query
                + """
                RETURN n.uuid AS uuid
            """
            )


def get_entity_node_return_query(provider: GraphProvider, *, include_embeddings: bool = False) -> str:
    # `name_embedding` is not returned by default and must be loaded manually using `load_name_embedding()`.
    if provider == GraphProvider.KUZU:
        return """
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.labels AS labels,
            n.created_at AS created_at,
            n.summary AS summary,
            n.attributes AS attributes
        """

    embedding_fields = (
        ',\n        n.name_embedding AS name_embedding,\n'
        '        n.summary_embedding AS summary_embedding'
        if include_embeddings else ''
    )
    return """
        n.uuid AS uuid,
        n.name AS name,
        n.group_id AS group_id,
        n.created_at AS created_at,
        n.summary AS summary,
        labels(n) AS labels,
        properties(n) AS attributes
    """ + embedding_fields


def get_community_node_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at, name_embedding: vecf32($name_embedding)}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
                SET n.name_embedding = join([x IN coalesce($name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
                WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
                RETURN n.uuid AS uuid
            """


COMMUNITY_NODE_RETURN = """
    c.uuid AS uuid,
    c.name AS name,
    c.group_id AS group_id,
    c.created_at AS created_at,
    c.name_embedding AS name_embedding,
    c.summary AS summary
"""

COMMUNITY_NODE_RETURN_NEPTUNE = """
    n.uuid AS uuid,
    n.name AS name,
    [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
"""
