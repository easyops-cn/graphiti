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

from graphiti_core.driver.driver import GraphProvider

EPISODIC_EDGE_SAVE = """
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MATCH (node:Entity {uuid: $entity_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""


def get_episodic_edge_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (episode:Episodic {uuid: $source_node_uuid})
            MATCH (node:Entity {uuid: $target_node_uuid})
            MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at
            RETURN e.uuid AS uuid
        """

    return """
        UNWIND $episodic_edges AS edge
        MATCH (episode:Episodic {uuid: edge.source_node_uuid})
        MATCH (node:Entity {uuid: edge.target_node_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
        SET
            e.group_id = edge.group_id,
            e.created_at = edge.created_at
        RETURN e.uuid AS uuid
    """


EPISODIC_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_entity_edge_save_query(
    provider: GraphProvider, has_aoss: bool = False, *, maintain_revision: bool = False
) -> str:
    if maintain_revision:
        if provider != GraphProvider.FALKORDB:
            raise NotImplementedError('Edge revision maintenance requires FalkorDB')
        return """
            OPTIONAL MATCH (candidate_source)-[candidate]->(candidate_target)
            WHERE candidate.uuid = $edge_data.uuid
            WITH collect(candidate) AS matches, collect(candidate_source) AS source_nodes,
                 collect(candidate_target) AS target_nodes
            WITH matches, source_nodes, target_nodes, size(matches) AS match_count
            WITH match_count,
                 CASE WHEN match_count = 1 THEN matches[0] ELSE NULL END AS existing,
                 CASE WHEN match_count = 1 THEN source_nodes[0] ELSE NULL END AS existing_source,
                 CASE WHEN match_count = 1 THEN target_nodes[0] ELSE NULL END AS existing_target
            WITH match_count, existing, existing._revision AS prior_revision,
                 existing_source, existing_target
            WHERE match_count = 0 OR
                  (match_count = 1 AND existing.group_id = $edge_data.group_id AND
                   existing_source.uuid = $edge_data.source_uuid AND
                   existing_target.uuid = $edge_data.target_uuid AND
                   type(existing) = 'RELATES_TO' AND
                   (prior_revision IS NULL OR
                    (prior_revision >= 1 AND
                     toString(prior_revision) = toString(toInteger(prior_revision)))))
            OPTIONAL MATCH (source_candidate {uuid: $edge_data.source_uuid})
            WITH match_count, existing, prior_revision, existing_source, existing_target,
                 collect(source_candidate) AS sources
            WHERE size(sources) = 1 AND 'Entity' IN labels(sources[0])
            WITH match_count, existing, prior_revision, existing_source, existing_target,
                 sources[0] AS source
            OPTIONAL MATCH (target_candidate {uuid: $edge_data.target_uuid})
            WITH match_count, existing, prior_revision, existing_source, existing_target,
                 source, collect(target_candidate) AS targets
            WHERE size(targets) = 1 AND 'Entity' IN labels(targets[0]) AND
                  source.group_id = $edge_data.group_id AND
                  targets[0].group_id = $edge_data.group_id AND
                  (match_count = 0 OR
                   (existing_source = source AND existing_target = targets[0]))
            WITH match_count, existing, prior_revision, source, targets[0] AS target
            WITH match_count, existing, prior_revision, source, target,
                 CASE WHEN match_count = 0 THEN true ELSE NOT (
                     all(key IN [key IN keys(existing) WHERE key <> '_revision' AND
                         key <> 'fact_embedding']
                         WHERE key IN keys($edge_data) AND $edge_data[key] IS NOT NULL AND
                               existing[key] = $edge_data[key]) AND
                     all(key IN [key IN keys($edge_data) WHERE key <> 'fact_embedding' AND
                         $edge_data[key] IS NOT NULL]
                         WHERE key IN keys(existing)) AND
                     (CASE WHEN $edge_data.fact_embedding IS NULL
                           THEN existing.fact_embedding IS NULL
                           ELSE coalesce(existing.fact_embedding =
                                vecf32($edge_data.fact_embedding), false) END)
                 ) END AS changed
            FOREACH (write_marker IN CASE WHEN match_count = 0 THEN [1] ELSE [] END |
                CREATE (source)-[created:RELATES_TO]->(target)
                SET created = $edge_data,
                    created.fact_embedding = vecf32($edge_data.fact_embedding),
                    created._revision = 1)
            FOREACH (write_marker IN CASE WHEN match_count = 1 AND changed THEN [1] ELSE [] END |
                SET existing = $edge_data,
                    existing.fact_embedding = vecf32($edge_data.fact_embedding),
                    existing._revision = coalesce(prior_revision, 1) + 1)
            WITH source, target, changed
            OPTIONAL MATCH (source)-[saved]->(target)
            WHERE saved.uuid = $edge_data.uuid AND type(saved) = 'RELATES_TO'
            WITH changed, collect(saved) AS saved_matches
            WHERE size(saved_matches) = 1
            RETURN saved_matches[0].uuid AS uuid,
                   saved_matches[0]._revision AS revision, changed AS changed
        """
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = $edge_data
                SET e.fact_embedding = vecf32($edge_data.fact_embedding)
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = removeKeyFromMap(removeKeyFromMap($edge_data, "fact_embedding"), "episodes")
                SET e.fact_embedding = join([x IN coalesce($edge_data.fact_embedding, []) | toString(x) ], ",")
                SET e.episodes = join($edge_data.episodes, ",")
                RETURN $edge_data.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (source:Entity {uuid: $source_uuid})
                MATCH (target:Entity {uuid: $target_uuid})
                MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at,
                    e.name = $name,
                    e.fact = $fact,
                    e.fact_embedding = $fact_embedding,
                    e.episodes = $episodes,
                    e.expired_at = $expired_at,
                    e.valid_at = $valid_at,
                    e.invalid_at = $invalid_at,
                    e.attributes = $attributes
                RETURN e.uuid AS uuid
            """
        case _:  # Neo4j
            save_embedding_query = (
                """WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)"""
                if not has_aoss
                else ''
            )
            return (
                (
                    """
                        MATCH (source:Entity {uuid: $edge_data.source_uuid})
                        MATCH (target:Entity {uuid: $edge_data.target_uuid})
                        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                        SET e = $edge_data
                        """
                    + save_embedding_query
                )
                + """
                RETURN e.uuid AS uuid
                """
            )


def get_entity_edge_temporal_replace_query(provider: GraphProvider) -> str:
    if provider != GraphProvider.FALKORDB:
        raise NotImplementedError('Temporal edge replacement requires FalkorDB')
    return """
        OPTIONAL MATCH ()-[old_candidate]->()
        WHERE old_candidate.uuid = $old_data.uuid
        WITH collect(old_candidate) AS old_matches
        WHERE size(old_matches) = 1
        WITH old_matches[0] AS old
        WITH old, old._revision AS prior_revision
        MATCH (old_source:Entity)-[old]->(old_target:Entity)
        WHERE old.group_id = $old_data.group_id AND old.name = $old_data.name AND
              type(old) = 'RELATES_TO' AND
              old.invalid_at IS NULL AND
              old.expired_at IS NULL AND
              old_source.uuid = $old_data.source_uuid AND
              old_target.uuid = $old_data.target_uuid AND
              (prior_revision IS NULL OR
               (prior_revision >= 1 AND
                toString(prior_revision) = toString(toInteger(prior_revision))))
        OPTIONAL MATCH ()-[new_candidate]->()
        WHERE new_candidate.uuid = $edge_data.uuid
        WITH old, prior_revision, collect(new_candidate) AS new_matches
        WHERE size(new_matches) = 0
        OPTIONAL MATCH (source_candidate:Entity {uuid: $edge_data.source_uuid})
        WITH old, prior_revision, collect(source_candidate) AS sources
        WHERE size(sources) = 1
        WITH old, prior_revision, sources[0] AS source
        OPTIONAL MATCH (target_candidate:Entity {uuid: $edge_data.target_uuid})
        WITH old, prior_revision, source, collect(target_candidate) AS targets
        WHERE size(targets) = 1 AND source.group_id = $edge_data.group_id AND
              targets[0].group_id = $edge_data.group_id
        WITH old, prior_revision, source, targets[0] AS target
        SET old.invalid_at = $old_data.invalid_at,
            old.expired_at = $old_data.expired_at,
            old._revision = coalesce(prior_revision, 1) + 1
        CREATE (source)-[new:RELATES_TO]->(target)
        SET new = $edge_data
        SET new.fact_embedding = vecf32($edge_data.fact_embedding)
        SET new._revision = 1
        RETURN old.uuid AS old_uuid, new.uuid AS new_uuid
    """


def get_entity_edge_save_bulk_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = removeKeyFromMap(removeKeyFromMap(edge, "fact_embedding"), "episodes")
                SET r.fact_embedding = join([x IN coalesce(edge.fact_embedding, []) | toString(x) ], ",")
                SET r.episodes = join(edge.episodes, ",")
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (source:Entity {uuid: $source_node_uuid})
                MATCH (target:Entity {uuid: $target_node_uuid})
                MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at,
                    e.name = $name,
                    e.fact = $fact,
                    e.fact_embedding = $fact_embedding,
                    e.episodes = $episodes,
                    e.expired_at = $expired_at,
                    e.valid_at = $valid_at,
                    e.invalid_at = $invalid_at,
                    e.attributes = $attributes
                RETURN e.uuid AS uuid
            """
        case _:
            save_embedding_query = (
                'WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)'
                if not has_aoss
                else ''
            )
            return (
                """
                    UNWIND $entity_edges AS edge
                    MATCH (source:Entity {uuid: edge.source_node_uuid})
                    MATCH (target:Entity {uuid: edge.target_node_uuid})
                    MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
                    SET e = edge
                    """
                + save_embedding_query
                + """
                RETURN edge.uuid AS uuid
            """
            )


def get_entity_edge_save_bulk_query_by_type(
    provider: GraphProvider, edge_type: str, *, maintain_revision: bool = False
) -> str:
    """Generate edge save query for a specific edge type.

    This is necessary because Cypher doesn't support dynamic relationship types.
    We generate a query with the edge type hardcoded in the MERGE clause.

    EasyOps customization: MERGE by (source, target) only, not by uuid.
    This ensures that duplicate edges between the same nodes are merged,
    not created as separate edges.
    """
    if maintain_revision:
        if provider != GraphProvider.FALKORDB:
            raise NotImplementedError('Typed relation revision maintenance requires FalkorDB')
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', edge_type) is None:
            raise ValueError(f'Invalid typed relation name: {edge_type}')
        return f"""
            OPTIONAL MATCH ()-[uuid_candidate]->()
            WHERE uuid_candidate.uuid = $edge.uuid
            WITH collect(uuid_candidate) AS uuid_matches
            WHERE size(uuid_matches) <= 1
            OPTIONAL MATCH (source_candidate {{uuid: $edge.source_node_uuid}})
            WITH uuid_matches, collect(source_candidate) AS sources
            WHERE size(sources) = 1 AND 'Entity' IN labels(sources[0])
            WITH uuid_matches, sources[0] AS source
            OPTIONAL MATCH (target_candidate {{uuid: $edge.target_node_uuid}})
            WITH uuid_matches, source, collect(target_candidate) AS targets
            WHERE size(targets) = 1 AND 'Entity' IN labels(targets[0]) AND
                  source.group_id = $edge.group_id AND
                  targets[0].group_id = $edge.group_id
            WITH uuid_matches, source, targets[0] AS target
            OPTIONAL MATCH (source)-[typed]->(target)
            WHERE type(typed) = $edge.name
            WITH uuid_matches, source, target, collect(typed) AS typed_matches
            WHERE size(typed_matches) <= 1
            WITH uuid_matches, source, target,
                 CASE WHEN size(typed_matches) = 1 THEN typed_matches[0] ELSE NULL END AS existing
            WITH uuid_matches, source, target, existing,
                 existing.uuid AS prior_uuid, existing._revision AS prior_revision
            WHERE (size(uuid_matches) = 0 OR
                   (existing IS NOT NULL AND uuid_matches[0] = existing)) AND
                  (existing IS NULL OR
                   (existing.group_id = $edge.group_id AND existing.name = $edge.name AND
                    (prior_revision IS NULL OR
                     (prior_revision >= 1 AND
                      toString(prior_revision) = toString(toInteger(prior_revision))))))
            WITH source, target, existing, prior_uuid, prior_revision,
                 CASE WHEN existing IS NULL OR prior_uuid IS NULL OR prior_uuid <> $edge.uuid
                      THEN true
                      ELSE NOT (
                          all(property_key IN [key IN keys(existing)
                               WHERE key <> '_revision' AND key <> 'fact_embedding']
                              WHERE property_key IN keys($edge) AND
                                    CASE WHEN $edge[property_key] IS NULL THEN false
                                         ELSE existing[property_key] = $edge[property_key]
                                    END) AND
                          all(property_key IN keys($edge)
                              WHERE property_key = 'fact_embedding' OR
                                    $edge[property_key] IS NULL OR
                                    property_key IN keys(existing)) AND
                          ((existing.fact_embedding IS NULL AND $edge.fact_embedding IS NULL) OR
                           (existing.fact_embedding IS NOT NULL AND
                            $edge.fact_embedding IS NOT NULL AND
                            existing.fact_embedding = vecf32($edge.fact_embedding)))
                      )
                 END AS changed
            WITH source, target, existing, prior_uuid, changed,
                 CASE WHEN existing IS NULL OR prior_uuid IS NULL OR prior_uuid <> $edge.uuid
                      THEN 1
                      WHEN changed THEN coalesce(prior_revision, 1) + 1
                      ELSE prior_revision END AS next_revision
            FOREACH (_ IN CASE WHEN changed AND existing IS NULL THEN [1] ELSE [] END |
                MERGE (source)-[created:{edge_type}]->(target)
                SET created = $edge,
                    created.fact_embedding = vecf32($edge.fact_embedding),
                    created._revision = next_revision)
            FOREACH (_ IN CASE WHEN changed AND existing IS NOT NULL THEN [1] ELSE [] END |
                SET existing = $edge,
                    existing.fact_embedding = vecf32($edge.fact_embedding),
                    existing._revision = next_revision)
            WITH source, target, changed, prior_uuid
            OPTIONAL MATCH (source)-[saved]->(target)
            WHERE type(saved) = $edge.name AND saved.uuid = $edge.uuid
            WITH changed, prior_uuid, collect(saved) AS saved_matches
            WHERE size(saved_matches) = 1
            RETURN saved_matches[0].uuid AS uuid, saved_matches[0]._revision AS revision,
                   changed AS changed,
                   CASE WHEN prior_uuid <> $edge.uuid THEN prior_uuid ELSE NULL END AS previous_uuid
        """

    # Sanitize edge_type to prevent injection (only allow alphanumeric and underscore)
    safe_edge_type = ''.join(c for c in edge_type if c.isalnum() or c == '_')
    if not safe_edge_type:
        safe_edge_type = 'RELATES_TO'

    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[r:{safe_edge_type}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[r:{safe_edge_type}]->(target)
                SET r = removeKeyFromMap(removeKeyFromMap(edge, "fact_embedding"), "episodes")
                SET r.fact_embedding = join([x IN coalesce(edge.fact_embedding, []) | toString(x) ], ",")
                SET r.episodes = join(edge.episodes, ",")
                RETURN edge.uuid AS uuid
            """
        case _:
            # For Neo4j and others
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[e:{safe_edge_type}]->(target)
                SET e += edge
                WITH e, edge
                CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
                RETURN edge.uuid AS uuid
            """


def get_entity_edge_return_query(provider: GraphProvider) -> str:
    # `fact_embedding` is not returned by default and must be manually loaded using `load_fact_embedding()`.

    if provider == GraphProvider.NEPTUNE:
        return """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.name AS name,
        e.fact AS fact,
        split(e.episodes, ',') AS episodes,
        e.created_at AS created_at,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes
    """

    return """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
    """ + (
        'e.attributes AS attributes'
        if provider == GraphProvider.KUZU
        else 'properties(e) AS attributes'
    )


def get_community_edge_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                WHERE node:Entity OR node:Community
                MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
                SET r.uuid= $uuid
                SET r.group_id= $group_id
                SET r.created_at= $created_at
                RETURN r.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Entity {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at
                RETURN e.uuid AS uuid
                UNION
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Community {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at
                RETURN e.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Entity | Community {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """


COMMUNITY_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""
