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

import asyncio
import datetime
import json
import logging
import re
from typing import TYPE_CHECKING, Any

# Import storage classes
from graphiti_core.storage.content_storage import ContentStorage, StorageResult
from graphiti_core.storage.local_storage import LocalFileStorage
from graphiti_core.storage.redis_storage import RedisContentStorage

if TYPE_CHECKING:
    from falkordb import Graph as FalkorGraph
    from falkordb.asyncio import FalkorDB
else:
    try:
        from falkordb import Graph as FalkorGraph
        from falkordb.asyncio import FalkorDB
    except ImportError:
        # If falkordb is not installed, raise an ImportError
        raise ImportError(
            'falkordb is required for FalkorDriver. '
            'Install it with: pip install graphiti-core[falkordb]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

logger = logging.getLogger(__name__)

STOPWORDS = [
    'a',
    'is',
    'the',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'for',
    'if',
    'in',
    'into',
    'it',
    'no',
    'not',
    'of',
    'on',
    'or',
    'such',
    'that',
    'their',
    'then',
    'there',
    'these',
    'they',
    'this',
    'to',
    'was',
    'will',
    'with',
]


class FalkorDriverSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self, graph: FalkorGraph):
        self.graph = graph
        # Log graph name for debugging
        graph_name = getattr(graph, 'name', getattr(graph, '_name', 'unknown'))
        logger.debug(f'[FalkorDB] Created session for graph: {graph_name}')

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        graph_name = getattr(self.graph, 'name', getattr(self.graph, '_name', 'unknown'))
        if isinstance(query, list):
            for cypher, params in query:
                params = convert_datetimes_to_strings(params)
                logger.debug(f'[FalkorDB] graph={graph_name}, query={cypher[:200]}..., params_keys={list(params.keys()) if params else []}')
                await self.graph.query(str(cypher), params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            # Validate query is not empty
            if not query or not query.strip():
                params_preview = {k: f'<{type(v).__name__}:{len(v) if isinstance(v, (list, dict, str)) else v}>' for k, v in params.items()} if params else {}
                logger.error(f'[FalkorDB] Empty query detected! graph={graph_name}, params={params_preview}')
                raise ValueError(f'Empty Cypher query for graph {graph_name}')
            try:
                await self.graph.query(str(query), params)  # type: ignore[reportUnknownArgumentType]
            except Exception as e:
                # Log query details on error for debugging
                query_preview = str(query)[:300] if query else '<empty>'
                params_preview = {k: f'<{type(v).__name__}:{len(v) if isinstance(v, (list, dict, str)) else v}>' for k, v in params.items()} if params else {}
                logger.error(f'[FalkorDB] Query failed! graph={graph_name}, error={e}, query={query_preview}, params={params_preview}')
                raise
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None

    async def run_with_result(self, query: str, **kwargs: Any) -> tuple[list[dict], list[str], None]:
        if not query or not query.strip():
            raise ValueError('Empty Cypher query')
        result = await self.graph.query(str(query), convert_datetimes_to_strings(dict(kwargs)))
        header = [column[1] for column in result.header]
        records = [dict(zip(header, row)) for row in result.result_set]
        return records, header, None


class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    default_group_id: str = '\\_'
    fulltext_syntax: str = '@'  # FalkorDB uses a redisearch-like syntax for fulltext queries
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = 'default_db',
        embedding_dimension: int = 1024,
        # NEW: Content storage configuration
        content_storage_type: str = 'local',  # redis | local | oss | s3
        content_storage_config: dict | None = None,
        # Flag to skip index building (used for cloned drivers)
        skip_index_build: bool = False,
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) FalkorDB instance.

        Args:
        host (str): The host where FalkorDB is running.
        port (int): The port on which FalkorDB is listening.
        username (str | None): The username for authentication (if required).
        password (str | None): The password for authentication (if required).
        falkor_db (FalkorDB | None): An existing FalkorDB instance to use instead of creating a new one.
        database (str): The name of the database to connect to. Defaults to 'default_db'.
        embedding_dimension (int): Vector embedding dimension. Defaults to 1024 (qwen3-embedding).
        content_storage_type (str): Type of content storage (redis, local, oss, s3). Defaults to 'local'.
        content_storage_config (dict | None): Configuration dict for content storage backend.
        skip_index_build (bool): Skip building indices (used for cloned drivers). Defaults to False.
        """
        super().__init__()
        # Ensure database is not empty (FalkorDB 1.2.0 requires non-empty database name)
        if not database or not database.strip():
            logger.warning(f"Empty database name provided, using default 'default_db'")
            database = 'default_db'
        self._database = database
        self._embedding_dimension = embedding_dimension
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

        # NEW: Initialize content storage
        self.content_storage_type = content_storage_type
        self.content_storage = self._create_storage(content_storage_type, content_storage_config or {})

        # Schedule the indices and constraints to be built (skip for cloned drivers)
        if not skip_index_build:
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # Schedule the build_indices_and_constraints to run
                loop.create_task(self.build_indices_and_constraints())
            except RuntimeError:
                # No event loop running, this will be handled later
                pass

    def _create_storage(self, storage_type: str, config: dict) -> ContentStorage:
        """
        Factory method: create storage instance based on type

        Args:
            storage_type: Storage type (redis, local, oss, s3)
            config: Configuration dict for the storage backend

        Returns:
            ContentStorage instance

        Raises:
            ValueError: If storage_type is unknown
        """
        if storage_type == 'redis':
            return RedisContentStorage()
        elif storage_type == 'local':
            base_path = config.get('base_path', './data/episodes')
            return LocalFileStorage(base_path=base_path)
        elif storage_type == 'oss':
            # TODO: Implement OSS storage
            raise NotImplementedError('OSS storage is not yet implemented')
        elif storage_type == 's3':
            # TODO: Implement S3 storage
            raise NotImplementedError('S3 storage is not yet implemented')
        else:
            raise ValueError(f'Unknown storage type: {storage_type}')

    def _get_graph(self, graph_name: str | None) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "default_db"
        if graph_name is None:
            graph_name = self._database
        logger.debug(f'[FalkorDB] _get_graph: requested={graph_name}, default={self._database}')
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any):
        graph = self._get_graph(self._database)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))

        try:
            result = await graph.query(cypher_query_, params)  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            if 'already indexed' in str(e):
                # check if index already exists
                logger.info(f'Index already exists: {e}')
                return None
            logger.error(f'Error executing FalkorDB query: {e}\n{cypher_query_}\n{params}')
            raise

        # Convert the result header to a list of strings
        header = [h[1] for h in result.header]

        # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
        records = []
        for row in result.result_set:
            record = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    record[field_name] = row[i]
                else:
                    # If there are more fields in header than values in row, set to None
                    record[field_name] = None
            records.append(record)

        return records, header, None

    @staticmethod
    def _graph_write_identifier(value: Any, field: str) -> str:
        """Validate an identifier before it is interpolated into Cypher."""
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
            raise ValueError(f'Invalid graph-write {field}')
        return value

    @staticmethod
    def _graph_write_properties(value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f'Graph-write {field} must be an object')
        reserved = {
            'uuid', 'group_id', '_revision', '_action_execution_id', '_updated_at',
            'source_uuid', 'target_uuid', 'valid_at', 'invalid_at', 'expired_at',
            'created_at', 'updated_at', 'episodes', 'name_embedding', 'summary_embedding', 'fact_embedding',
            'labels', 'source_node_uuid', 'target_node_uuid',
            '_previous_edge_uuid',
        }
        if reserved.intersection(value):
            raise ValueError(f'Graph-write {field} contains reserved properties')
        # A JSON round-trip rejects non-primitive values before they reach FalkorDB.
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Graph-write {field} is not JSON serializable') from error
        return value

    @classmethod
    def _graph_write_query(cls, request: Any, context: Any) -> tuple[str, dict[str, Any]]:
        """Build one guarded Cypher query for a structured graph write.

        The query deliberately uses only fixed graph primitives (``Entity`` and
        ``RELATES_TO``). Logical entity/edge names are values in the request; the
        only interpolated labels have been validated by ``_graph_write_identifier``.
        All checks happen before the write ``FOREACH`` clauses, so a rejected plan
        executes no mutation and a runtime FalkorDB error rolls the query back.
        """
        if not hasattr(request, 'model_dump'):
            raise ValueError('Structured graph-write request is required')
        payload = request.model_dump(mode='python')
        edits = payload.get('edits')
        state = payload.get('expected_state') or {}
        if not isinstance(edits, list) or not edits:
            raise ValueError('Graph-write request must contain edits')
        group_id = getattr(context, 'group_id', None)
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError('Graph-write tenant is required')
        person_id = getattr(context, 'person_id', None)
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError('Graph-write authenticated person is required')

        entity_ids: list[str] = list(state.get('absent_ids') or [])
        edge_ids: list[str] = list(state.get('absent_ids') or [])
        created_entity_ids = [edit['uuid'] for edit in edits
                              if edit.get('kind') == 'entity' and edit.get('operation') == 'create']
        for edit in edits:
            if edit.get('kind') == 'entity':
                entity_ids.append(edit['uuid'])
            elif edit.get('kind') == 'edge':
                edge_ids.append(edit['uuid'])
            else:
                raise ValueError('Unsupported graph-write resource kind')
            for endpoint in ('source_id', 'target_id'):
                if edit.get(endpoint):
                    entity_ids.append(edit[endpoint])
        for snapshot in (state.get('snapshots') or {}).values():
            if snapshot.get('kind') == 'entity':
                entity_ids.append(snapshot['uuid'])
            elif snapshot.get('kind') == 'edge':
                edge_ids.append(snapshot['uuid'])
                entity_ids.extend([snapshot['source_id'], snapshot['target_id']])
        for dependency in state.get('relation_dependencies') or []:
            entity_ids.extend([dependency['source_id'], dependency['target_id']])
            for member in dependency.get('expected_matches') or []:
                edge_ids.append(member['uuid'])
        endpoint_ids = list(dict.fromkeys(
            item[endpoint]
            for item in (
                *[edit for edit in edits if edit['kind'] == 'edge'],
                *[snapshot for snapshot in (state.get('snapshots') or {}).values()
                  if snapshot['kind'] == 'edge'],
                *(state.get('relation_dependencies') or []),
                *(state.get('unique_active_edges') or []),
            )
            for endpoint in ('source_id', 'target_id')
        ))
        entity_ids.extend(endpoint_ids)
        entity_ids = list(dict.fromkeys(entity_ids))
        edge_ids = list(dict.fromkeys(edge_ids))

        params: dict[str, Any] = {
            'group_id': group_id,
            'person_id': person_id,
            'request_id': payload['request_id'],
            'source_ref': payload['source_ref'],
            'request_hash': '',
            'entity_ids': entity_ids,
            'edge_ids': edge_ids,
            'created_entity_ids': created_entity_ids,
            'now': (payload['effective_at'].isoformat() if payload.get('effective_at') is not None
                    else datetime.datetime.now(datetime.UTC).isoformat()),
            'snapshot_reserved_entity_keys': [
                'uuid', 'group_id', '_revision', '_action_execution_id', '_updated_at',
                'source_uuid', 'target_uuid', 'valid_at', 'invalid_at', 'expired_at',
                'created_at', 'name', 'summary', 'name_embedding', 'summary_embedding', 'labels',
            ],
            'snapshot_reserved_edge_keys': [
                'uuid', 'group_id', 'name', '_revision', '_action_execution_id', '_updated_at',
                'source_uuid', 'target_uuid', 'valid_at', 'invalid_at', 'expired_at',
                'created_at', 'updated_at', 'fact', 'fact_embedding', 'episodes',
                'source_node_uuid', 'target_node_uuid',
                '_previous_edge_uuid',
            ],
        }
        property_keys: dict[int, str] = {}
        for index, edit in enumerate(edits):
            if edit['operation'] != 'delete':
                property_keys[index] = f'props_{index}'
                params[property_keys[index]] = cls._graph_write_properties(
                    edit.get('properties') or {}, f'edit {edit["uuid"]}'
                )
        clauses = [
            'OPTIONAL MATCH (prior_receipt:ActionGraphWriteReceipt) '
            'WHERE prior_receipt.group_id = $group_id AND prior_receipt.request_id = $request_id',
            'WITH [item IN collect(prior_receipt) WHERE item IS NOT NULL] AS prior_receipts',
            'OPTIONAL MATCH (node_candidate) WHERE node_candidate.uuid IN $entity_ids',
            'WITH prior_receipts, collect(node_candidate) AS node_candidates',
            'OPTIONAL MATCH (edge_source)-[edge_candidate]->(edge_target) '
            'WHERE edge_candidate.uuid IN $edge_ids',
            "WITH prior_receipts, node_candidates, [item IN collect(CASE WHEN edge_candidate IS NULL THEN NULL "
            "ELSE {uuid: edge_candidate.uuid, edge: edge_candidate, source_uuid: edge_source.uuid, "
            "target_uuid: edge_target.uuid, source: edge_source, target: edge_target, "
            "logical_type: coalesce(edge_candidate.name, type(edge_candidate))} END) "
            'WHERE item IS NOT NULL] AS edge_candidates',
            'OPTIONAL MATCH (incident_source)-[incident_edge]-(incident_target) '
            'WHERE (incident_source.uuid IN $entity_ids OR incident_target.uuid IN $entity_ids) '
            'AND incident_edge.invalid_at IS NULL AND incident_edge.expired_at IS NULL '
            "AND 'Entity' IN labels(incident_source) AND 'Entity' IN labels(incident_target)",
            "WITH prior_receipts, node_candidates, edge_candidates, [item IN collect(CASE WHEN incident_edge IS NULL THEN NULL "
            "ELSE {source_uuid: incident_source.uuid, target_uuid: incident_target.uuid} END) "
            'WHERE item IS NOT NULL] AS incident_edges',
            'OPTIONAL MATCH (active_source)-[active_edge]->(active_target) '
            'WHERE active_edge.group_id = $group_id AND active_edge.invalid_at IS NULL '
            'AND active_edge.expired_at IS NULL',
            "WITH prior_receipts, node_candidates, edge_candidates, incident_edges, [item IN collect(CASE WHEN active_edge IS NULL THEN NULL "
            "ELSE {uuid: active_edge.uuid, source_uuid: active_source.uuid, target_uuid: active_target.uuid, "
            "source: active_source, target: active_target, "
            "logical_type: coalesce(active_edge.name, type(active_edge)), edge: active_edge} END) "
            'WHERE item IS NOT NULL] AS active_edges',
        ]

        def node_expr(uuid: str) -> str:
            return (
                f'head([n IN node_candidates WHERE n.uuid = {json.dumps(uuid)} '
                f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)])'
            )

        def edge_expr(uuid: str) -> str:
            return f'head([item IN edge_candidates WHERE item.uuid = {json.dumps(uuid)}])'

        def edge_matches_expr(type_name: str, source_id: str, target_id: str, parameter: str | None = None) -> str:
            """Return a bounded active-edge selection expression.

            The relation dependency is evaluated against the endpoint-bound edge
            objects collected by the same query. UUID strings alone are not enough:
            a non-Entity node may share an UUID with a legitimate endpoint.
            """
            if parameter is None:
                return (
                    f'[item IN active_edges WHERE item.logical_type = {json.dumps(type_name)} '
                    f'AND item.source_uuid = {json.dumps(source_id)} '
                    f'AND item.target_uuid = {json.dumps(target_id)} '
                    f'AND item.source = {node_expr(source_id)} '
                    f'AND item.target = {node_expr(target_id)}]'
                )
            return (
                f'[item IN active_edges WHERE item.logical_type = {parameter}.type_name '
                f'AND item.source_uuid = {parameter}.source_id '
                f'AND item.target_uuid = {parameter}.target_id '
                f'AND item.source = {node_expr(source_id)} '
                f'AND item.target = {node_expr(target_id)}]'
            )

        gates: list[str] = []
        revisions = state.get('revisions') or {}
        snapshots = state.get('snapshots') or {}
        absent_ids = set(state.get('absent_ids') or [])
        isolated_ids = set(state.get('isolated_entity_ids') or [])
        endpoint_map = state.get('expected_edge_endpoints') or {}
        for snapshot_index, snapshot in enumerate(snapshots.values()):
            params[f'snapshot_props_{snapshot_index}'] = cls._graph_write_properties(
                snapshot.get('properties') or {}, f'snapshot {snapshot["uuid"]}'
            )
            params[f'snapshot_keys_{snapshot_index}'] = list(
                params[f'snapshot_props_{snapshot_index}'].keys()
            )
            params[f'snapshot_valid_{snapshot_index}'] = snapshot.get('valid_at')
            params[f'snapshot_invalid_{snapshot_index}'] = snapshot.get('invalid_at')
            params[f'snapshot_expired_{snapshot_index}'] = snapshot.get('expired_at')
            if snapshot.get('created_at') is not None:
                params[f'snapshot_created_{snapshot_index}'] = snapshot['created_at'].isoformat()
        for edit in edits:
            uuid = edit['uuid']
            if edit['operation'] == 'create':
                gates.append(
                    f'size([n IN node_candidates WHERE n.uuid = {json.dumps(uuid)}]) = 0 '
                    f'AND size([item IN edge_candidates WHERE item.uuid = {json.dumps(uuid)}]) = 0'
                )
                if uuid not in absent_ids:
                    gates.append('false')
            else:
                if edit['kind'] == 'entity':
                    current = node_expr(uuid)
                    gates.extend([
                        f'size([n IN node_candidates WHERE n.uuid = {json.dumps(uuid)}]) = 1',
                        f'coalesce({current}.group_id, "") = $group_id',
                        f'{json.dumps(edit["type_name"])} IN labels({current})',
                    ])
                    if uuid in revisions:
                        gates.append(f'typeOf({current}._revision) = "Integer"')
                        gates.append(f'coalesce({current}._revision, -1) = {int(revisions[uuid])}')
                    if uuid in isolated_ids:
                        gates.append(
                            f'size([incident IN incident_edges WHERE incident.source_uuid = {json.dumps(uuid)} '
                            f'OR incident.target_uuid = {json.dumps(uuid)}]) = 0'
                        )
                else:
                    current = edge_expr(uuid)
                    gates.extend([
                        f'size([item IN edge_candidates WHERE item.uuid = {json.dumps(uuid)}]) = 1',
                        f'coalesce({current}.edge.group_id, "") = $group_id',
                        f'{current}.logical_type = {json.dumps(edit["type_name"])}',
                        f'{current}.source_uuid = {json.dumps(edit["source_id"])}',
                        f'{current}.target_uuid = {json.dumps(edit["target_id"])}',
                        f'{current}.source = {node_expr(edit["source_id"])}',
                        f'{current}.target = {node_expr(edit["target_id"])}',
                        f'{current}.edge.invalid_at IS NULL',
                        f'(size([n IN node_candidates WHERE n.uuid = {json.dumps(edit["source_id"])} '
                        f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)]) = 1 '
                        f'OR {json.dumps(edit["source_id"])} IN $created_entity_ids)',
                        f'(size([n IN node_candidates WHERE n.uuid = {json.dumps(edit["target_id"])} '
                        f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)]) = 1 '
                        f'OR {json.dumps(edit["target_id"])} IN $created_entity_ids)',
                    ])
                    if uuid in revisions:
                        gates.append(f'typeOf({current}.edge._revision) = "Integer"')
                        gates.append(f'coalesce({current}.edge._revision, -1) = {int(revisions[uuid])}')
                    expected = endpoint_map.get(uuid)
                    if expected and (expected.get('source_id'), expected.get('target_id')) != (
                        edit.get('source_id'), edit.get('target_id')
                    ):
                        gates.append('false')

        # A relation dependency is a complete server-generated result, including
        # an explicit empty result. Re-resolve it in this query so an unchanged
        # endpoint revision cannot hide a relation insert/delete/replace.
        for dependency_index, dependency in enumerate(state.get('relation_dependencies') or []):
            member = (dependency.get('expected_matches') or [None])[0]
            parameter = f'$relation_dependency_{dependency_index}'
            params[f'relation_dependency_{dependency_index}'] = {
                'type_name': dependency['type_name'],
                'source_id': dependency['source_id'],
                'target_id': dependency['target_id'],
                'expected_uuid': member.get('uuid') if member else None,
                'expected_revision': member.get('revision') if member else None,
            }
            source_id = dependency['source_id']
            target_id = dependency['target_id']
            relation_matches = edge_matches_expr(
                dependency['type_name'], source_id, target_id,
                parameter=parameter,
            )
            gates.extend([
                f'size([n IN node_candidates WHERE n.uuid = {json.dumps(source_id)} '
                f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)]) = 1',
                f'size([n IN node_candidates WHERE n.uuid = {json.dumps(target_id)} '
                f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)]) = 1',
            ])
            if member is None:
                gates.append(f'size({relation_matches}) = 0')
            else:
                gates.extend([
                    f'size({relation_matches}) = 1',
                    f'head({relation_matches}).uuid = {parameter}.expected_uuid',
                    f'typeOf(head({relation_matches}).edge._revision) = "Integer"',
                    f'coalesce(head({relation_matches}).edge._revision, -1) = '
                    f'{parameter}.expected_revision',
                ])

        for uuid, expected in revisions.items():
            if any(edit['uuid'] == uuid for edit in edits):
                continue
            if uuid in snapshots:
                continue
            gates.append('false')
        for uuid, snapshot in snapshots.items():
            expected_revision = snapshot.get('revision') or revisions.get(uuid)
            snapshot_index = list(snapshots).index(uuid)
            if snapshot.get('kind') == 'entity':
                current = node_expr(uuid)
                gates.extend([
                    f'size([n IN node_candidates WHERE n.uuid = {json.dumps(uuid)}]) = 1',
                    f'coalesce({current}.group_id, "") = $group_id',
                    f'{json.dumps(snapshot["type_name"])} IN labels({current})',
                    f'all(key IN keys($snapshot_props_{snapshot_index}) WHERE '
                    f'({current}[key] = $snapshot_props_{snapshot_index}[key] OR '
                    f'({current}[key] IS NULL AND $snapshot_props_{snapshot_index}[key] IS NULL)))',
                    f'all(key IN keys({current}) WHERE key IN $snapshot_keys_{snapshot_index} '
                    f'OR {current}[key] IS NULL '
                    f'OR key IN $snapshot_reserved_entity_keys)',
                ])
                if expected_revision is not None:
                    gates.append(f'typeOf({current}._revision) = "Integer"')
                    gates.append(f'{current}._revision = {int(expected_revision)}')
                if snapshot.get('created_at') is not None:
                    gates.append(f'{current}.created_at = $snapshot_created_{snapshot_index}')
            else:
                current = edge_expr(uuid)
                gates.extend([
                    f'size([item IN edge_candidates WHERE item.uuid = {json.dumps(uuid)}]) = 1',
                    f'coalesce({current}.edge.group_id, "") = $group_id',
                    f'{current}.logical_type = {json.dumps(snapshot["type_name"])}',
                    f'{current}.source_uuid = {json.dumps(snapshot["source_id"])}',
                    f'{current}.target_uuid = {json.dumps(snapshot["target_id"])}',
                    f'{current}.source = {node_expr(snapshot["source_id"])}',
                    f'{current}.target = {node_expr(snapshot["target_id"])}',
                    f'({current}.edge.valid_at = $snapshot_valid_{snapshot_index} OR '
                    f'({current}.edge.valid_at IS NULL AND $snapshot_valid_{snapshot_index} IS NULL))',
                    f'({current}.edge.invalid_at = $snapshot_invalid_{snapshot_index} OR '
                    f'({current}.edge.invalid_at IS NULL AND $snapshot_invalid_{snapshot_index} IS NULL))',
                    f'({current}.edge.expired_at = $snapshot_expired_{snapshot_index} OR '
                    f'({current}.edge.expired_at IS NULL AND $snapshot_expired_{snapshot_index} IS NULL))',
                    f'all(key IN keys($snapshot_props_{snapshot_index}) WHERE '
                    f'({current}.edge[key] = $snapshot_props_{snapshot_index}[key] OR '
                    f'({current}.edge[key] IS NULL AND $snapshot_props_{snapshot_index}[key] IS NULL)))',
                    f'all(key IN keys({current}.edge) WHERE key IN $snapshot_keys_{snapshot_index} '
                    f'OR {current}.edge[key] IS NULL '
                    f'OR key IN $snapshot_reserved_edge_keys)',
                ])
                if expected_revision is not None:
                    gates.append(f'typeOf({current}.edge._revision) = "Integer"')
                    gates.append(f'{current}.edge._revision = {int(expected_revision)}')
                if snapshot.get('created_at') is not None:
                    gates.append(f'{current}.edge.created_at = $snapshot_created_{snapshot_index}')

        for selection_index, selection in enumerate(state.get('unique_active_edges') or []):
            params[f'sel_{selection_index}'] = selection
            sel = f'$sel_{selection_index}'
            match = (
                f'[item IN active_edges WHERE item.logical_type = {sel}.type_name '
                f'AND item.source_uuid = {sel}.source_id AND item.target_uuid = {sel}.target_id '
                f'AND item.source = {node_expr(selection["source_id"])} '
                f'AND item.target = {node_expr(selection["target_id"])}]'
            )
            gates.append(f'size({match}) = 1' if selection.get('expected_uuid') else f'size({match}) = 0')
            if selection.get('expected_uuid'):
                gates.append(f'head({match}).uuid = {sel}.expected_uuid')
        for uuid in state.get('isolated_entity_ids') or []:
            gates.append(
                f'size([incident IN incident_edges WHERE incident.source_uuid = {json.dumps(uuid)} '
                f'OR incident.target_uuid = {json.dumps(uuid)}]) = 0'
            )
        for endpoint_id in endpoint_ids:
            gates.append(
                f'((size([n IN node_candidates WHERE n.uuid = {json.dumps(endpoint_id)}]) = 1 '
                f'AND size([n IN node_candidates WHERE n.uuid = {json.dumps(endpoint_id)} '
                f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)]) = 1 '
                f') OR {json.dumps(endpoint_id)} IN $created_entity_ids)'
            )
        # Every absence assertion is checked against both node and edge UUIDs
        # before the CREATE branches execute.
        for absent_id in absent_ids:
            gates.append(
                f'size([n IN node_candidates WHERE n.uuid = {json.dumps(absent_id)}]) = 0'
            )
            gates.append(
                f'size([item IN edge_candidates WHERE item.uuid = {json.dumps(absent_id)}]) = 0'
            )
        if not gates:
            gates.append('false')
        clauses.append(
            'WITH *, CASE '
            "WHEN size(prior_receipts) > 1 THEN 'HASH_CONFLICT' "
            "WHEN size(prior_receipts) = 1 AND prior_receipts[0].request_hash <> $request_hash THEN 'HASH_CONFLICT' "
            "WHEN size(prior_receipts) = 1 AND ("
            "coalesce(prior_receipts[0].request_hash, '') <> $request_hash OR "
            "coalesce(prior_receipts[0].person_id, '') <> $person_id OR "
            "coalesce(prior_receipts[0].source_ref, '') <> $source_ref OR "
            "coalesce(prior_receipts[0].status, '') <> 'APPLIED') THEN 'REJECTED' "
            "WHEN size(prior_receipts) = 1 THEN 'REPLAYED' "
            "WHEN " + ' AND '.join(gates) + " THEN 'READY' ELSE 'REJECTED' END AS gate_status"
        )
        # Create entities first so a relation in the same request may target a new entity.
        for index, edit in enumerate(edits):
            if edit['kind'] != 'entity' or edit['operation'] != 'create':
                continue
            label = cls._graph_write_identifier(edit['type_name'], 'entity type')
            prop_key = property_keys[index]
            clauses.append(
                f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | '
                f'CREATE (created_{index}:Entity:{label}) SET created_{index} = ${prop_key}, '
                f'created_{index}.uuid = {json.dumps(edit["uuid"])}, created_{index}.group_id = $group_id, '
                f'created_{index}.name = coalesce(created_{index}.name, {json.dumps(edit["type_name"])}), '
                f'created_{index}.summary = coalesce(created_{index}.summary, ""), '
                f'created_{index}.created_at = $now, '
                f'created_{index}._revision = 1, created_{index}._action_execution_id = $request_id, '
                f'created_{index}._updated_at = $now)'
            )
        clauses.extend([
            'WITH gate_status, prior_receipts, node_candidates, edge_candidates, incident_edges, active_edges',
            'OPTIONAL MATCH (post_node) WHERE post_node.uuid IN $entity_ids',
            'WITH gate_status, prior_receipts, node_candidates, edge_candidates, incident_edges, active_edges, '
            '[item IN collect(post_node) WHERE item IS NOT NULL] AS post_nodes',
        ])

        def post_node_expr(uuid: str) -> str:
            return (
                f'head([n IN post_nodes WHERE n.uuid = {json.dumps(uuid)} '
                f'AND n.group_id = $group_id AND \'Entity\' IN labels(n)])'
            )

        # Bind existing nodes/edges after entity creation; all remaining writes remain gated.
        aliases: list[str] = []
        for index, edit in enumerate(edits):
            if edit['kind'] == 'entity' and edit['operation'] != 'create':
                alias = f'entity_{index}'
                clauses.append(f'WITH *, {post_node_expr(edit["uuid"])} AS {alias}')
                aliases.append(alias)
            elif edit['kind'] == 'edge':
                alias = f'edge_{index}'
                clauses.append(
                    f'WITH *, head([item IN edge_candidates WHERE item.uuid = {json.dumps(edit["uuid"])}]).edge AS {alias}'
                )
                aliases.append(alias)
        # Persist the idempotency receipt inside the same graph query. A retry
        # with the same request/hash reaches REPLAYED above; a different hash is
        # rejected before this MERGE is reached.
        clauses.append(
            "FOREACH (_ IN CASE WHEN gate_status = 'READY' THEN [1] ELSE [] END | "
            "MERGE (write_receipt:ActionGraphWriteReceipt {group_id: $group_id, request_id: $request_id}) "
            "SET write_receipt.request_hash = $request_hash, write_receipt.status = 'APPLIED', "
            "write_receipt.source_ref = $source_ref, write_receipt.person_id = $person_id, "
            "write_receipt.created_at = $now)"
        )
        for index, edit in enumerate(edits):
            if edit['kind'] == 'entity' and edit['operation'] != 'create':
                alias = f'entity_{index}'
                if edit['operation'] == 'update':
                    prop_key = property_keys[index]
                    clauses.append(f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | SET {alias} += ${prop_key}, {alias}._revision = coalesce({alias}._revision, 1) + 1, {alias}._action_execution_id = $request_id, {alias}._updated_at = $now)')
                else:
                    clauses.append(f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | DELETE {alias})')
            elif edit['kind'] == 'edge':
                alias = f'edge_{index}'
                if edit['operation'] == 'update':
                    prop_key = property_keys[index]
                    clauses.append(f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | SET {alias} += ${prop_key}, {alias}._revision = coalesce({alias}._revision, 1) + 1, {alias}._action_execution_id = $request_id, {alias}._updated_at = $now)')
                elif edit.get('delete_mode') == 'expire':
                    clauses.append(f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | SET {alias}.invalid_at = $now, {alias}.expired_at = $now, {alias}._revision = coalesce({alias}._revision, 1) + 1, {alias}._action_execution_id = $request_id, {alias}._updated_at = $now)')
                elif edit.get('delete_mode') == 'hard':
                    raise ValueError('Hard edge delete is not supported by the atomic graph writer')

        # Edge creation uses post-write endpoint nodes and the logical edge name.
        for index, edit in enumerate(edits):
            if edit['kind'] != 'edge' or edit['operation'] != 'create':
                continue
            prop_key = property_keys[index]
            source = post_node_expr(edit['source_id'])
            target = post_node_expr(edit['target_id'])
            clauses.append(f'WITH *, {source} AS create_source_{index}, {target} AS create_target_{index}')
            previous_uuid = edit.get('previous_edge_uuid')
            new_revision = 1
            lineage = ''
            if previous_uuid is not None:
                previous_snapshot = snapshots.get(previous_uuid) or {}
                previous_revision = revisions.get(previous_uuid) or previous_snapshot.get('revision')
                if type(previous_revision) is not int or previous_revision < 1:
                    raise ValueError('Replacement revision is required')
                new_revision = previous_revision + 1
                lineage = f'created_edge_{index}._previous_edge_uuid = {json.dumps(previous_uuid)}, '
            clauses.append(
                f'FOREACH (_ IN CASE WHEN gate_status = \'READY\' THEN [1] ELSE [] END | '
                f'CREATE (create_source_{index})-[created_edge_{index}:RELATES_TO]->(create_target_{index}) '
                f'SET created_edge_{index} = ${prop_key}, created_edge_{index}.uuid = {json.dumps(edit["uuid"])}, '
                f'created_edge_{index}.name = {json.dumps(edit["type_name"])}, created_edge_{index}.group_id = $group_id, '
                f'created_edge_{index}.fact = coalesce(created_edge_{index}.fact, {json.dumps(edit["type_name"])}), '
                f'created_edge_{index}.episodes = [], created_edge_{index}.created_at = $now, '
                f'created_edge_{index}.valid_at = $now, created_edge_{index}._revision = {new_revision}, '
                f'{lineage}'
                f'created_edge_{index}._action_execution_id = $request_id, created_edge_{index}._updated_at = $now)'
            )
        clauses.append(
            "RETURN $request_id AS request_id, $group_id AS group_id, $request_hash AS request_hash, "
            "gate_status, CASE WHEN gate_status = 'READY' THEN 'APPLIED' "
            "WHEN gate_status = 'REPLAYED' THEN 'REPLAYED' ELSE 'REJECTED' END AS status, "
            f"CASE WHEN gate_status = 'READY' THEN {len(edits)} ELSE 0 END AS planned_changes, "
            "CASE WHEN gate_status = 'HASH_CONFLICT' THEN 'GRAPH_WRITE_HASH_CONFLICT' "
            "ELSE NULL END AS error_code"
        )
        return '\n'.join(clauses), params

    async def execute_atomic_graph_write(
        self,
        *,
        context: Any,
        request: Any,
        request_hash: str,
        required_read_fields: Any,
    ) -> dict[str, Any]:
        """Execute a validated structured graph write in exactly one FalkorDB query."""
        if getattr(self, '_atomic_graph_write_verified', False) is not True:
            raise NotImplementedError('Atomic graph-write deployment gates are not verified')
        if not isinstance(request_hash, str) or not request_hash.strip():
            raise ValueError('Graph-write request hash is required')
        query, params = self._graph_write_query(request, context)
        params['request_hash'] = request_hash
        result = await self.execute_query(query, **params)
        records = result[0] if result else []
        if len(records) != 1:
            raise RuntimeError('Atomic graph-write returned an invalid receipt')
        receipt = dict(records[0])
        receipt.setdefault('request_id', request.request_id)
        receipt.setdefault('group_id', context.group_id)
        return receipt

    def session(self, database: str | None = None) -> GraphDriverSession:
        effective_db = database if database is not None else self._database
        logger.info(f'[FalkorDB] Creating session: requested_db={database}, effective_db={effective_db}')
        return FalkorDriverSession(self._get_graph(database))

    async def close(self) -> None:
        """Close the driver connection."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()  # type: ignore[reportUnknownMemberType]
        elif hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()
        elif hasattr(self.client.connection, 'close'):
            await self.client.connection.close()

    async def delete_all_indexes(self) -> None:
        result = await self.execute_query('CALL db.indexes()')
        if not result:
            return

        records, _, _ = result
        drop_tasks = []

        for record in records:
            label = record['label']
            entity_type = record['entitytype']

            for field_name, index_type in record['types'].items():
                if 'RANGE' in index_type:
                    drop_tasks.append(self.execute_query(f'DROP INDEX ON :{label}({field_name})'))
                elif 'FULLTEXT' in index_type:
                    if entity_type == 'NODE':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field_name})'
                            )
                        )
                    elif entity_type == 'RELATIONSHIP':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field_name})'
                            )
                        )

        if drop_tasks:
            await asyncio.gather(*drop_tasks)

    async def build_indices_and_constraints(self, delete_existing=False):
        try:
            if delete_existing:
                await self.delete_all_indexes()
            index_queries = get_range_indices(self.provider) + get_fulltext_indices(self.provider)
            for query in index_queries:
                try:
                    await self.execute_query(query)
                except Exception as e:
                    # Handle connection errors gracefully
                    if 'connection closed' in str(e).lower() or 'connectionerror' in str(type(e).__name__).lower():
                        logger.warning(f"Connection error while building index, skipping: {e}")
                        continue
                    # Re-raise other errors
                    raise

            # Create vector index for Episodic.content_embedding (Document mode fast lane)
            # This index enables semantic search on episodes before entity extraction completes
            try:
                await self.execute_query(
                    f"CREATE VECTOR INDEX FOR (e:Episodic) ON (e.content_embedding) "
                    f"OPTIONS {{dimension:{self._embedding_dimension}, similarityFunction:'cosine'}}"
                )
            except Exception as e:
                # Index may already exist or connection error
                if 'already indexed' not in str(e).lower() and 'already exists' not in str(e).lower():
                    if 'connection closed' in str(e).lower() or 'connectionerror' in str(type(e).__name__).lower():
                        logger.warning(f"Connection error while creating Episodic vector index: {e}")
                    else:
                        logger.warning(f"Failed to create Episodic content_embedding vector index: {e}")
        except Exception as e:
            # Catch any unexpected errors to prevent background task from crashing
            logger.error(f"Error in build_indices_and_constraints: {e}", exc_info=True)

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        if database == self._database:
            cloned = self
        elif database == self.default_group_id:
            cloned = FalkorDriver(
                falkor_db=self.client,
                embedding_dimension=self._embedding_dimension,
                content_storage_type=self.content_storage_type,
                content_storage_config={},
                skip_index_build=True,  # Skip index building for cloned drivers
            )
        else:
            # Create a new instance of FalkorDriver with the same connection but a different database
            cloned = FalkorDriver(
                falkor_db=self.client,
                database=database,
                embedding_dimension=self._embedding_dimension,
                content_storage_type=self.content_storage_type,
                content_storage_config={},
                skip_index_build=True,  # Skip index building for cloned drivers
            )

        return cloned

    async def health_check(self) -> None:
        """Check FalkorDB connectivity by running a simple query."""
        try:
            await self.execute_query('MATCH (n) RETURN 1 LIMIT 1')
            return None
        except Exception as e:
            print(f'FalkorDB health check failed: {e}')
            raise

    @staticmethod
    def convert_datetimes_to_strings(obj):
        if isinstance(obj, dict):
            return {k: FalkorDriver.convert_datetimes_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FalkorDriver.convert_datetimes_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(FalkorDriver.convert_datetimes_to_strings(item) for item in obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def sanitize(self, query: str) -> str:
        """
        Replace FalkorDB special characters with whitespace.
        Based on FalkorDB tokenization rules: ,.<>{}[]"':;!@#$%^&*()-+=~
        Also includes RediSearch query syntax characters: | /
        """
        # FalkorDB separator characters that break text into tokens
        # Plus RediSearch special characters (| is OR operator, / can cause issues)
        separator_map = str.maketrans(
            {
                ',': ' ',
                '.': ' ',
                '<': ' ',
                '>': ' ',
                '{': ' ',
                '}': ' ',
                '[': ' ',
                ']': ' ',
                '"': ' ',
                "'": ' ',
                ':': ' ',
                ';': ' ',
                '!': ' ',
                '@': ' ',
                '#': ' ',
                '$': ' ',
                '%': ' ',
                '^': ' ',
                '&': ' ',
                '*': ' ',
                '(': ' ',
                ')': ' ',
                '-': ' ',
                '+': ' ',
                '=': ' ',
                '~': ' ',
                '?': ' ',
                '|': ' ',  # RediSearch OR operator
                '/': ' ',  # Path separator, can cause issues
                '\\': ' ', # Backslash
                '_': ' ',  # Underscore - avoid standalone _ token in fulltext query
            }
        )
        sanitized = query.translate(separator_map)
        # Clean up multiple spaces
        sanitized = ' '.join(sanitized.split())
        return sanitized

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Build a fulltext query string for FalkorDB using RedisSearch syntax.
        FalkorDB uses RedisSearch-like syntax where:
        - Text search doesn't need @ prefix for content fields
        - AND is implicit with space
        - OR uses pipe within parentheses

        EasyOps fixes (see docs/easyops-customizations/40):
        - group_id filter is NOT embedded in the fulltext string: UUID group ids
          contain hyphens which the tokenizer splits, making exact @group_id
          matching always empty (BM25 silently returned 0 rows). Tenant filtering
          is applied downstream via Cypher WHERE n.group_id IN $group_ids.
        - Identifier-shaped queries (IPs, >=2 bare-number tokens) join tokens
          with AND instead of OR: OR over numeric fragments matches thousands
          of unrelated entities and buries the real hit.
        """
        sanitized_query = self.sanitize(query)

        # Remove stopwords from the sanitized query
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]

        # EasyOps: 混合 query 的 IP 子串识别（真实 Agent 流量场景）
        # 「主机 10.10.217.245」这类中文+IP 混合 query：现有判定对它也返回 True
        # （点分≥3段）走整串 AND，中文词（主机/信息/K8s节点）会被要求全部命中
        # 而目标实体的 name/summary 里没有这些词 → 0 命中。
        # 正确形态：IP 段 AND 必选 + 文字词 OR 附加：'(10 217 245 主机 | K8s节点)'。
        # 仅当 query 含 IP 且还有其他非 IP 文字时走该分支（纯 IP query 维持整串 AND）。
        ip_groups = self._extract_ip_groups(query)
        has_non_ip_words = any(w for w in filtered_words if not w.isdigit())
        if ip_groups and has_non_ip_words:
            return self._build_mixed_ip_query(ip_groups, filtered_words)

        # EasyOps: identifier-shaped queries use AND (space join), others OR
        joiner = ' ' if self._is_identifier_query(query, filtered_words) else ' | '
        # EasyOps: 去重保序——重复 token（如 IP 10.44.44.44 → 10/44/44/44）
        # 在 AND 语义下会被解析为要求词频，导致含该 token 的文档反而不命中
        seen: set[str] = set()
        deduped = [w for w in filtered_words if not (w in seen or seen.add(w))]
        sanitized_query = joiner.join(deduped)

        # If the query is too long return no query
        if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        # Handle empty search query
        if not sanitized_query.strip():
            return '*'

        return '(' + sanitized_query + ')'

    IP_PATTERN = re.compile(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b')

    @classmethod
    def _extract_ip_groups(cls, query: str) -> list[list[str]]:
        """从原始 query 抽取 IP 子串，返回去重后的段列表（每组一个 IP）。

        '主机 10.10.217.245' → [['10', '217', '245']]（10 重复段去重保序）
        无 IP 返回 []。
        """
        groups = []
        for m in cls.IP_PATTERN.finditer(query):
            octets = m.groups()
            seen: set[str] = set()
            deduped = [o for o in octets if not (o in seen or seen.add(o))]
            if deduped not in groups:
                groups.append(deduped)
        return groups

    @staticmethod
    def _build_mixed_ip_query(ip_groups: list[list[str]], words: list[str]) -> str:
        """混合 query 构造：IP 段 AND 必选，文字词作为可选 OR 组并列。

        单 IP：'((10 217 245) | 主机 | K8s节点)'
        - IP 段组是高置信约束，命中它的目标排最高（唯一且分数高）
        - 文字词 OR 并列做召回兜底：即便 IP 段因数据缺失不命中，
          含'主机'等的实体仍可被召回（不影响 IP 精确目标的排序）
        多 IP：'((10 217 245) | (10 42 1) | 文字词...)'
        """
        def dedup_keep_order(items: list[str]) -> list[str]:
            seen: set[str] = set()
            return [w for w in items if not (w in seen or seen.add(w))]

        all_ip_tokens = {t for g in ip_groups for t in g}
        word_terms = [w for w in dedup_keep_order(words) if w not in all_ip_tokens]
        or_parts = ['(' + ' '.join(g) + ')' for g in ip_groups] + word_terms
        return '(' + ' | '.join(or_parts) + ')'

    @staticmethod
    def _is_identifier_query(query: str, words: list[str]) -> bool:
        """标识符特征判定（原始 query 形态 + sanitize 后 token 双重判定）。

        - 原始 query 含点分多段（IP/域名，如 10.252.12.40、retail-brs-dev.belle.cn）
        - 原始 query 含 >=3 个连字符段（长 hostname，如 szsjhl-openstack-compute-test-node-01）
        - sanitize 后 >=2 个纯数字 token（工单号/版本号等）
        以上查询在 OR 语义下是噪声海洋；AND 要求全部 token 命中，实测可将正确
        目标排到首位。自然语言单词（how/fix/service）不受影响。
        """
        import re

        # 原始形态判定：点分 >=3 段（IP 或域名）
        segments = [s for s in re.split(r'[./]', query) if s.strip()]
        if len(segments) >= 3 and not any(' ' in s.strip() for s in segments[:3]):
            return True
        # 原始形态判定：长 hostname（>=3 个连字符段）
        if query.count('-') >= 3:
            return True
        # token 判定：>=2 个纯数字
        number_tokens = 0
        for w in words:
            if re.fullmatch(r'\d+', w):
                number_tokens += 1
            # IP 片段形态（未被拆完的残留，如 10.252）
            elif re.search(r'\d+\.\d+', w):
                return True
        return number_tokens >= 2

    async def load_episode_content(
        self, episode: 'EpisodicNode', max_length: int | None = None
    ) -> str:
        """
        Load episode content from storage if needed.

        Args:
            episode: EpisodicNode to load content for
            max_length: Optional max length to truncate content

        Returns:
            Episode content (loaded from storage if content field is empty)
        """
        # If content is already populated, return it
        if episode.content:
            if max_length and len(episode.content) > max_length:
                return episode.content[:max_length]
            return episode.content

        # If content is empty but we have file storage metadata, retrieve from storage
        if episode.content_file_path or episode.content_storage_type == 'redis':
            content = await self.content_storage.retrieve(
                file_path=episode.content_file_path,
                content=episode.content,  # May be None or empty
                max_length=max_length,
            )
            return content

        # No content available
        return ''

    async def prepare_episode_record(self, record: dict) -> dict:
        """
        Prepare episode record by loading content from file storage if needed.

        This method is called before creating EpisodicNode to ensure content
        is populated (required field) even when using file-based storage.

        Args:
            record: Database record dict with episode data

        Returns:
            Record dict with content loaded from storage if applicable
        """
        if record.get('content') is None and record.get('content_file_path'):
            content = await self.content_storage.retrieve(
                file_path=record['content_file_path'],
                content=None,
                max_length=None,
            )
            record = dict(record)  # Make mutable copy
            record['content'] = content or ''
        return record

    async def prepare_episode_for_save(self, episode: dict) -> dict:
        """
        Prepare episode dict for saving by storing content to file storage.

        This method is called before bulk saving episodes to:
        1. Store content to file storage
        2. Set storage metadata fields (content_hash, content_file_path, etc.)
        3. Replace content with empty string (FalkorDB stores only metadata)

        Args:
            episode: Episode dict with content to store

        Returns:
            Episode dict with content stored and metadata set
        """
        content = episode.get('content')
        if not content:
            return episode

        # Store content to file storage
        storage_result = await self.content_storage.store(
            episode_uuid=episode['uuid'],
            group_id=episode['group_id'],
            content=content,
        )

        # Update episode with storage metadata
        episode['content_hash'] = storage_result.content_hash
        episode['content_storage_type'] = self.content_storage_type
        episode['content_file_path'] = storage_result.file_path
        episode['content_file_size'] = storage_result.file_size

        # For file-based storage, set content to empty string (not None)
        # This ensures EpisodicNode validation passes when reading back
        if self.content_storage_type != 'redis':
            episode['content'] = ''

        return episode
