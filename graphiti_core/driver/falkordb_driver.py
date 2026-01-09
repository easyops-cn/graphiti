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
import logging
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
        - Field queries use @ prefix: @field:value
        - Multiple values for same field: (@field:value1|value2)
        - Text search doesn't need @ prefix for content fields
        - AND is implicit with space: (@group_id:value) (text)
        - OR uses pipe within parentheses: (@group_id:value1|value2)

        Note: Underscores are NOT separators in RediSearch, so group_ids with underscores
        don't need escaping. See: https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/escaping/
        """
        if group_ids is None or len(group_ids) == 0:
            group_filter = ''
        else:
            group_values = '|'.join(group_ids)
            group_filter = f'(@group_id:{group_values})'

        sanitized_query = self.sanitize(query)

        # Remove stopwords from the sanitized query
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
        sanitized_query = ' | '.join(filtered_words)

        # If the query is too long return no query
        if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        # Handle empty search query - only use group filter (avoid empty "()" which causes syntax error)
        if not sanitized_query.strip():
            # Return just the group filter, or '*' if no filter
            return group_filter if group_filter else '*'

        full_query = group_filter + ' (' + sanitized_query + ')'

        return full_query

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
