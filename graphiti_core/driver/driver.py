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

import copy
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from enum import Enum
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.search_interface.search_interface import SearchInterface

if TYPE_CHECKING:
    from graphiti_core.nodes import EpisodicNode

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 10

load_dotenv()

ENTITY_INDEX_NAME = os.environ.get('ENTITY_INDEX_NAME', 'entities')
EPISODE_INDEX_NAME = os.environ.get('EPISODE_INDEX_NAME', 'episodes')
COMMUNITY_INDEX_NAME = os.environ.get('COMMUNITY_INDEX_NAME', 'communities')
ENTITY_EDGE_INDEX_NAME = os.environ.get('ENTITY_EDGE_INDEX_NAME', 'entity_edges')


class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'


class GraphDriverSession(ABC):
    provider: GraphProvider

    async def __aenter__(self):
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def close(self):
        raise NotImplementedError()

    @abstractmethod
    async def execute_write(self, func, *args, **kwargs):
        raise NotImplementedError()


class GraphDriver(ABC):
    provider: GraphProvider
    fulltext_syntax: str = (
        ''  # Neo4j (default) syntax does not require a prefix for fulltext queries
    )
    _database: str
    default_group_id: str = ''
    search_interface: SearchInterface | None = None
    graph_operations_interface: GraphOperationsInterface | None = None

    @abstractmethod
    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str | None = None) -> GraphDriverSession:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def delete_all_indexes(self) -> Coroutine:
        raise NotImplementedError()

    def with_database(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        cloned = copy.copy(self)
        cloned._database = database

        return cloned

    @abstractmethod
    async def build_indices_and_constraints(self, delete_existing: bool = False):
        raise NotImplementedError()

    def clone(self, database: str) -> 'GraphDriver':
        """Clone the driver with a different database or graph name."""
        return self

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Specific fulltext query builder for database providers.
        Only implemented by providers that need custom fulltext query building.
        """
        raise NotImplementedError(f'build_fulltext_query not implemented for {self.provider}')

    async def prepare_episode_for_save(self, episode: dict) -> dict:
        """
        Prepare episode dict for saving to database.

        Default implementation returns episode unchanged.
        Override in drivers that use external content storage (e.g., FalkorDB with file storage)
        to store content externally and update metadata fields.

        Args:
            episode: Episode dict with content to store

        Returns:
            Episode dict ready for database insertion
        """
        return episode

    async def prepare_episode_record(self, record: dict) -> dict:
        """
        Prepare episode record after loading from database.

        Default implementation returns record unchanged.
        Override in drivers that use external content storage to load content
        from storage before creating EpisodicNode.

        Args:
            record: Database record dict with episode data

        Returns:
            Record dict with content loaded if applicable
        """
        return record

    async def load_episode_content(
        self, episode: 'EpisodicNode', max_length: int | None = None
    ) -> str:
        """
        Load episode content from storage if needed.

        Default implementation returns episode.content directly.
        Override in drivers that use external content storage (e.g., FalkorDB with file storage).

        Args:
            episode: EpisodicNode to load content for
            max_length: Optional max length to truncate content

        Returns:
            Episode content (loaded from storage if content field is empty)
        """
        content = episode.content or ''
        if max_length and len(content) > max_length:
            return content[:max_length]
        return content
