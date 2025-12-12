"""
Test edge deduplication MERGE logic.

This test ensures that when saving edges with the same (source, target, edge_type),
they are merged into a single edge instead of creating duplicates.

EasyOps customization: MERGE by (source, target) only, not by uuid.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from tests.helpers_test import group_id

pytest_plugins = ('pytest_asyncio',)

logger = logging.getLogger(__name__)


async def count_edges_by_type(
    driver: GraphDriver,
    source_uuid: str,
    target_uuid: str,
    edge_type: str,
) -> int:
    """Count edges of a specific type between two nodes."""
    results, _, _ = await driver.execute_query(
        f"""
        MATCH (source:Entity {{uuid: $source_uuid}})-[e:{edge_type}]->(target:Entity {{uuid: $target_uuid}})
        RETURN COUNT(e) AS count
        """,
        source_uuid=source_uuid,
        target_uuid=target_uuid,
    )
    return int(results[0]['count']) if results else 0


async def get_edge_fact(
    driver: GraphDriver,
    source_uuid: str,
    target_uuid: str,
    edge_type: str,
) -> str | None:
    """Get the fact of an edge between two nodes."""
    results, _, _ = await driver.execute_query(
        f"""
        MATCH (source:Entity {{uuid: $source_uuid}})-[e:{edge_type}]->(target:Entity {{uuid: $target_uuid}})
        RETURN e.fact AS fact
        """,
        source_uuid=source_uuid,
        target_uuid=target_uuid,
    )
    return results[0]['fact'] if results else None


@pytest.fixture
def mock_embedding():
    """Generate a random embedding for testing."""
    return np.random.uniform(0.0, 0.9, 1024).tolist()


@pytest.mark.asyncio
async def test_bulk_edge_save_merge(graph_driver, mock_embedding):
    """
    Test bulk edge save with duplicates - should merge, not create duplicates.

    This is the CORE test for the edge deduplication fix.
    It simulates the real-world scenario where multiple episodes
    mention the same relationship.

    Expected behavior:
    - 3 edges with same (source, target, edge_type) should be merged into 1
    - The last edge's fact should win (SET overwrites)
    """
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    now = datetime.now()

    # Create nodes
    source_node = EntityNode(
        name='BulkTestProduct',
        labels=['Entity', 'Product'],
        created_at=now,
        summary='Bulk Test Product',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    target_node = EntityNode(
        name='BulkTestFeature',
        labels=['Entity', 'Feature'],
        created_at=now,
        summary='Bulk Test Feature',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    # Create 3 edges with SAME (source, target, type) but different facts
    edges = [
        EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name='HAS_FEATURE',
            fact=f'Fact version {i}',
            group_id=group_id,
            created_at=now,
            episodes=[],
            fact_embedding=mock_embedding,
        )
        for i in range(3)
    ]

    # Create mock embedder
    class MockEmbedder:
        async def create(self, text: str) -> list[float]:
            return mock_embedding

    # Save in bulk
    await add_nodes_and_edges_bulk(
        graph_driver,
        episodic_nodes=[],
        episodic_edges=[],
        entity_nodes=[source_node, target_node],
        entity_edges=edges,
        embedder=MockEmbedder(),
    )

    # CRITICAL: Should have only 1 edge, not 3
    count = await count_edges_by_type(
        graph_driver, source_node.uuid, target_node.uuid, 'HAS_FEATURE'
    )
    assert count == 1, (
        f'Bulk save should merge duplicate edges. Expected 1 edge, got {count}. '
        'Bulk edge dedup MERGE is broken!'
    )

    logger.info('Bulk edge merge test passed: 3 duplicate edges merged into 1')


@pytest.mark.asyncio
async def test_bulk_edge_save_merge_updates_fact(graph_driver, mock_embedding):
    """
    Test that bulk save merges edges and the last fact wins.

    This verifies that MERGE + SET correctly updates the edge properties
    when duplicate edges are saved.
    """
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    now = datetime.now()

    source_node = EntityNode(
        name='FactUpdateProduct',
        labels=['Entity', 'Product'],
        created_at=now,
        summary='Fact Update Test Product',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    target_node = EntityNode(
        name='FactUpdateFeature',
        labels=['Entity', 'Feature'],
        created_at=now,
        summary='Fact Update Test Feature',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    class MockEmbedder:
        async def create(self, text: str) -> list[float]:
            return mock_embedding

    # First save
    edge1 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='HAS_FEATURE',
        fact='First fact',
        group_id=group_id,
        created_at=now,
        episodes=[],
        fact_embedding=mock_embedding,
    )

    await add_nodes_and_edges_bulk(
        graph_driver,
        episodic_nodes=[],
        episodic_edges=[],
        entity_nodes=[source_node, target_node],
        entity_edges=[edge1],
        embedder=MockEmbedder(),
    )

    # Verify first fact
    fact1 = await get_edge_fact(graph_driver, source_node.uuid, target_node.uuid, 'HAS_FEATURE')
    assert fact1 == 'First fact'

    # Second save with different fact (same source, target, type)
    edge2 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='HAS_FEATURE',
        fact='Updated fact',  # Different fact
        group_id=group_id,
        created_at=now,
        episodes=[],
        fact_embedding=mock_embedding,
    )

    await add_nodes_and_edges_bulk(
        graph_driver,
        episodic_nodes=[],
        episodic_edges=[],
        entity_nodes=[],  # Nodes already exist
        entity_edges=[edge2],
        embedder=MockEmbedder(),
    )

    # Should still have only 1 edge
    count = await count_edges_by_type(
        graph_driver, source_node.uuid, target_node.uuid, 'HAS_FEATURE'
    )
    assert count == 1, f'Expected 1 edge after second save, got {count}'

    # Fact should be updated
    fact2 = await get_edge_fact(graph_driver, source_node.uuid, target_node.uuid, 'HAS_FEATURE')
    assert fact2 == 'Updated fact', f'Expected "Updated fact", got "{fact2}"'

    logger.info('Fact update test passed: edge fact correctly updated on merge')


@pytest.mark.asyncio
async def test_bulk_edge_different_types_not_merged(graph_driver, mock_embedding):
    """
    Test that edges with different types are NOT merged.

    This is the negative test - we want to make sure different edge types
    between the same nodes are kept as separate edges.
    """
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

    now = datetime.now()

    source_node = EntityNode(
        name='DiffTypeProduct',
        labels=['Entity', 'Product'],
        created_at=now,
        summary='Different Type Test Product',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    target_node = EntityNode(
        name='DiffTypeFeature',
        labels=['Entity', 'Feature'],
        created_at=now,
        summary='Different Type Test Feature',
        group_id=group_id,
        name_embedding=mock_embedding,
    )

    # Create edges with DIFFERENT types
    edges = [
        EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name='HAS_FEATURE',
            fact='Product has feature',
            group_id=group_id,
            created_at=now,
            episodes=[],
            fact_embedding=mock_embedding,
        ),
        EntityEdge(
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            name='RELATES_TO',
            fact='Product relates to feature',
            group_id=group_id,
            created_at=now,
            episodes=[],
            fact_embedding=mock_embedding,
        ),
    ]

    class MockEmbedder:
        async def create(self, text: str) -> list[float]:
            return mock_embedding

    await add_nodes_and_edges_bulk(
        graph_driver,
        episodic_nodes=[],
        episodic_edges=[],
        entity_nodes=[source_node, target_node],
        entity_edges=edges,
        embedder=MockEmbedder(),
    )

    # Should have 2 edges (one of each type)
    count_has_feature = await count_edges_by_type(
        graph_driver, source_node.uuid, target_node.uuid, 'HAS_FEATURE'
    )
    count_relates_to = await count_edges_by_type(
        graph_driver, source_node.uuid, target_node.uuid, 'RELATES_TO'
    )

    assert count_has_feature == 1, f'Expected 1 HAS_FEATURE edge, got {count_has_feature}'
    assert count_relates_to == 1, f'Expected 1 RELATES_TO edge, got {count_relates_to}'

    logger.info('Different types test passed: edges with different types kept separate')
