from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode


def test_revision_query_reads_live_value_before_replacing_properties():
    query = get_entity_node_save_query(GraphProvider.FALKORDB, 'Service:Entity', maintain_revision=True)

    assert 'OPTIONAL MATCH (candidate {uuid: $entity_data.uuid})' in query
    assert 'collect(candidate) AS matches' in query
    assert 'size(matches) AS match_count' in query
    assert 'existing._revision AS current_revision' in query
    assert query.index('collect(candidate) AS matches') < query.index('MERGE (n:Entity')
    assert 'match_count = 0 OR' in query
    assert 'match_count = 1 AND' in query
    assert 'ON CREATE SET n._revision = 0' not in query
    assert 'n._revision = CASE WHEN existing IS NULL THEN 1' in query
    assert 'ELSE coalesce(current_revision, 1) + 1 END' in query
    assert 'RETURN n.uuid AS uuid, n._revision AS revision, changed AS changed' in query
    assert 'vecf32($entity_data.name_embedding)' in query
    assert 'THEN existing.name_embedding IS NULL' in query
    assert 'current_group_id = $entity_data.group_id' in query
    assert 'toString(current_revision) = toString(toInteger(current_revision))' in query


def test_revision_query_counts_untyped_uuid_and_validates_entity_label():
    query = get_entity_node_save_query(GraphProvider.FALKORDB, 'Alert:Entity', maintain_revision=True)

    assert 'OPTIONAL MATCH (candidate {uuid: $entity_data.uuid})' in query
    assert 'OPTIONAL MATCH (candidate:Entity' not in query
    assert "'Entity' IN labels(existing)" in query
    assert query.index("'Entity' IN labels(existing)") < query.index('MERGE (n:Entity')


def test_revision_query_compares_effective_full_map_and_returns_live_receipt():
    query = get_entity_node_save_query(GraphProvider.FALKORDB, 'Alert:Entity', maintain_revision=True)

    assert "key <> '_revision'" in query
    assert 'keys(existing)' in query and 'keys($entity_data)' in query
    assert 'vecf32($entity_data.name_embedding)' in query
    assert 'labels(existing)' in query
    assert 'CASE WHEN changed THEN [1] ELSE [] END' in query
    assert 'RETURN n.uuid AS uuid, n._revision AS revision, changed AS changed' in query


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [
    {'uuid': 'node-1', 'revision': 0, 'changed': True},
    {'uuid': 'node-1', 'revision': 1, 'changed': 1},
    {'uuid': 'node-1', 'revision': None, 'changed': True},
    {'uuid': 'node-1', 'revision': True, 'changed': False},
    {'uuid': 'node-1', 'revision': 1, 'changed': False, 'extra': 1},
])
async def test_revision_save_rejects_invalid_internal_receipt(record):
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        execute_query=AsyncMock(return_value=([record], None, None)),
    )
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant')

    with pytest.raises(ValueError):
        await node.save(driver, maintain_revision=True)


@pytest.mark.asyncio
async def test_revision_save_accepts_unchanged_legacy_receipt_and_returns_uuid_only():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        execute_query=AsyncMock(return_value=(
            [{'uuid': 'node-1', 'revision': None, 'changed': False}],
            ['uuid', 'revision', 'changed'], None,
        )),
    )
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant')

    assert (await node.save(driver, maintain_revision=True))[0] == [{'uuid': 'node-1'}]


@pytest.mark.asyncio
async def test_entity_get_by_uuid_exposes_vectors_only_for_explicit_write_snapshot():
    def record():
        return {
            'uuid': 'node-1', 'name': 'service', 'group_id': 'tenant',
            'created_at': datetime.now(timezone.utc), 'summary': 'summary',
            'labels': ['Entity', 'Service'],
            'attributes': {
                'uuid': 'node-1', 'name': 'service', 'group_id': 'tenant',
                'summary': 'summary', 'name_embedding': [0.1],
                'summary_embedding': [0.2], 'status': 'active',
            },
        }

    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        execute_query=AsyncMock(side_effect=[
            ([record()], None, None),
            ([{**record(), 'name_embedding': [0.1], 'summary_embedding': [0.2]}], None, None),
        ]),
    )

    default_node = await EntityNode.get_by_uuid(driver, 'node-1')
    write_node = await EntityNode.get_by_uuid(driver, 'node-1', include_embeddings=True)

    assert default_node.name_embedding is None
    assert default_node.summary_embedding is None
    assert write_node.name_embedding == [0.1]
    assert write_node.summary_embedding == [0.2]
    assert default_node.attributes == write_node.attributes == {'status': 'active'}
    assert 'n.name_embedding AS name_embedding' not in get_entity_node_return_query(
        GraphProvider.FALKORDB,
    )
    assert 'n.name_embedding AS name_embedding' in driver.execute_query.await_args.args[0]
    assert 'n.summary_embedding AS summary_embedding' in driver.execute_query.await_args.args[0]


def test_revision_query_rejects_unsupported_provider():
    with pytest.raises(NotImplementedError):
        get_entity_node_save_query(GraphProvider.KUZU, '', maintain_revision=True)


def test_default_save_query_has_no_revision_write():
    query = get_entity_node_save_query(GraphProvider.FALKORDB, 'Service:Entity')

    assert '_revision' not in query


@pytest.mark.asyncio
async def test_revision_save_strips_stale_attribute_and_preserves_uuid_result():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        graph_operations_interface=None,
        execute_query=AsyncMock(return_value=([{'uuid': 'node-1', 'revision': 1, 'changed': True}], None, None)),
    )
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant', attributes={'_revision': 99, 'status': 'up'})

    result = await node.save(driver, maintain_revision=True)

    assert result[0][0]['uuid'] == 'node-1'
    assert '_revision' not in driver.execute_query.call_args.kwargs['entity_data']
    assert 'maintain_revision' not in driver.execute_query.call_args.kwargs


@pytest.mark.asyncio
async def test_revision_save_rejects_custom_adapter_before_write():
    adapter = SimpleNamespace(node_save=AsyncMock())
    driver = SimpleNamespace(provider=GraphProvider.FALKORDB, graph_operations_interface=adapter)
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant')

    with pytest.raises(NotImplementedError):
        await node.save(driver, maintain_revision=True)
    adapter.node_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_revision_save_rejects_empty_query_result():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        graph_operations_interface=None,
        execute_query=AsyncMock(return_value=([], None, None)),
    )
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant')

    with pytest.raises(ValueError):
        await node.save(driver, maintain_revision=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('records', [
    {'uuid': 'node-1'},
    [{'uuid': 'other-node'}],
    [{'uuid': 'node-1'}, {'uuid': 'node-1'}],
    [{'uuid': 'node-1'}, {'uuid': 'other-node'}],
])
async def test_revision_save_rejects_invalid_uuid_receipt(records):
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        graph_operations_interface=None,
        execute_query=AsyncMock(return_value=(records, None, None)),
    )
    node = EntityNode(uuid='node-1', name='service', labels=['Service'], group_id='tenant')

    with pytest.raises(ValueError):
        await node.save(driver, maintain_revision=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('attribute_name,attribute_value', [
    ('uuid', 'other-node'),
    ('group_id', 'other-tenant'),
])
async def test_revision_save_rejects_conflicting_identity_before_graph_io(attribute_name, attribute_value):
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        graph_operations_interface=None,
        execute_query=AsyncMock(),
    )
    node = EntityNode(
        uuid='node-1', name='service', labels=['Service'], group_id='tenant',
        attributes={attribute_name: attribute_value},
    )

    with pytest.raises(ValueError, match='identity'):
        await node.save(driver, maintain_revision=True)
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_revision_save_accepts_matching_identity_attributes():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        graph_operations_interface=None,
        execute_query=AsyncMock(return_value=([{'uuid': 'node-1', 'revision': 1, 'changed': True}], None, None)),
    )
    node = EntityNode(
        uuid='node-1', name='service', labels=['Service'], group_id='tenant',
        attributes={'uuid': 'node-1', 'group_id': 'tenant'},
    )

    await node.save(driver, maintain_revision=True)

    entity_data = driver.execute_query.call_args.kwargs['entity_data']
    assert entity_data['uuid'] == 'node-1'
    assert entity_data['group_id'] == 'tenant'
