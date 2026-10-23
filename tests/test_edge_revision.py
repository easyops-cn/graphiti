from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_query


def make_edge(**overrides):
    fields = dict(
        uuid='edge-1', group_id='tenant', source_node_uuid='source-1',
        target_node_uuid='target-1', name='RELATES_TO', fact='source relates target',
        created_at=datetime.now(timezone.utc), fact_embedding=[0.1], attributes={'status': 'open'},
    )
    fields.update(overrides)
    return EntityEdge(**fields)


def make_driver(records):
    return SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        execute_query=AsyncMock(return_value=(records, None, None)),
    )


def test_opt_in_save_query_gates_all_uuid_and_endpoint_matches_before_write():
    query = get_entity_edge_save_query(GraphProvider.FALKORDB, maintain_revision=True)

    assert 'collect(candidate) AS matches' in query
    assert 'size(matches) AS match_count' in query
    assert 'match_count = 0 OR' in query
    assert 'match_count = 1 AND' in query
    assert 'size(sources) = 1' in query
    assert 'size(targets) = 1' in query
    assert query.index('existing._revision') < query.index('SET existing = $edge_data')
    assert 'coalesce(prior_revision, 1) + 1' in query
    assert 'created._revision = 1' in query
    assert 'RETURN saved_matches[0].uuid AS uuid' in query


def test_opt_in_save_updates_matched_relation_without_unconditional_typed_merge():
    query = get_entity_edge_save_query(GraphProvider.FALKORDB, maintain_revision=True)

    assert 'MERGE (source)-[e:RELATES_TO' not in query
    assert 'CASE WHEN match_count = 1 AND changed THEN [1] ELSE [] END' in query
    assert 'SET existing = $edge_data' in query
    assert 'CASE WHEN match_count = 0 THEN [1] ELSE [] END' in query
    assert 'CREATE (source)-[created:RELATES_TO]->(target)' in query
    assert 'type(saved) = \'RELATES_TO\'' in query
    assert 'size(saved_matches) = 1' in query


def test_opt_in_save_guards_global_endpoint_identity_before_writing():
    query = get_entity_edge_save_query(GraphProvider.FALKORDB, maintain_revision=True)

    assert 'OPTIONAL MATCH (source_candidate {uuid: $edge_data.source_uuid})' in query
    assert 'OPTIONAL MATCH (target_candidate {uuid: $edge_data.target_uuid})' in query
    assert "'Entity' IN labels(sources[0])" in query
    assert "'Entity' IN labels(targets[0])" in query
    assert 'existing_source = source AND existing_target = target' in query
    assert query.index('existing_source = source AND existing_target = target') < query.index(
        'CREATE (source)-[created:RELATES_TO]'
    )


def test_opt_in_save_compares_full_map_and_returns_live_receipt():
    query = get_entity_edge_save_query(GraphProvider.FALKORDB, maintain_revision=True)

    assert "key <> '_revision'" in query
    assert 'keys(existing)' in query and 'keys($edge_data)' in query
    assert 'vecf32($edge_data.fact_embedding)' in query
    assert 'CASE WHEN match_count = 1 AND changed THEN [1] ELSE [] END' in query
    assert 'saved_matches[0]._revision AS revision, changed AS changed' in query


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [
    {'uuid': 'edge-1', 'revision': 0, 'changed': True},
    {'uuid': 'edge-1', 'revision': 1, 'changed': 1},
    {'uuid': 'edge-1', 'revision': None, 'changed': True},
    {'uuid': 'edge-1', 'revision': True, 'changed': False},
    {'uuid': 'edge-1', 'revision': 1, 'changed': False, 'extra': 1},
])
async def test_opt_in_save_rejects_invalid_internal_receipt(record):
    driver = make_driver([record])

    with pytest.raises(ValueError):
        await make_edge().save(driver, maintain_revision=True)


@pytest.mark.asyncio
async def test_opt_in_save_accepts_unchanged_legacy_receipt_and_returns_uuid_only():
    driver = make_driver([{'uuid': 'edge-1', 'revision': None, 'changed': False}])

    assert (await make_edge().save(driver, maintain_revision=True))[0] == [{'uuid': 'edge-1'}]


@pytest.mark.asyncio
async def test_opt_in_save_strips_stale_revision_and_returns_uuid():
    driver = make_driver([{'uuid': 'edge-1', 'revision': 1, 'changed': True}])
    edge = make_edge(attributes={'_revision': 999, 'status': 'open'})

    result = await edge.save(driver, maintain_revision=True)

    assert result[0] == [{'uuid': 'edge-1'}]
    assert '_revision' not in driver.execute_query.call_args.kwargs['edge_data']


@pytest.mark.asyncio
@pytest.mark.parametrize('attribute_name,attribute_value', [
    ('uuid', 'other-edge'), ('group_id', 'other-tenant'),
    ('source_uuid', 'other-source'), ('target_uuid', 'other-target'),
    ('source_node_uuid', 'other-source'), ('target_node_uuid', 'other-target'),
])
async def test_opt_in_save_rejects_identity_spoof_before_graph_io(attribute_name, attribute_value):
    driver = make_driver([{'uuid': 'edge-1', 'revision': 1, 'changed': True}])
    edge = make_edge(attributes={attribute_name: attribute_value})

    with pytest.raises(ValueError, match='identity'):
        await edge.save(driver, maintain_revision=True)
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('records', [[], [{'uuid': 'other'}], [{'uuid': 'edge-1'}] * 2])
async def test_opt_in_save_rejects_bad_receipt(records):
    driver = make_driver(records)

    with pytest.raises(ValueError):
        await make_edge().save(driver, maintain_revision=True)


@pytest.mark.asyncio
async def test_opt_in_save_rejects_custom_adapter_and_provider():
    edge = make_edge()
    adapter = SimpleNamespace(edge_save=AsyncMock())
    driver = make_driver([{'uuid': 'edge-1', 'revision': 1, 'changed': True}])
    driver.graph_operations_interface = adapter
    with pytest.raises(NotImplementedError):
        await edge.save(driver, maintain_revision=True)
    driver.execute_query.assert_not_awaited()
    adapter.edge_save.assert_not_awaited()
    driver.graph_operations_interface = None
    driver.provider = GraphProvider.KUZU
    with pytest.raises(NotImplementedError):
        await edge.save(driver, maintain_revision=True)


@pytest.mark.asyncio
async def test_temporal_replace_uses_one_query_and_exact_pair_receipt():
    driver = make_driver([{'old_uuid': 'edge-1', 'new_uuid': 'edge-2'}])
    old_edge = make_edge(invalid_at=datetime.now(timezone.utc), expired_at=datetime.now(timezone.utc))
    replacement = make_edge(uuid='edge-2')

    result = await old_edge.replace_temporally(driver, replacement)

    driver.execute_query.assert_awaited_once()
    assert result[0] == [{'old_uuid': 'edge-1', 'new_uuid': 'edge-2'}]
    query = driver.execute_query.call_args.args[0]
    assert 'OPTIONAL MATCH ()-[old_candidate]->()' in query
    assert 'size(old_matches) = 1' in query
    assert 'MATCH (old_source:Entity)-[old]->(old_target:Entity)' in query
    assert 'old.name = $old_data.name' in query
    assert "type(old) = 'RELATES_TO'" in query
    assert 'size(new_matches) = 0' in query
    assert 'SET old.invalid_at' in query
    assert 'CREATE (source)-[new:RELATES_TO]->(target)' in query
    assert query.index('size(new_matches) = 0') < query.index('SET old.invalid_at')
    assert 'coalesce(prior_revision, 1) + 1' in query
    assert 'new._revision = 1' in query


@pytest.mark.asyncio
async def test_temporal_replace_bad_receipt_fails():
    driver = make_driver([])
    old_edge = make_edge(invalid_at=datetime.now(timezone.utc), expired_at=datetime.now(timezone.utc))

    with pytest.raises(ValueError):
        await old_edge.replace_temporally(driver, make_edge(uuid='edge-2'))


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement_changes', [
    {'name': 'OTHER'},
    {'invalid_at': datetime.now(timezone.utc)},
    {'expired_at': datetime.now(timezone.utc)},
    {'attributes': {'invalid_at': datetime.now(timezone.utc)}},
    {'attributes': {'expired_at': datetime.now(timezone.utc)}},
    {'attributes': {'valid_at': datetime.now(timezone.utc)}},
])
async def test_temporal_replace_rejects_type_or_validity_conflict_before_io(replacement_changes):
    driver = make_driver([{'old_uuid': 'edge-1', 'new_uuid': 'edge-2'}])
    old_edge = make_edge(invalid_at=datetime.now(timezone.utc), expired_at=datetime.now(timezone.utc))
    replacement = make_edge(uuid='edge-2', **replacement_changes)

    with pytest.raises(ValueError):
        await old_edge.replace_temporally(driver, replacement)
    driver.execute_query.assert_not_awaited()
