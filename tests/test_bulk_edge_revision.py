from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_bulk_query_by_type
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk, add_nodes_and_edges_bulk_tx


def make_edge(**changes):
    fields = dict(
        uuid='edge-1', group_id='tenant', source_node_uuid='source-1',
        target_node_uuid='target-1', name='LINKS_TO', fact='initial fact',
        fact_embedding=[0.1, 0.2], episodes=['episode-1'],
        created_at=datetime.now(timezone.utc), attributes={'custom': 'value'},
    )
    fields.update(changes)
    return EntityEdge(**fields)


def make_driver(**changes):
    fields = dict(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def make_tx(records=None):
    if records is None:
        records = [{'uuid': 'edge-1', 'revision': 1, 'changed': True, 'previous_uuid': None}]
    return SimpleNamespace(run=AsyncMock(), run_with_result=AsyncMock(return_value=(records, [], None)))


def test_typed_edge_default_query_retains_batch_unwind_and_last_wins():
    query = get_entity_edge_save_bulk_query_by_type(GraphProvider.FALKORDB, 'LINKS_TO')
    assert 'UNWIND $entity_edges AS edge' in query
    assert 'MERGE (source)-[r:LINKS_TO]->(target)' in query
    assert 'SET r = edge' in query


def test_typed_edge_revision_query_filters_physical_type_at_runtime():
    query = get_entity_edge_save_bulk_query_by_type(
        GraphProvider.FALKORDB, 'LINKS_TO', maintain_revision=True,
    )
    assert 'OPTIONAL MATCH (source)-[typed]->(target)' in query
    assert 'WHERE type(typed) = $edge.name' in query
    assert 'OPTIONAL MATCH (source)-[typed:LINKS_TO]->(target)' not in query
    assert 'MERGE (source)-[created:LINKS_TO]->(target)' in query


def test_typed_edge_revision_query_guards_global_endpoint_identity():
    query = get_entity_edge_save_bulk_query_by_type(
        GraphProvider.FALKORDB, 'LINKS_TO', maintain_revision=True,
    )

    assert 'OPTIONAL MATCH (source_candidate {uuid: $edge.source_node_uuid})' in query
    assert 'OPTIONAL MATCH (target_candidate {uuid: $edge.target_node_uuid})' in query
    assert "'Entity' IN labels(sources[0])" in query
    assert "'Entity' IN labels(targets[0])" in query
    assert 'uuid_matches[0] = existing' in query


def test_typed_edge_revision_query_returns_saved_relationship_receipt():
    query = get_entity_edge_save_bulk_query_by_type(
        GraphProvider.FALKORDB, 'LINKS_TO', maintain_revision=True,
    )

    assert 'saved_matches[0].uuid AS uuid' in query
    assert 'saved_matches[0]._revision AS revision' in query
    assert 'RETURN $edge.uuid AS uuid' not in query
    assert 'OPTIONAL MATCH (source)-[saved]->(target)' in query


def test_typed_edge_revision_query_rejects_invalid_type_or_provider():
    with pytest.raises(ValueError):
        get_entity_edge_save_bulk_query_by_type(
            GraphProvider.FALKORDB, 'LINKS-TO', maintain_revision=True,
        )
    with pytest.raises(NotImplementedError):
        get_entity_edge_save_bulk_query_by_type(
            GraphProvider.KUZU, 'LINKS_TO', maintain_revision=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,adapter', [
    (GraphProvider.KUZU, None), (GraphProvider.FALKORDB, object()),
])
async def test_typed_edge_opt_in_rejects_unsupported_path_before_session(provider, adapter):
    driver = make_driver(provider=provider, graph_operations_interface=adapter, session=MagicMock())
    with pytest.raises(NotImplementedError):
        await add_nodes_and_edges_bulk(
            driver, [], [], [], [make_edge(fact_embedding=None)], SimpleNamespace(),
            maintain_edge_revisions=True,
        )
    driver.session.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('attribute,value', [
    ('uuid', 'other'), ('group_id', 'other'), ('source_uuid', 'other'),
    ('target_uuid', 'other'), ('source_node_uuid', 'other'),
    ('target_node_uuid', 'other'), ('name', 'OTHER'),
    ('created_at', 'other'), ('valid_at', 'other'),
    ('invalid_at', 'other'), ('expired_at', 'other'),
])
async def test_typed_edge_opt_in_rejects_trusted_attribute_conflicts_before_session(attribute, value):
    driver = make_driver(session=MagicMock())
    edge = make_edge(fact_embedding=None, attributes={attribute: value})
    embedder = SimpleNamespace(create=AsyncMock())
    with pytest.raises(ValueError, match='identity'):
        await add_nodes_and_edges_bulk(
            driver, [], [], [], [edge], embedder, maintain_edge_revisions=True,
        )
    driver.session.assert_not_called()
    embedder.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_typed_edge_opt_in_uses_one_result_query_and_trusted_fields():
    edge = make_edge(attributes={
        'uuid': 'edge-1', 'source_node_uuid': 'source-1', '_revision': 999,
        'custom': 'value',
    })
    tx = make_tx()
    result = await add_nodes_and_edges_bulk_tx(
        tx, [], [], [], [edge], SimpleNamespace(), make_driver(),
        maintain_edge_revisions=True,
    )
    tx.run_with_result.assert_awaited_once()
    payload = tx.run_with_result.await_args.kwargs['edge']
    assert payload['uuid'] == edge.uuid
    assert payload['source_node_uuid'] == edge.source_node_uuid
    assert '_revision' not in payload
    assert result['edges'] == [{'uuid': edge.uuid, 'revision': 1, 'changed': True,
                                'previous_uuid': None}]


@pytest.mark.asyncio
@pytest.mark.parametrize('records', [
    [], [{'uuid': 'other', 'revision': 1, 'changed': True, 'previous_uuid': None}],
    [{'uuid': 'edge-1', 'revision': 0, 'changed': True, 'previous_uuid': None}],
    [{'uuid': 'edge-1', 'revision': 1, 'changed': 1, 'previous_uuid': None}],
    [{'uuid': 'edge-1', 'revision': 1, 'changed': True, 'previous_uuid': None}] * 2,
])
async def test_typed_edge_opt_in_requires_exact_receipt(records):
    tx = make_tx(records)
    with pytest.raises(ValueError):
        await add_nodes_and_edges_bulk_tx(
            tx, [], [], [], [make_edge()], SimpleNamespace(), make_driver(),
            maintain_edge_revisions=True,
        )


@pytest.mark.asyncio
async def test_typed_edge_default_uses_one_unwind_per_type():
    tx = make_tx()
    await add_nodes_and_edges_bulk_tx(
        tx, [], [], [], [make_edge(), make_edge(uuid='edge-2')],
        SimpleNamespace(), make_driver(),
    )
    typed_calls = [call for call in tx.run.await_args_list
                   if 'MERGE (source)-[r:LINKS_TO]->(target)' in call.args[0]]
    assert len(typed_calls) == 1
    assert len(typed_calls[0].kwargs['entity_edges']) == 2
    tx.run_with_result.assert_not_awaited()
