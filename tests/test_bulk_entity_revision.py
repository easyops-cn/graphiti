from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_bulk_query
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk, add_nodes_and_edges_bulk_tx


def node_data(**changes):
    data = {
        'uuid': 'node-1', 'name': 'service', 'group_id': 'tenant',
        'labels': ['Entity', 'Service', 'Managed'],
        'summary': '', 'created_at': datetime.now(timezone.utc),
        'name_embedding': [0.1, 0.2], 'summary_embedding': None,
        'reasoning': None, 'type_scores': None, 'type_confidence': None,
        'synonyms': None, 'status': 'open',
    }
    data.update(changes)
    return data


def make_node(**changes):
    fields = dict(
        uuid='node-1', name='service', group_id='tenant',
        labels=['Service', 'Managed'], created_at=datetime.now(timezone.utc),
        name_embedding=[0.1, 0.2], summary_embedding=None,
        attributes={'status': 'open'},
    )
    fields.update(changes)
    return EntityNode(**fields)


def test_revision_bulk_builder_emits_one_guarded_query_per_multilabel_node():
    queries = get_entity_node_save_bulk_query(
        GraphProvider.FALKORDB, [node_data()], maintain_revision=True,
    )

    assert len(queries) == 1
    query, params = queries[0]
    assert 'collect(candidate) AS matches' in query
    assert 'size(matches) AS match_count' in query
    assert 'SET n:Entity:Service:Managed' in query
    assert 'n.labels = labels_to_save' in query
    assert 'n.labels = labels(n)' not in query
    assert query.count('n._revision = CASE') == 1
    assert 'coalesce(prior_revision, 1) + 1' in query
    assert 'n.name_embedding = vecf32($node.name_embedding)' in query
    assert 'THEN existing.name_embedding IS NULL' in query
    assert 'n.name_embedding = vecf32($node.name_embedding)' in query
    assert 'existing.summary_embedding IS NULL' in query
    assert 'changed' in query
    assert query.index('prior_revision') < query.index('n._revision = CASE')
    assert params['node']['uuid'] == 'node-1'
    assert '_revision' not in params['node']


def test_revision_bulk_builder_counts_untyped_uuid_and_validates_entity_label():
    query, _ = get_entity_node_save_bulk_query(
        GraphProvider.FALKORDB, [node_data()], maintain_revision=True,
    )[0]

    assert 'OPTIONAL MATCH (candidate {uuid: $node.uuid})' in query
    assert 'OPTIONAL MATCH (candidate:Entity' not in query
    assert "'Entity' IN labels(existing)" in query
    assert query.index("'Entity' IN labels(existing)") < query.index('MERGE (n:Entity')


def test_revision_bulk_builder_returns_live_revision_receipt():
    query, _ = get_entity_node_save_bulk_query(
        GraphProvider.FALKORDB, [node_data()], maintain_revision=True,
    )[0]

    assert 'RETURN n.uuid AS uuid, n._revision AS revision, changed AS changed' in query


def test_revision_bulk_builder_rejects_unsupported_provider():
    with pytest.raises(NotImplementedError):
        get_entity_node_save_bulk_query(GraphProvider.KUZU, [node_data()], maintain_revision=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,adapter', [
    (GraphProvider.KUZU, None), (GraphProvider.FALKORDB, object()),
])
async def test_bulk_opt_in_rejects_unsupported_path_before_session_or_embedding(provider, adapter):
    driver = SimpleNamespace(provider=provider, graph_operations_interface=adapter, session=MagicMock())
    node = make_node(name_embedding=None)
    embedder = SimpleNamespace(create=AsyncMock())

    with pytest.raises(NotImplementedError):
        await add_nodes_and_edges_bulk(
            driver, [], [], [node], [], embedder, maintain_entity_revisions=True,
        )
    driver.session.assert_not_called()
    embedder.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_opt_in_rejects_spoof_before_session():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None, session=MagicMock(),
    )
    node = make_node(attributes={'uuid': 'other'})

    with pytest.raises(ValueError, match='identity'):
        await add_nodes_and_edges_bulk(
            driver, [], [], [node], [], SimpleNamespace(), maintain_entity_revisions=True,
        )
    driver.session.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_opt_in_rejects_conflicting_type_labels_before_session():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None, session=MagicMock(),
    )
    node = make_node(attributes={'labels': ['ForeignType']}, name_embedding=None)
    embedder = SimpleNamespace(create=AsyncMock())

    with pytest.raises(ValueError, match='labels'):
        await add_nodes_and_edges_bulk(
            driver, [], [], [node], [], embedder, maintain_entity_revisions=True,
        )
    driver.session.assert_not_called()
    embedder.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_opt_in_rebinds_matching_type_label_metadata():
    tx = SimpleNamespace(
        run=AsyncMock(),
        run_with_result=AsyncMock(return_value=([
            {'uuid': 'node-1', 'revision': 1, 'changed': False},
        ], [], None)),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )
    node = make_node(attributes={'labels': ['Managed', 'Entity', 'Service', 'Managed']})

    await add_nodes_and_edges_bulk_tx(
        tx, [], [], [node], [], SimpleNamespace(), driver,
        maintain_entity_revisions=True,
    )

    saved_labels = tx.run_with_result.await_args.kwargs['node']['labels']
    assert set(saved_labels) == {'Entity', 'Service', 'Managed'}
    assert len(saved_labels) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('records', [[], [{'uuid': 'other', 'changed': True}], [
    {'uuid': 'node-1', 'changed': True}, {'uuid': 'node-1', 'changed': True},
]])
async def test_bulk_opt_in_rejects_missing_or_wrong_receipt(records):
    tx = SimpleNamespace(run=AsyncMock(), run_with_result=AsyncMock(return_value=(records, [], None)))
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )

    with pytest.raises(ValueError):
        await add_nodes_and_edges_bulk_tx(
            tx, [], [], [make_node()], [], SimpleNamespace(), driver,
            maintain_entity_revisions=True,
        )
    tx.run_with_result.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [
    {'uuid': 'node-1', 'changed': False},
    {'uuid': 'node-1', 'revision': None, 'changed': True},
    {'uuid': 'node-1', 'revision': 0, 'changed': True},
    {'uuid': 'node-1', 'revision': 1.5, 'changed': True},
    {'uuid': 'node-1', 'revision': '1', 'changed': True},
    {'uuid': 'node-1', 'revision': True, 'changed': False},
    {'uuid': 'node-1', 'revision': 1, 'changed': False, 'extra': 'unexpected'},
])
async def test_bulk_opt_in_rejects_non_live_revision_receipt(record):
    tx = SimpleNamespace(
        run=AsyncMock(), run_with_result=AsyncMock(return_value=([record], [], None)),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )

    with pytest.raises(ValueError):
        await add_nodes_and_edges_bulk_tx(
            tx, [], [], [make_node()], [], SimpleNamespace(), driver,
            maintain_entity_revisions=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [
    {'uuid': 'node-1', 'revision': 1, 'changed': True},
    {'uuid': 'node-1', 'revision': 2, 'changed': True},
    {'uuid': 'node-1', 'revision': None, 'changed': False},
])
async def test_bulk_opt_in_returns_exact_live_entity_receipt(record):
    tx = SimpleNamespace(
        run=AsyncMock(), run_with_result=AsyncMock(return_value=([record], [], None)),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )

    result = await add_nodes_and_edges_bulk_tx(
        tx, [], [], [make_node()], [], SimpleNamespace(), driver,
        maintain_entity_revisions=True,
    )

    assert result == [record]


@pytest.mark.asyncio
async def test_bulk_opt_in_uses_one_result_bearing_query_and_keeps_other_writes():
    tx = SimpleNamespace(
        run=AsyncMock(),
        run_with_result=AsyncMock(return_value=([
            {'uuid': 'node-1', 'revision': 1, 'changed': False},
        ], [], None)),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )

    await add_nodes_and_edges_bulk_tx(
        tx, [], [], [make_node()], [], SimpleNamespace(), driver,
        maintain_entity_revisions=True,
    )

    tx.run_with_result.assert_awaited_once()
    assert tx.run.await_count == 2


@pytest.mark.asyncio
async def test_default_bulk_keeps_one_query_per_label():
    tx = SimpleNamespace(run=AsyncMock())
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB, graph_operations_interface=None,
        prepare_episode_for_save=AsyncMock(),
    )

    await add_nodes_and_edges_bulk_tx(
        tx, [], [], [make_node()], [], SimpleNamespace(), driver,
    )

    label_queries = [call.args[0] for call in tx.run.await_args_list if 'SET n:' in call.args[0]]
    assert len(label_queries) == 3


@pytest.mark.asyncio
async def test_falkor_session_result_helper_returns_records():
    graph = SimpleNamespace(query=AsyncMock(return_value=SimpleNamespace(
        header=[(0, 'uuid'), (0, 'changed')], result_set=[['node-1', True]],
    )))
    session = FalkorDriverSession(graph)

    result = await session.run_with_result('RETURN $uuid AS uuid', uuid='node-1')

    assert result[0] == [{'uuid': 'node-1', 'changed': True}]
