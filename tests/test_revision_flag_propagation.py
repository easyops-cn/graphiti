import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.tracer import NoOpTracer
from graphiti_core.edges import EntityEdge


def make_graphiti():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        _database='tenant',
        clone=lambda **kwargs: driver,
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = SimpleNamespace(driver=driver)
    graphiti.embedder = SimpleNamespace()
    graphiti.llm_client = SimpleNamespace()
    graphiti.tracer = NoOpTracer()
    graphiti.max_coroutines = None
    graphiti.pre_write_hook = None
    graphiti.store_raw_episode_content = True
    return graphiti


def make_episode(group_id='tenant'):
    return EpisodicNode(
        uuid=str(uuid4()),
        name='episode',
        group_id=group_id,
        labels=[],
        source=EpisodeType.text,
        source_description='test',
        content='content',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
        entity_edges=[],
    )


async def gather_tasks(*awaitables, max_coroutines=None):
    return await asyncio.gather(*awaitables)


def test_graphiti_revision_flags_are_keyword_only_with_legacy_defaults():
    for method_name in ('add_triplet', 'add_episode', 'add_episode_bulk'):
        signature = inspect.signature(getattr(Graphiti, method_name))
        assert signature.parameters['maintain_entity_revisions'].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters['maintain_edge_revisions'].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters['maintain_entity_revisions'].default is False
        assert signature.parameters['maintain_edge_revisions'].default is False


@pytest.mark.asyncio
async def test_process_episode_data_forwards_revision_flags_only_to_entity_and_edge_bulk_save(
    monkeypatch,
):
    graphiti = make_graphiti()
    episode = make_episode()
    node = EntityNode(uuid=str(uuid4()), name='node', group_id='tenant', labels=['Node'])
    edge = EntityEdge(
        uuid=str(uuid4()),
        name='LINKS_TO',
        fact='node links node',
        source_node_uuid=node.uuid,
        target_node_uuid=node.uuid,
        group_id='tenant',
        created_at=datetime.now(timezone.utc),
    )
    save = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', save)
    monkeypatch.setattr(graphiti_module, 'build_episodic_edges', lambda *args: [])

    await graphiti._process_episode_data(
        episode,
        [node],
        [edge],
        datetime.now(timezone.utc),
        maintain_entity_revisions=True,
        maintain_edge_revisions=True,
    )

    save.assert_awaited_once()
    assert save.await_args.kwargs['maintain_entity_revisions'] is True
    assert save.await_args.kwargs['maintain_edge_revisions'] is True


@pytest.mark.asyncio
async def test_add_episode_forwards_explicit_flags_and_omitted_flags_stay_disabled(monkeypatch):
    graphiti = make_graphiti()
    graphiti.retrieve_episodes = AsyncMock(return_value=[])
    monkeypatch.setattr(graphiti_module, 'extract_nodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(graphiti_module, 'extract_edges', AsyncMock(return_value=[]))
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_nodes',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_edges',
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        graphiti_module,
        'extract_attributes_from_nodes',
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        graphiti_module,
        'filter_extracted_nodes',
        AsyncMock(return_value=([], None)),
    )
    monkeypatch.setattr(graphiti_module, 'semaphore_gather', gather_tasks)

    captured = []

    async def process(episode, nodes, edges, now, **kwargs):
        captured.append(kwargs)
        return [], episode

    graphiti._process_episode_data = process
    common = dict(
        name='episode',
        episode_body='content',
        source_description='test',
        reference_time=datetime.now(timezone.utc),
        group_id='tenant',
    )

    await graphiti.add_episode(**common)
    await graphiti.add_episode(
        **common,
        maintain_entity_revisions=True,
        maintain_edge_revisions=True,
    )

    assert captured == [
        {'maintain_entity_revisions': False, 'maintain_edge_revisions': False},
        {'maintain_entity_revisions': True, 'maintain_edge_revisions': True},
    ]


@pytest.mark.asyncio
async def test_add_episode_bulk_flags_only_second_extracted_save(monkeypatch):
    graphiti = make_graphiti()
    save = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', save)
    monkeypatch.setattr(
        graphiti_module,
        'retrieve_previous_episodes_bulk',
        AsyncMock(side_effect=lambda driver, episodes: [(episode, []) for episode in episodes]),
    )

    async def extract(episode_context, *args):
        episode = episode_context[0][0]
        return {episode.uuid: []}, {}, [[]]

    graphiti._extract_and_dedupe_nodes_bulk = extract
    graphiti._resolve_nodes_and_edges_bulk = AsyncMock(return_value=([], [], [], {}))
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', AsyncMock(return_value={}))
    monkeypatch.setattr(graphiti_module, 'semantic_dedupe_nodes_bulk', AsyncMock(return_value=([], {})))
    monkeypatch.setattr(graphiti_module, 'build_episodic_edges', lambda *args: [])
    monkeypatch.setattr(graphiti_module, 'resolve_edge_pointers', lambda edges, uuid_map: edges)

    await graphiti.add_episode_bulk(
        [graphiti_module.RawEpisode(
            name='episode',
            content='content',
            source_description='test',
            source=EpisodeType.text,
            reference_time=datetime.now(timezone.utc),
        )],
        group_id='tenant',
        maintain_entity_revisions=True,
        maintain_edge_revisions=True,
    )

    assert save.await_count == 2
    assert 'maintain_entity_revisions' not in save.await_args_list[0].kwargs
    assert 'maintain_edge_revisions' not in save.await_args_list[0].kwargs
    assert save.await_args_list[1].kwargs['maintain_entity_revisions'] is True
    assert save.await_args_list[1].kwargs['maintain_edge_revisions'] is True


@pytest.mark.asyncio
async def test_add_triplet_forwards_explicit_flags_once(monkeypatch):
    graphiti = make_graphiti()
    source = EntityNode(uuid=str(uuid4()), name='source', group_id='tenant', labels=['Node'], name_embedding=[0.1])
    target = EntityNode(uuid=str(uuid4()), name='target', group_id='tenant', labels=['Node'], name_embedding=[0.1])
    edge = EntityEdge(
        uuid=str(uuid4()),
        name='LINKS_TO',
        fact='source links target',
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id='tenant',
        created_at=datetime.now(timezone.utc),
        fact_embedding=[0.1],
    )
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_nodes',
        AsyncMock(return_value=([source, target], {}, [])),
    )
    monkeypatch.setattr(graphiti_module.EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))
    monkeypatch.setattr(graphiti_module, 'search', AsyncMock(return_value=SimpleNamespace(edges=[])))
    monkeypatch.setattr(
        graphiti_module,
        'resolve_extracted_edge',
        AsyncMock(return_value=(edge, [], None)),
    )
    monkeypatch.setattr(graphiti_module, 'create_entity_edge_embeddings', AsyncMock())
    monkeypatch.setattr(graphiti_module, 'create_entity_node_embeddings', AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', save)

    await graphiti.add_triplet(
        source,
        edge,
        target,
        maintain_entity_revisions=True,
        maintain_edge_revisions=True,
    )

    save.assert_awaited_once()
    assert save.await_args.kwargs['maintain_entity_revisions'] is True
    assert save.await_args.kwargs['maintain_edge_revisions'] is True


@pytest.mark.asyncio
async def test_add_triplet_keeps_structured_type_and_maturity_before_revision_save(monkeypatch):
    graphiti = make_graphiti()
    source = EntityNode(uuid=str(uuid4()), name='source', group_id='tenant', labels=['Node'], name_embedding=[0.1])
    target = EntityNode(uuid=str(uuid4()), name='target', group_id='tenant', labels=['Node'], name_embedding=[0.1])
    requested = EntityEdge(
        uuid=str(uuid4()), name='OWNS', fact='source owns target',
        source_node_uuid=source.uuid, target_node_uuid=target.uuid,
        group_id='tenant', created_at=datetime.now(timezone.utc),
        fact_embedding=[0.1], attributes={'maturity_level': 2},
    )
    other_type = requested.model_copy(update={'uuid': str(uuid4()), 'name': 'RELATES_TO'})
    same_type = requested.model_copy(update={
        'uuid': str(uuid4()), 'attributes': {'maturity_level': 0},
    })
    monkeypatch.setattr(
        graphiti_module, 'resolve_extracted_nodes',
        AsyncMock(return_value=([source, target], {}, [])),
    )
    monkeypatch.setattr(
        graphiti_module.EntityEdge, 'get_between_nodes',
        AsyncMock(return_value=[other_type, same_type]),
    )

    async def search_candidates(clients, fact, *, search_filter, **kwargs):
        if search_filter.edge_uuids is not None:
            return SimpleNamespace(edges=[
                candidate for candidate in [other_type, same_type]
                if candidate.uuid in search_filter.edge_uuids
            ])
        return SimpleNamespace(edges=[])

    monkeypatch.setattr(graphiti_module, 'search', search_candidates)
    captured = {}

    async def resolve(llm_client, extracted, related, existing, *args):
        captured['related'] = related
        same_type.name = 'RELATES_TO'
        same_type.attributes = {}
        return same_type, [], None

    monkeypatch.setattr(graphiti_module, 'resolve_extracted_edge', resolve)
    monkeypatch.setattr(graphiti_module, 'create_entity_edge_embeddings', AsyncMock())
    monkeypatch.setattr(graphiti_module, 'create_entity_node_embeddings', AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', save)

    result = await graphiti.add_triplet(
        source, requested, target, maintain_edge_revisions=True,
    )

    assert [candidate.uuid for candidate in captured['related']] == [same_type.uuid]
    assert result.edges[0].uuid == same_type.uuid
    assert result.edges[0].name == 'OWNS'
    assert result.edges[0].attributes['maturity_level'] == 2
    assert save.await_args.args[4][0].attributes['maturity_level'] == 2
    assert save.await_args.kwargs['maintain_edge_revisions'] is True
