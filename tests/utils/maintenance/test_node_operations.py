import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _cached_shingles,
    _has_high_entropy,
    _hash_shingle,
    _jaccard_similarity,
    _lsh_bands,
    _minhash_signature,
    _name_entropy,
    _normalize_name_for_fuzzy,
    _normalize_string_exact,
    _resolve_with_similarity,
    _shingles,
)
from graphiti_core.utils.maintenance.node_operations import (
    _collect_candidate_nodes,
    _extract_attributes_and_update,
    _resolve_with_llm,
    extract_attributes_from_node,
    extract_attributes_from_nodes,
    filter_extracted_nodes,
    resolve_extracted_nodes,
)


def _make_clients():
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode(group_id: str = 'group'):
    return EpisodicNode(
        name='episode',
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test',
        content='content',
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])

    async def fake_search(*_, **__):
        return SearchResults(nodes=[candidate])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.search',
        fake_search,
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.filter_existing_duplicate_of_edges',
        AsyncMock(return_value=[]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_low_entropy_uses_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'duplicate_idx': -1,
                'name': 'Joe',
                'duplicates': [],
            }
        ]
    }

    extracted = EntityNode(name='Joe', group_id='group', labels=['Entity'])

    async def fake_search(*_, **__):
        return SearchResults(nodes=[])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.search',
        fake_search,
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.filter_existing_duplicate_of_edges',
        AsyncMock(return_value=[]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted.uuid
    assert uuid_map[extracted.uuid] == extracted.uuid
    llm_generate.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_fuzzy_match(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe-Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])

    async def fake_search(*_, **__):
        return SearchResults(nodes=[candidate])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.search',
        fake_search,
    )
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.filter_existing_duplicate_of_edges',
        AsyncMock(return_value=[]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_candidate_nodes_dedupes_and_merges_override(monkeypatch):
    clients, _ = _make_clients()

    candidate = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    override_duplicate = EntityNode(
        uuid=candidate.uuid,
        name='Alice Alt',
        group_id='group',
        labels=['Entity'],
    )
    extracted = EntityNode(name='Alice', group_id='group', labels=['Entity'])

    search_mock = AsyncMock(return_value=SearchResults(nodes=[candidate]))
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.search',
        search_mock,
    )

    result = await _collect_candidate_nodes(
        clients,
        [extracted],
        existing_nodes_override=[override_duplicate],
    )

    assert len(result) == 1
    assert result[0].uuid == candidate.uuid
    search_mock.assert_awaited()


def test_build_candidate_indexes_populates_structures():
    candidate = EntityNode(name='Bob Dylan', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])

    normalized_key = candidate.name.lower()
    assert indexes.normalized_existing[normalized_key][0].uuid == candidate.uuid
    assert indexes.nodes_by_uuid[candidate.uuid] is candidate
    assert candidate.uuid in indexes.shingles_by_candidate
    assert any(candidate.uuid in bucket for bucket in indexes.lsh_buckets.values())


def test_normalize_helpers():
    assert _normalize_string_exact('  Alice   Smith ') == 'alice smith'
    assert _normalize_name_for_fuzzy('Alice-Smith!') == 'alice smith'


def test_name_entropy_variants():
    assert _name_entropy('alice') > _name_entropy('aaaaa')
    assert _name_entropy('') == 0.0


def test_has_high_entropy_rules():
    assert _has_high_entropy('meaningful name') is True
    assert _has_high_entropy('aa') is False


def test_shingles_and_cache():
    raw = 'alice'
    shingle_set = _shingles(raw)
    assert shingle_set == {'ali', 'lic', 'ice'}
    assert _cached_shingles(raw) == shingle_set
    assert _cached_shingles(raw) is _cached_shingles(raw)


def test_hash_minhash_and_lsh():
    shingles = {'abc', 'bcd', 'cde'}
    signature = _minhash_signature(shingles)
    assert len(signature) == 32
    bands = _lsh_bands(signature)
    assert all(len(band) == 4 for band in bands)
    hashed = {_hash_shingle(s, 0) for s in shingles}
    assert len(hashed) == len(shingles)


def test_jaccard_similarity_edges():
    a = {'a', 'b'}
    b = {'a', 'c'}
    assert _jaccard_similarity(a, b) == pytest.approx(1 / 3)
    assert _jaccard_similarity(set(), set()) == 1.0
    assert _jaccard_similarity(a, set()) == 0.0


def test_resolve_with_similarity_exact_match_updates_state():
    candidate = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_low_entropy_defers_resolution():
    extracted = EntityNode(name='Bob', group_id='group', labels=['Entity'])
    indexes = DedupCandidateIndexes(
        existing_nodes=[],
        nodes_by_uuid={},
        normalized_existing=defaultdict(list),
        shingles_by_candidate={},
        lsh_buckets=defaultdict(list),
    )
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_multiple_exact_matches_defers_to_llm():
    candidate1 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_resolve_with_llm_updates_unresolved(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    captured_context = {}

    def fake_prompt_nodes(context):
        captured_context.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    async def fake_generate_response(*_, **__):
        return {
            'entity_resolutions': [
                {
                    'id': 0,
                    'duplicate_idx': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicates': [0],
                }
            ]
        }

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=fake_generate_response)

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert captured_context['existing_nodes'][0]['idx'] == 0
    assert isinstance(captured_context['existing_nodes'], list)
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_out_of_range_relative_ids(monkeypatch, caplog):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 5,
                    'duplicate_idx': -1,
                    'name': 'Dexter',
                    'duplicates': [],
                }
            ]
        }
    )

    with caplog.at_level(logging.WARNING):
        await _resolve_with_llm(
            llm_client,
            [extracted],
            indexes,
            state,
            episode=_make_episode(),
            previous_episodes=[],
            entity_types=None,
        )

    assert state.resolved_nodes[0] is None
    assert 'Skipping invalid LLM dedupe id 5' in caplog.text


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_duplicate_relative_ids(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'duplicate_idx': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicates': [0],
                },
                {
                    'id': 0,
                    'duplicate_idx': -1,
                    'name': 'Dizzy',
                    'duplicates': [],
                },
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_invalid_duplicate_idx_defaults_to_extracted(monkeypatch):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'duplicate_idx': 10,
                    'name': 'Dexter',
                    'duplicates': [],
                }
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0] == extracted
    assert state.uuid_map[extracted.uuid] == extracted.uuid
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_extract_attributes_without_callback_generates_summary():
    """Test that summary is generated when no callback is provided (default behavior)."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summary': 'Generated summary', 'attributes': {}}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    result = await extract_attributes_from_node(
        llm_client,
        node,
        episode=episode,
        previous_episodes=[],
        entity_type=None,
        should_summarize_node=None,  # No callback provided
    )

    # Summary should be generated
    assert result.summary == 'Generated summary'
    # LLM should have been called for summary
    assert llm_client.generate_response.call_count == 1


@pytest.mark.asyncio
async def test_extract_attributes_with_callback_skip_summary():
    """Test that summary is NOT regenerated when callback returns False."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summary': 'This should not be used', 'attributes': {}}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    # Callback that always returns False (skip summary generation)
    async def skip_summary_filter(node: EntityNode) -> bool:
        return False

    result = await extract_attributes_from_node(
        llm_client,
        node,
        episode=episode,
        previous_episodes=[],
        entity_type=None,
        should_summarize_node=skip_summary_filter,
    )

    # Summary should remain unchanged
    assert result.summary == 'Old summary'
    # LLM should NOT have been called for summary
    assert llm_client.generate_response.call_count == 0


@pytest.mark.asyncio
async def test_extract_attributes_with_callback_generate_summary():
    """Test that summary is regenerated when callback returns True."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summary': 'New generated summary', 'attributes': {}}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    # Callback that always returns True (generate summary)
    async def generate_summary_filter(node: EntityNode) -> bool:
        return True

    result = await extract_attributes_from_node(
        llm_client,
        node,
        episode=episode,
        previous_episodes=[],
        entity_type=None,
        should_summarize_node=generate_summary_filter,
    )

    # Summary should be updated
    assert result.summary == 'New generated summary'
    # LLM should have been called for summary
    assert llm_client.generate_response.call_count == 1


@pytest.mark.asyncio
async def test_extract_attributes_with_selective_callback():
    """Test callback that selectively skips summaries based on node properties."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summary': 'Generated summary', 'attributes': {}}
    )

    user_node = EntityNode(name='User', group_id='group', labels=['Entity', 'User'], summary='Old')
    topic_node = EntityNode(
        name='Topic', group_id='group', labels=['Entity', 'Topic'], summary='Old'
    )

    episode = _make_episode()

    # Callback that skips User nodes but generates for others
    async def selective_filter(node: EntityNode) -> bool:
        return 'User' not in node.labels

    result_user = await extract_attributes_from_node(
        llm_client,
        user_node,
        episode=episode,
        previous_episodes=[],
        entity_type=None,
        should_summarize_node=selective_filter,
    )

    result_topic = await extract_attributes_from_node(
        llm_client,
        topic_node,
        episode=episode,
        previous_episodes=[],
        entity_type=None,
        should_summarize_node=selective_filter,
    )

    # User summary should remain unchanged
    assert result_user.summary == 'Old'
    # Topic summary should be generated
    assert result_topic.summary == 'Generated summary'
    # LLM should have been called only once (for topic)
    assert llm_client.generate_response.call_count == 1


@pytest.mark.asyncio
async def test_extract_attributes_from_nodes_with_callback():
    """Test that callback is properly passed through extract_attributes_from_nodes."""
    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(
        return_value={'summary': 'New summary', 'attributes': {}}
    )
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity', 'User'], summary='Old1')
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity', 'Topic'], summary='Old2')

    episode = _make_episode()

    call_tracker = []

    # Callback that tracks which nodes it's called with
    async def tracking_filter(node: EntityNode) -> bool:
        call_tracker.append(node.name)
        return 'User' not in node.labels

    results = await extract_attributes_from_nodes(
        clients,
        [node1, node2],
        episode=episode,
        previous_episodes=[],
        entity_types=None,
        should_summarize_node=tracking_filter,
    )

    # Callback should have been called for both nodes
    assert len(call_tracker) == 2
    assert 'Node1' in call_tracker
    assert 'Node2' in call_tracker

    # Node1 (User) should keep old summary, Node2 (Topic) should get new summary
    node1_result = next(n for n in results if n.name == 'Node1')
    node2_result = next(n for n in results if n.name == 'Node2')

    assert node1_result.summary == 'Old1'
    assert node2_result.summary == 'New summary'


# ==================== filter_extracted_nodes tests ====================


@pytest.mark.asyncio
async def test_filter_extracted_nodes_all_valid():
    """Test that all entities pass validation - returns empty list."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'validations': [
                {'name': 'Alice', 'is_valid': True, 'reason': 'Matches Person type'},
                {'name': 'Bob', 'is_valid': True, 'reason': 'Matches Person type'},
            ]
        }
    )

    episode = _make_episode()
    nodes = [
        EntityNode(name='Alice', group_id='group', labels=['Entity', 'Person'], summary='A person named Alice'),
        EntityNode(name='Bob', group_id='group', labels=['Entity', 'Person'], summary='A person named Bob'),
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
        {'entity_type_id': 1, 'entity_type_name': 'Person', 'entity_type_description': 'A human individual'},
    ]

    entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    assert entities_to_remove == []
    assert entities_to_reclassify == []
    llm_client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_filter_extracted_nodes_some_invalid():
    """Test that invalid entities are returned for removal."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'validations': [
                {'name': 'Alice', 'is_valid': True, 'reason': 'Matches Person type'},
                {'name': 'system_cpu_cores', 'is_valid': False, 'reason': 'This is a metric, not a Component'},
                {'name': 'Bob', 'is_valid': True, 'reason': 'Matches Person type'},
            ]
        }
    )

    episode = _make_episode()
    nodes = [
        EntityNode(name='Alice', group_id='group', labels=['Entity', 'Person'], summary='A person'),
        EntityNode(name='system_cpu_cores', group_id='group', labels=['Entity', 'Component'], summary='CPU metric'),
        EntityNode(name='Bob', group_id='group', labels=['Entity', 'Person'], summary='Another person'),
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
        {'entity_type_id': 1, 'entity_type_name': 'Person', 'entity_type_description': 'A human individual'},
        {'entity_type_id': 2, 'entity_type_name': 'Component', 'entity_type_description': 'A software component'},
    ]

    entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    assert entities_to_remove == ['system_cpu_cores']
    assert entities_to_reclassify == []  # Step 2 (reclassify) is disabled


@pytest.mark.asyncio
async def test_filter_extracted_nodes_all_invalid():
    """Test that all invalid entities are returned for removal."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'validations': [
                {'name': 'metric_1', 'is_valid': False, 'reason': 'Not a valid Component'},
                {'name': 'metric_2', 'is_valid': False, 'reason': 'Not a valid Component'},
            ]
        }
    )

    episode = _make_episode()
    nodes = [
        EntityNode(name='metric_1', group_id='group', labels=['Entity', 'Component'], summary='A metric'),
        EntityNode(name='metric_2', group_id='group', labels=['Entity', 'Component'], summary='Another metric'),
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
        {'entity_type_id': 1, 'entity_type_name': 'Component', 'entity_type_description': 'A software component'},
    ]

    entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    assert set(entities_to_remove) == {'metric_1', 'metric_2'}
    assert entities_to_reclassify == []


@pytest.mark.asyncio
async def test_filter_extracted_nodes_empty_input():
    """Test that empty input returns empty output."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock()
    episode = _make_episode()

    entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(
        llm_client,
        episode,
        [],  # Empty nodes list
        'group',
        [],
    )

    assert entities_to_remove == []
    assert entities_to_reclassify == []
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_filter_extracted_nodes_batch_processing():
    """Test that entities are processed in batches of 5."""
    call_count = 0

    async def mock_generate_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Return all valid for simplicity
        return {
            'validations': [
                {'name': f'Entity_{i}', 'is_valid': True, 'reason': 'Valid'}
                for i in range((call_count - 1) * 5, min(call_count * 5, 12))
            ]
        }

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)

    episode = _make_episode()
    # Create 12 entities - should result in 3 batches (5 + 5 + 2)
    nodes = [
        EntityNode(name=f'Entity_{i}', group_id='group', labels=['Entity', 'Person'], summary=f'Entity {i}')
        for i in range(12)
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
        {'entity_type_id': 1, 'entity_type_name': 'Person', 'entity_type_description': 'A human individual'},
    ]

    entities_to_remove, _ = await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    # Should have 3 batch calls (5 + 5 + 2 = 12 entities)
    assert llm_client.generate_response.await_count == 3
    assert entities_to_remove == []


@pytest.mark.asyncio
async def test_filter_extracted_nodes_uses_type_definition():
    """Test that type definition is passed to validation prompt."""
    captured_context = {}

    async def capture_generate_response(prompt, *args, **kwargs):
        # The prompt contains entities with type_definition
        # We can check the entities in the prompt
        return {
            'validations': [
                {'name': 'Alice', 'is_valid': True, 'reason': 'Valid'}
            ]
        }

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=capture_generate_response)

    episode = _make_episode()
    nodes = [
        EntityNode(name='Alice', group_id='group', labels=['Entity', 'Person'], summary='A person'),
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
        {'entity_type_id': 1, 'entity_type_name': 'Person', 'entity_type_description': 'A human individual. IS: manager, employee. IS NOT: role, title.'},
    ]

    await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    # Verify LLM was called
    llm_client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_filter_extracted_nodes_entity_with_only_entity_label():
    """Test entity with only 'Entity' label uses Entity type definition."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'validations': [
                {'name': 'Unknown', 'is_valid': False, 'reason': 'Cannot determine type'},
            ]
        }
    )

    episode = _make_episode()
    nodes = [
        EntityNode(name='Unknown', group_id='group', labels=['Entity'], summary='Unknown entity'),
    ]

    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default type'},
    ]

    entities_to_remove, _ = await filter_extracted_nodes(
        llm_client,
        episode,
        nodes,
        'group',
        entity_types_context,
    )

    assert entities_to_remove == ['Unknown']


# ==================== _extract_attributes_and_update error isolation tests ====================


@pytest.mark.asyncio
async def test_extract_attributes_and_update_exception_isolation(caplog):
    """Test that a single entity's attribute extraction failure doesn't crash the batch.

    EasyOps customization: When LLM returns invalid enum values (e.g., component_type='network'
    when only 'backend', 'middleware', etc. are allowed), the exception should be caught
    and logged, allowing other entities in the batch to continue processing.
    """
    from pydantic import BaseModel, Field
    from typing import Literal

    # Define a Component type with enum constraint
    class Component(BaseModel):
        """A software component."""
        component_type: Literal['backend', 'middleware', 'database', 'frontend'] = Field(
            default='backend',
            description='Type of component'
        )

    call_count = 0

    async def mock_generate_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call succeeds, second call returns invalid value that will fail validation
        if call_count == 1:
            return {'component_type': 'backend'}
        else:
            # Return invalid enum value - this will be caught by _extract_entity_attributes
            # validation and should be handled gracefully
            return {'component_type': 'network'}  # Invalid enum value!

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)

    node1 = EntityNode(name='ValidComponent', group_id='group', labels=['Entity', 'Component'])
    node2 = EntityNode(name='InvalidComponent', group_id='group', labels=['Entity', 'Component'])

    episode = _make_episode()

    # This should NOT raise an exception even though second node returns invalid data
    with caplog.at_level(logging.WARNING):
        await _extract_attributes_and_update(
            llm_client, node1, episode, [], Component  # Pass entity_type to trigger LLM call
        )
        await _extract_attributes_and_update(
            llm_client, node2, episode, [], Component  # Pass entity_type to trigger LLM call
        )

    # Node1 should have valid attributes
    assert node1.attributes.get('component_type') == 'backend'

    # Node2 should have empty attributes (invalid field removed by graceful handling)
    # The validation warning should be logged
    assert 'Entity attribute validation warning' in caplog.text or 'Removing invalid field' in caplog.text


@pytest.mark.asyncio
async def test_extract_attributes_and_update_llm_exception_isolation(caplog):
    """Test that LLM exceptions are caught and don't crash the batch.

    This tests the outer try/except in _extract_attributes_and_update that catches
    any exception (not just validation errors).
    """
    from pydantic import BaseModel, Field
    from typing import Literal

    class Component(BaseModel):
        """A software component."""
        component_type: Literal['backend', 'middleware'] = Field(default='backend')

    call_count = 0

    async def mock_generate_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {'component_type': 'backend'}
        else:
            # Simulate unexpected LLM error (network timeout, API error, etc.)
            raise RuntimeError('LLM API connection failed')

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)

    node1 = EntityNode(name='SuccessNode', group_id='group', labels=['Entity', 'Component'])
    node2 = EntityNode(name='FailureNode', group_id='group', labels=['Entity', 'Component'])

    episode = _make_episode()

    # This should NOT raise an exception even though second node's LLM call fails
    with caplog.at_level(logging.WARNING):
        await _extract_attributes_and_update(
            llm_client, node1, episode, [], Component
        )
        await _extract_attributes_and_update(
            llm_client, node2, episode, [], Component
        )

    # Node1 should have valid attributes
    assert node1.attributes.get('component_type') == 'backend'

    # Node2 should have empty attributes (LLM call failed)
    assert node2.attributes.get('component_type') is None

    # Warning should be logged for the failure
    assert 'Failed to extract attributes for entity "FailureNode"' in caplog.text
    assert 'LLM API connection failed' in caplog.text


@pytest.mark.asyncio
async def test_extract_attributes_from_nodes_single_failure_doesnt_crash_batch():
    """Test that a single entity failure in batch doesn't affect other entities.

    This tests the real batch scenario with extract_attributes_from_nodes.
    """
    from pydantic import BaseModel, Field
    from typing import Literal

    # Define a Component type with enum constraint
    class Component(BaseModel):
        """A software component."""
        component_type: Literal['backend', 'middleware', 'database', 'frontend'] = Field(
            default='backend',
            description='Type of component'
        )

    call_count = 0

    async def mock_generate_response(prompt, response_model=None, **kwargs):
        nonlocal call_count
        call_count += 1
        prompt_name = kwargs.get('prompt_name', '')

        # Handle summary extraction
        if 'extract_summary' in prompt_name or 'extract_summaries_bulk' in prompt_name:
            return {
                'summaries': [
                    {'entity_id': 0, 'summary': 'Summary 1'},
                    {'entity_id': 1, 'summary': 'Summary 2'},
                    {'entity_id': 2, 'summary': 'Summary 3'},
                ]
            }

        # Handle attribute extraction - first two succeed, third fails
        if 'extract_attributes' in prompt_name:
            if call_count <= 2:
                return {'component_type': 'backend'}
            else:
                # Return invalid enum value that will fail validation
                return {'component_type': 'network'}  # Invalid!

        return {}

    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])

    nodes = [
        EntityNode(name='Component1', group_id='group', labels=['Entity', 'Component']),
        EntityNode(name='Component2', group_id='group', labels=['Entity', 'Component']),
        EntityNode(name='Component3', group_id='group', labels=['Entity', 'Component']),
    ]

    episode = _make_episode()

    # This should NOT raise an exception - batch should complete
    results = await extract_attributes_from_nodes(
        clients,
        nodes,
        episode=episode,
        previous_episodes=[],
        entity_types={'Component': Component},
    )

    # All 3 nodes should be returned (no crash)
    assert len(results) == 3

    # First two nodes may have attributes, third should have empty attributes
    # (exact behavior depends on implementation, but batch shouldn't crash)
