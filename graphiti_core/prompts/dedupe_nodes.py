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

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json


class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. If no duplicate entities are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are a duplicate of the entity with the above id.',
    )
    reasoning: str = Field(
        default='',
        description='Brief explanation of why this entity is or is not a duplicate. Required when duplicate_idx != -1.',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')


# Scoring-based deduplication models
class CandidateMatch(BaseModel):
    """Score for a single candidate match."""
    candidate_idx: int = Field(..., description='Index of the candidate entity in EXISTING ENTITIES')
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Similarity score from 0.0 to 1.0. 1.0 = definitely same entity, 0.0 = definitely different',
    )
    is_same_entity: bool = Field(
        ...,
        description='Whether this candidate refers to the same real-world entity',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of the similarity assessment',
    )


class NodeDuplicateWithScores(BaseModel):
    """Entity deduplication result with similarity scores."""
    id: int = Field(..., description='integer id of the entity from ENTITIES')
    name: str = Field(
        ...,
        description='Best name for the entity',
    )
    candidate_scores: list[CandidateMatch] = Field(
        ...,
        description='Similarity scores for each candidate in EXISTING ENTITIES that was evaluated',
    )
    best_match_idx: int = Field(
        ...,
        description='Index of the best matching candidate (highest score with is_same_entity=true), or -1 if no match',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Confidence in the final deduplication decision (0.0-1.0)',
    )


class NodeResolutionsWithScores(BaseModel):
    """Collection of scored entity deduplication results."""
    entity_resolutions: list[NodeDuplicateWithScores] = Field(
        ..., description='List of resolved nodes with scores'
    )


# Clustering-based deduplication models (EasyOps)
class EntityGroup(BaseModel):
    """A group of entities that refer to the same real-world object."""

    entity_ids: list[int] = Field(
        ..., description='IDs of entities in this group (from ENTITIES list)'
    )
    canonical_id: int = Field(
        ..., description='ID of the entity with the best/most complete name'
    )
    reasoning: str = Field(
        default='', description='Why these entities are the same'
    )


class EntityClustering(BaseModel):
    """Clustering result for entity deduplication."""

    groups: list[EntityGroup] = Field(
        ...,
        description='Groups of duplicate entities. Single entities should also be in their own group.',
    )


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion
    nodes: PromptVersion
    nodes_with_scores: PromptVersion
    cluster_entities: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction
    nodes: PromptFunction
    nodes_with_scores: PromptFunction
    cluster_entities: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether or not a NEW ENTITY is a duplicate of any EXISTING ENTITIES.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {to_prompt_json([ep for ep in context['previous_episodes']])}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>
        <NEW ENTITY>
        {to_prompt_json(context['extracted_node'])}
        </NEW ENTITY>
        <ENTITY TYPE DESCRIPTION>
        {to_prompt_json(context['entity_type_description'])}
        </ENTITY TYPE DESCRIPTION>

        <EXISTING ENTITIES>
        {to_prompt_json(context['existing_nodes'])}
        </EXISTING ENTITIES>
        
        Given the above EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES; Determine if the NEW ENTITY extracted from the conversation
        is a duplicate entity of one of the EXISTING ENTITIES.

        Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
        Semantic Equivalence: if a descriptive label in existing_entities clearly refers to a named entity in context, treat them as duplicates.

        Do NOT mark entities as duplicates if:
        - They are related but distinct.
        - They have similar names or purposes but refer to separate instances or concepts.

         TASK:
         1. Compare `new_entity` against each item in `existing_entities`.
         2. If it refers to the same real-world object or concept, collect its index.
         3. Let `duplicate_idx` = the smallest collected index, or -1 if none.
         4. Let `duplicates` = the sorted list of all collected indices (empty list if none).

        Respond with a JSON object containing an "entity_resolutions" array with a single entry:
        {{
            "entity_resolutions": [
                {{
                    "id": integer id from NEW ENTITY,
                    "name": the best full name for the entity,
                    "duplicate_idx": integer index of the best duplicate in EXISTING ENTITIES, or -1 if none,
                    "duplicates": sorted list of all duplicate indices you collected (deduplicate the list, use [] when none),
                    "reasoning": a brief explanation (1-2 sentences) of why you determined this entity is or is not a duplicate. REQUIRED when duplicate_idx != -1.
                }}
            ]
        }}

        Only reference indices that appear in EXISTING ENTITIES, and return [] / -1 when unsure.
        When marking as duplicate, explain what evidence shows they refer to the same real-world object.
        """,
        ),
    ]


def nodes(context: dict[str, Any]) -> list[Message]:
    # Build entity type definitions section if available
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    alias_instruction = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """
        # EasyOps: Add explicit alias matching instruction when type definitions are available
        alias_instruction = """
        **IMPORTANT - ALIAS MATCHING**:
        The ENTITY TYPE DEFINITIONS above may contain alias patterns using "或" (Chinese "or").
        For example: "(5)EasyITSM 或 ITSM 或 IT服务中心" means EasyITSM, ITSM, and IT服务中心 are ALL aliases for the SAME entity.
        When you see entities with names matching these aliases, they MUST be marked as duplicates.
        """

    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether or not ENTITIES extracted from a conversation are duplicates'
            ' of existing entities.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {to_prompt_json([ep for ep in context['previous_episodes']])}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>

        {type_defs_section}

        Each of the following ENTITIES were extracted from the CURRENT MESSAGE.
        Each entity in ENTITIES is represented as a JSON object with the following structure:
        {{
            id: integer id of the entity,
            name: "name of the entity",
            summary: "brief description of the entity",
            entity_type: ["Entity", "<optional additional label>", ...],
            entity_type_description: "description of the entity type, may contain alias patterns"
        }}

        <ENTITIES>
        {to_prompt_json(context['extracted_nodes'])}
        </ENTITIES>

        <EXISTING ENTITIES>
        {to_prompt_json(context['existing_nodes'])}
        </EXISTING ENTITIES>

        Each entry in EXISTING ENTITIES is an object with the following structure:
        {{
            idx: integer index of the candidate entity (use this when referencing a duplicate),
            name: "name of the candidate entity",
            summary: "brief description of the candidate entity",
            entity_types: ["Entity", "<optional additional label>", ...],
            ...<additional attributes such as code, module_name, or other metadata>
        }}

        For each of the above ENTITIES, determine if the entity is a duplicate of any of the EXISTING ENTITIES.

        Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
        {alias_instruction}
        Do NOT mark entities as duplicates if:
        - They are related but distinct.
        - They have similar names or purposes but refer to separate instances or concepts.

        Task:
        ENTITIES contains {len(context['extracted_nodes'])} entities with IDs 0 through {len(context['extracted_nodes']) - 1}.
        Your response MUST include EXACTLY {len(context['extracted_nodes'])} resolutions with IDs 0 through {len(context['extracted_nodes']) - 1}. Do not skip or add IDs.

        For every entity, return an object with the following keys:
        {{
            "id": integer id from ENTITIES,
            "name": the best full name for the entity (preserve the original name unless a duplicate has a more complete name),
            "duplicate_idx": the idx of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate,
            "duplicates": a sorted list of all idx values from EXISTING ENTITIES that refer to duplicates (deduplicate the list, use [] when none or unsure),
            "reasoning": a brief explanation (1-2 sentences) of why you determined this entity is or is not a duplicate. REQUIRED when duplicate_idx != -1.
        }}

        - Only use idx values that appear in EXISTING ENTITIES.
        - Set duplicate_idx to the smallest idx you collected for that entity, or -1 if duplicates is empty.
        - Never fabricate entities or indices.
        - When marking as duplicate, explain what evidence shows they refer to the EXACT same real-world object.
        - Default to -1 (no duplicate) when the entities are merely related but could be distinct.
        """,
        ),
    ]


def node_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, deduplicate a list of nodes:

        Nodes:
        {to_prompt_json(context['nodes'])}

        Task:
        1. Group nodes together such that all duplicate nodes are in the same list of uuids
        2. All duplicate uuids should be grouped together in the same list
        3. Also return a new summary that synthesizes the summary into a new short summary

        Guidelines:
        1. Each uuid from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one uuid

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "node that is a duplicate of 5d643020624c42fa9de13f97b1b3fa39"],
                    "summary": "Brief summary of the node summaries that appear in the list of names."
                }}
            ]
        }}
        """,
        ),
    ]


def nodes_with_scores(context: dict[str, Any]) -> list[Message]:
    """Deduplication with similarity scoring for each candidate match.

    This version requires the LLM to score ALL candidate entities,
    forcing careful comparison before making a deduplication decision.
    """
    # Build entity type definitions section if available
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """

    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether ENTITIES are duplicates '
            'of existing entities by SCORING each candidate match. '
            'You must evaluate and score EVERY candidate before making a decision.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {to_prompt_json([ep for ep in context['previous_episodes']])}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>

        {type_defs_section}

        Each of the following ENTITIES were extracted from the CURRENT MESSAGE.
        Each entity in ENTITIES is represented as a JSON object with the following structure:
        {{
            id: integer id of the entity,
            name: "name of the entity",
            entity_type: ["Entity", "<optional additional label>", ...]
        }}

        <ENTITIES>
        {to_prompt_json(context['extracted_nodes'])}
        </ENTITIES>

        <EXISTING ENTITIES>
        {to_prompt_json(context['existing_nodes'])}
        </EXISTING ENTITIES>

        Each entry in EXISTING ENTITIES is an object with the following structure:
        {{
            idx: integer index of the candidate entity (use this when referencing a duplicate),
            name: "name of the candidate entity",
            entity_types: ["Entity", "<optional additional label>", ...],
            ...<additional attributes such as summaries or metadata>
        }}

        **TASK**: For each entity in ENTITIES, SCORE its similarity against EACH candidate in EXISTING ENTITIES.

        **SCORING GUIDELINES**:
        - Score 0.9-1.0: Definitely the same entity - same name, same context, same meaning
        - Score 0.7-0.8: Very likely the same entity - minor name variations, clear semantic equivalence
        - Score 0.4-0.6: Possibly related - similar names or concepts but could be different instances
        - Score 0.1-0.3: Unlikely the same - related domain but different entities
        - Score 0.0: Definitely different entities

        **IMPORTANT RULES**:
        - Entities should only be considered duplicates if they refer to the *same real-world object or concept*
        - Do NOT merge entities that are merely related but distinct
        - Do NOT merge entities with similar names but different contexts
        - When in doubt, score lower and set is_same_entity=false

        **OUTPUT FORMAT**:
        For each entity in ENTITIES (IDs 0 through {len(context['extracted_nodes']) - 1}), return:
        {{
            "id": integer id from ENTITIES,
            "name": the best full name for the entity,
            "candidate_scores": [
                {{
                    "candidate_idx": idx from EXISTING ENTITIES,
                    "similarity_score": float from 0.0 to 1.0,
                    "is_same_entity": boolean - true only if you are confident they are the SAME entity,
                    "reasoning": brief explanation of the similarity assessment
                }}
                // Include an entry for EVERY candidate in EXISTING ENTITIES
            ],
            "best_match_idx": idx of the best matching candidate (highest score where is_same_entity=true), or -1 if no match,
            "confidence": your confidence in the final decision (0.0-1.0)
        }}

        Your response MUST include EXACTLY {len(context['extracted_nodes'])} resolutions with IDs 0 through {len(context['extracted_nodes']) - 1}.
        """,
        ),
    ]


def cluster_entities(context: dict[str, Any]) -> list[Message]:
    """Cluster entities into groups where each group represents the same real-world entity.

    EasyOps: This approach avoids the self-matching problem in batch deduplication
    by asking the LLM to group entities instead of comparing them pairwise.
    """
    entity_type = context.get('entity_type', 'Entity')
    type_definition = context.get('type_definition', '')

    type_def_section = ''
    if type_definition:
        type_def_section = f"""
<TYPE DEFINITION>
{entity_type}: "{type_definition}"
</TYPE DEFINITION>
"""

    return [
        Message(
            role='system',
            content='You are a helpful assistant that groups entities by identity. '
            'Entities that refer to the SAME real-world object should be in the same group.',
        ),
        Message(
            role='user',
            content=f"""
You are grouping entities of type "{entity_type}".
Entities referring to the SAME real-world object should be in the same group.

{type_def_section}

<ENTITIES>
{to_prompt_json(context['entities'])}
</ENTITIES>

Each entity has:
- id: unique identifier
- name: entity name
- summary: description of the entity

**TASK**: Group entities that refer to the SAME real-world object.
- Abbreviations, translations, or variations of the same name are the SAME entity
- Different objects, even if similar in nature, are DIFFERENT entities

**RULES**:
1. Check the TYPE DEFINITION for explicit alias patterns
2. Consider summary similarity - same functionality/description = likely same entity
3. Single entities with no duplicates should be in their own group
4. Every entity ID must appear in exactly ONE group

**OUTPUT**: Return groups, where each group contains entity_ids and the canonical_id (the one with the best/most complete name).
""",
        ),
    ]


versions: Versions = {
    'node': node,
    'node_list': node_list,
    'nodes': nodes,
    'nodes_with_scores': nodes_with_scores,
    'cluster_entities': cluster_entities,
}
