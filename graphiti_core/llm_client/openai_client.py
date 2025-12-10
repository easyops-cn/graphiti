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

import json
import logging
import typing

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import DEFAULT_REASONING, DEFAULT_VERBOSITY, BaseOpenAIClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        self._base_url = config.base_url
        self._use_structured_outputs = self._is_official_openai_endpoint(config.base_url)

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _is_official_openai_endpoint(self, base_url: str | None) -> bool:
        """Check if the base_url is an official OpenAI endpoint that supports structured outputs."""
        if base_url is None:
            return True  # Default OpenAI endpoint
        # Official OpenAI endpoints
        official_hosts = ['api.openai.com', 'openai.azure.com']
        is_official = any(host in base_url for host in official_hosts)
        logger.info(f'[OpenAIClient] base_url={base_url}, is_official={is_official}, use_structured_outputs={is_official}')
        return is_official

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a structured completion using OpenAI's beta parse API or fallback for non-official endpoints."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )

        # For non-official endpoints, use fallback to chat.completions with schema in prompt
        if not self._use_structured_outputs:
            logger.info(f'[OpenAIClient] Using fallback for non-official endpoint, model={model}, response_model={response_model.__name__}')
            return await self._create_structured_completion_fallback(
                model, messages, temperature, max_tokens, response_model, is_reasoning_model
            )

        logger.info(f'[OpenAIClient] Using responses.parse API, model={model}')
        response = await self.client.responses.parse(
            model=model,
            input=messages,  # type: ignore
            temperature=temperature if not is_reasoning_model else None,
            max_output_tokens=max_tokens,
            text_format=response_model,  # type: ignore
            reasoning={'effort': reasoning} if reasoning is not None else None,  # type: ignore
            text={'verbosity': verbosity} if verbosity is not None else None,  # type: ignore
        )

        return response

    async def _create_structured_completion_fallback(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        is_reasoning_model: bool,
    ):
        """Fallback for non-official endpoints: add schema to prompt and use chat.completions."""
        # Generate a clean schema description for the prompt
        schema = response_model.model_json_schema()

        # Build a cleaner prompt that describes expected fields
        field_descriptions = []
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            field_desc = field_info.get('description', '')
            is_required = field_name in required

            # Handle enum types
            if 'enum' in field_info:
                enum_values = ', '.join(f'"{v}"' for v in field_info['enum'])
                field_type = f'enum ({enum_values})'

            req_marker = '(required)' if is_required else '(optional)'
            field_descriptions.append(f'  - {field_name}: {field_type} {req_marker} - {field_desc}')

        schema_prompt = (
            f'\n\nRespond with a JSON object containing the following fields:\n'
            + '\n'.join(field_descriptions)
            + '\n\nIMPORTANT: Return ONLY the JSON object with actual values, NOT the schema definition.'
        )

        # Append schema to last message
        modified_messages = list(messages)
        if modified_messages:
            last_msg = modified_messages[-1]
            if isinstance(last_msg, dict) and 'content' in last_msg:
                modified_messages[-1] = {
                    **last_msg,
                    'content': last_msg['content'] + schema_prompt
                }

        response = await self.client.chat.completions.create(
            model=model,
            messages=modified_messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )

        # Wrap response to match structured output format
        class FallbackResponse:
            def __init__(self, content: str):
                self.output_text = content

        content = response.choices[0].message.content or '{}'
        return FallbackResponse(content)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )
