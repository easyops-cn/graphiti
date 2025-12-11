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
import os
import time
import typing
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens

        # Store base_url for endpoint detection
        self._base_url = config.base_url
        # Auto-detect if structured outputs should be used based on endpoint
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
        return any(host in base_url for host in official_hosts)

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or self.model or DEFAULT_MODEL
        else:
            return self.model or DEFAULT_MODEL

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        # Ensure "json" keyword exists in messages for APIs that require it (e.g., Alibaba Qwen)
        has_json_keyword = any('json' in str(m.get('content', '')).lower() for m in openai_messages)
        if not has_json_keyword and openai_messages:
            # Add JSON instruction to the last user message or system message
            for i in range(len(openai_messages) - 1, -1, -1):
                if openai_messages[i]['role'] in ('user', 'system'):
                    openai_messages[i]['content'] = str(openai_messages[i]['content']) + '\n\nRespond in JSON format.'
                    break

        try:
            # Use instance-level detection for structured output support
            # Can still be overridden by environment variable
            env_override = os.getenv('LLM_USE_JSON_SCHEMA')
            if env_override is not None:
                use_json_schema = env_override.lower() == 'true'
            else:
                use_json_schema = self._use_structured_outputs

            # Prepare response format
            response_format: dict[str, Any] = {'type': 'json_object'}
            if response_model is not None and use_json_schema:
                schema_name = getattr(response_model, '__name__', 'structured_response')
                json_schema = response_model.model_json_schema()
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                }
            elif response_model is not None:
                # For non-official endpoints: add schema to prompt with example
                json_schema = response_model.model_json_schema()
                defs = json_schema.get('$defs', {})

                def build_example_from_properties(props: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
                    """Recursively build example from schema properties"""
                    result = {}
                    for field_name, field_info in props.items():
                        field_type = field_info.get('type', 'string')
                        # Check for $ref
                        if '$ref' in field_info:
                            ref_name = field_info['$ref'].split('/')[-1]
                            if ref_name in defs:
                                ref_props = defs[ref_name].get('properties', {})
                                result[field_name] = build_example_from_properties(ref_props, defs)
                            else:
                                result[field_name] = {}
                        # Check for enum in anyOf (common pattern for optional enums)
                        elif 'anyOf' in field_info:
                            any_of = field_info['anyOf']
                            for option in any_of:
                                if 'enum' in option:
                                    result[field_name] = f'<one of {option["enum"]}>'
                                    break
                                elif '$ref' in option:
                                    ref_name = option['$ref'].split('/')[-1]
                                    if ref_name in defs:
                                        ref_props = defs[ref_name].get('properties', {})
                                        result[field_name] = build_example_from_properties(ref_props, defs)
                                    break
                            else:
                                result[field_name] = None
                        elif 'enum' in field_info:
                            result[field_name] = f'<one of {field_info["enum"]}>'
                        elif field_type == 'string':
                            result[field_name] = f'<actual {field_name} value>'
                        elif field_type == 'integer':
                            result[field_name] = 0
                        elif field_type == 'number':
                            result[field_name] = 0.0
                        elif field_type == 'boolean':
                            result[field_name] = True
                        elif field_type == 'array':
                            # Build example array item from items schema
                            items = field_info.get('items', {})
                            if '$ref' in items:
                                ref_name = items['$ref'].split('/')[-1]
                                if ref_name in defs:
                                    ref_props = defs[ref_name].get('properties', {})
                                    item_example = build_example_from_properties(ref_props, defs)
                                    result[field_name] = [item_example]
                                else:
                                    result[field_name] = ['<item>']
                            elif items:
                                item_type = items.get('type', 'string')
                                if item_type == 'string':
                                    result[field_name] = ['<item>']
                                elif item_type == 'integer':
                                    result[field_name] = [0]
                                elif item_type == 'object':
                                    item_props = items.get('properties', {})
                                    result[field_name] = [build_example_from_properties(item_props, defs)]
                                else:
                                    result[field_name] = ['<item>']
                            else:
                                result[field_name] = ['<item>']
                        else:
                            result[field_name] = None
                    return result

                # Build example based on schema properties
                example = {}
                enum_constraints = []  # Collect enum constraints for explicit instruction
                properties = json_schema.get('properties', {})

                # First pass: collect enum constraints
                def collect_enum_constraints(props: dict[str, Any], defs: dict[str, Any], prefix: str = ''):
                    """Recursively collect enum constraints from schema"""
                    constraints = []
                    for field_name, field_info in props.items():
                        full_name = f'{prefix}{field_name}' if prefix else field_name
                        # Check for enum in anyOf
                        if 'anyOf' in field_info:
                            for option in field_info['anyOf']:
                                if 'enum' in option:
                                    constraints.append(f'- "{full_name}" must be one of: {option["enum"]}')
                                    break
                        elif 'enum' in field_info:
                            constraints.append(f'- "{full_name}" must be one of: {field_info["enum"]}')
                        # Check array items
                        elif field_info.get('type') == 'array':
                            items = field_info.get('items', {})
                            if '$ref' in items:
                                ref_name = items['$ref'].split('/')[-1]
                                if ref_name in defs:
                                    ref_props = defs[ref_name].get('properties', {})
                                    constraints.extend(collect_enum_constraints(ref_props, defs, f'{full_name}[].'))
                    return constraints

                enum_constraints = collect_enum_constraints(properties, defs)
                example = build_example_from_properties(properties, defs)

                schema_hint = (
                    f'\n\nRespond with a JSON object matching this schema:\n{json.dumps(json_schema, ensure_ascii=False)}'
                    f'\n\nExample response format (replace placeholders with actual extracted values):\n{json.dumps(example, ensure_ascii=False, indent=2)}'
                    f'\n\nDo NOT wrap the response in "properties" or any other container. Return the fields directly at the top level.'
                )
                # Add explicit enum constraints if any
                if enum_constraints:
                    schema_hint += f'\n\nIMPORTANT - Enum constraints (you MUST use exactly one of these values, do NOT use any other value):\n' + '\n'.join(enum_constraints)
                for i in range(len(openai_messages) - 1, -1, -1):
                    if openai_messages[i]['role'] in ('user', 'system'):
                        openai_messages[i]['content'] = str(openai_messages[i]['content']) + schema_hint
                        break

            # Log request before API call
            model = self._get_model_for_size(model_size)
            logger.info(f'[LLM] >>> model={model}, response_model={response_model.__name__ if response_model else None}')
            # Log all messages (full content for debugging)
            for i, msg in enumerate(openai_messages):
                msg_content = str(msg.get('content', ''))
                logger.info(f'[LLM] >>> request msg[{i}] role={msg.get("role")}: {msg_content}')

            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
            elapsed_ms = (time.time() - start_time) * 1000
            result = response.choices[0].message.content or ''
            # Log response after API call (full content)
            logger.info(f'[LLM] <<< response ({elapsed_ms:.0f}ms): {result}')
            parsed = json.loads(result)

            # Validate with response_model if provided (for non-official endpoints)
            if response_model is not None and not use_json_schema:
                try:
                    response_model.model_validate(parsed)
                except Exception as validation_error:
                    # Try to fix common LLM output errors
                    fixed = self._fix_common_json_errors(parsed, response_model)
                    if fixed != parsed:
                        logger.warning(f'[LLM] Fixed JSON output errors, retrying validation')
                        try:
                            response_model.model_validate(fixed)
                            return fixed
                        except Exception:
                            pass
                    # Re-raise original validation error to trigger retry
                    logger.warning(f'[LLM] Validation failed: {validation_error}')
                    raise validation_error

            return parsed
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except openai.APIStatusError as e:
            # Log full HTTP response details for debugging
            logger.error(f'[LLM] API error: status_code={e.status_code}, message={e.message}')
            if hasattr(e, 'response') and e.response is not None:
                try:
                    response_text = e.response.text if hasattr(e.response, 'text') else str(e.response)
                    logger.error(f'[LLM] Full response body: {response_text}')
                except Exception:
                    logger.error(f'[LLM] Could not read response body')
            if hasattr(e, 'body') and e.body is not None:
                logger.error(f'[LLM] Error body: {e.body}')
            raise
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    def _fix_common_json_errors(self, data: Any, response_model: type[BaseModel]) -> Any:
        """
        Fix common JSON output errors from LLM, such as:
        - 'name:' instead of 'name' (extra colon in key)
        - Missing required fields that can be inferred
        """
        if isinstance(data, dict):
            fixed = {}
            for key, value in data.items():
                # Fix keys with trailing colon (e.g., 'name:' -> 'name')
                fixed_key = key.rstrip(':') if key.endswith(':') else key
                fixed[fixed_key] = self._fix_common_json_errors(value, response_model)
            return fixed
        elif isinstance(data, list):
            return [self._fix_common_json_errors(item, response_model) for item in data]
        else:
            return data

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None

            while retry_count <= self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                    return response
                except (RateLimitError, RefusalError):
                    # These errors should not trigger retries
                    span.set_status('error', str(last_error))
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ):
                    # Let OpenAI's client handle these retries
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e

                    # Don't retry if we've hit the max retries
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1

                    # Construct a detailed error message for the LLM
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response, ensuring the output matches '
                        f'the expected format and constraints.'
                    )

                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

            # If we somehow get here, raise the last error
            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
