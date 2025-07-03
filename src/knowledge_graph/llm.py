#!/usr/bin/env python3
"""
Enhanced LLM interaction utilities with provider abstraction.
Supports OpenAI, Mistral AI, Google Gemini, Anthropic Claude, and Ollama.
Provider-agnostic interface with configuration-driven provider selection.
"""

import requests
import json
import re
import logging
import time
import os
from typing import Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# --- Custom Exceptions ---
class QuotaError(Exception):
    """Custom exception for API quota/rate limit errors."""
    pass


class LLMProviderError(Exception):
    """Custom exception for provider-specific errors."""
    pass


class LLMConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


# --- Provider Types ---
class LLMProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


# --- Abstract Base Class for LLM Providers ---
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = None

    def set_session(self, session: requests.Session):
        """Set a requests session for connection pooling."""
        self.session = session

    @abstractmethod
    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        """Prepare the request URL, headers, and payload for the provider."""
        pass

    @abstractmethod
    def _parse_response(self, response_data: dict) -> str:
        """Parse the provider-specific response format."""
        pass

    def call_llm(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generic LLM call implementation."""
        # Override config with kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        timeout = kwargs.get('timeout', self.config.timeout)

        # Prepare request
        url, headers, payload = self._prepare_request(
            user_prompt, system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Make request
        requester = self.session if self.session else requests

        try:
            logger.debug(f"Sending request to {self.config.provider.value} at {url}")
            response = requester.post(url, headers=headers, json=payload, timeout=timeout)

            # Check for quota errors
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                error_msg = f"API Quota/Rate Limit Exceeded (HTTP 429) for {self.config.provider.value}"
                if retry_after:
                    error_msg += f". Retry after {retry_after} seconds."
                logger.warning(error_msg)
                raise QuotaError(error_msg)

            response.raise_for_status()

            # Parse response
            response_data = response.json()
            return self._parse_response(response_data)

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout ({timeout}s) for {self.config.provider.value}"
            logger.error(error_msg)
            raise TimeoutError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error for {self.config.provider.value}: {e}"
            logger.error(error_msg)
            raise LLMProviderError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error for {self.config.provider.value}: {e}"
            logger.error(error_msg)
            raise LLMProviderError(error_msg) from e


# --- Provider Implementations ---
class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""

    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)

        url = f"{self.config.base_url}/{self.config.model}:generateContent?key={self.config.api_key}"

        headers = {'Content-Type': 'application/json'}

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        return url, headers, payload

    def _parse_response(self, response_data: dict) -> str:
        candidates = response_data.get("candidates", [])
        if not candidates:
            # Check for quota error in response body
            if "error" in response_data and "exhausted" in response_data["error"].get("message", "").lower():
                raise QuotaError(f"Gemini quota exhausted: {response_data['error']}")
            logger.warning(f"No candidates in Gemini response: {response_data}")
            return ""

        candidate = candidates[0]

        # Check for safety blocks
        if candidate.get("finishReason") == "SAFETY":
            logger.warning("Gemini response blocked by safety settings")
            return "[ERROR: Response blocked by safety settings]"

        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if parts:
            return parts[0].get("text", "")

        logger.warning(f"Could not extract text from Gemini response: {response_data}")
        return ""


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)

        url = self.config.base_url

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return url, headers, payload

    def _parse_response(self, response_data: dict) -> str:
        if "error" in response_data:
            error_msg = response_data['error'].get("message", "").lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                raise QuotaError(f"OpenAI quota/rate limit: {response_data['error']}")
            else:
                raise LLMProviderError(f"OpenAI API error: {response_data['error']}")

        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")

        logger.warning(f"Could not extract content from OpenAI response: {response_data}")
        return ""


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)

        url = self.config.base_url

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.config.api_key,
            'anthropic-version': '2023-06-01'
        }

        messages = [{"role": "user", "content": user_prompt}]

        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            payload["system"] = system_prompt

        return url, headers, payload

    def _parse_response(self, response_data: dict) -> str:
        if "error" in response_data:
            error_msg = response_data['error'].get("message", "").lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                raise QuotaError(f"Anthropic quota/rate limit: {response_data['error']}")
            else:
                raise LLMProviderError(f"Anthropic API error: {response_data['error']}")

        content = response_data.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")

        logger.warning(f"Could not extract content from Anthropic response: {response_data}")
        return ""


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""

    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)

        url = self.config.base_url

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return url, headers, payload

    def _parse_response(self, response_data: dict) -> str:
        if "error" in response_data:
            error_msg = response_data['error'].get("message", "").lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                raise QuotaError(f"Mistral quota/rate limit: {response_data['error']}")
            else:
                raise LLMProviderError(f"Mistral API error: {response_data['error']}")

        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")

        logger.warning(f"Could not extract content from Mistral response: {response_data}")
        return ""


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider implementation."""

    def _prepare_request(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> tuple:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)

        url = self.config.base_url

        headers = {'Content-Type': 'application/json'}

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return url, headers, payload

    def _parse_response(self, response_data: dict) -> str:
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")

        logger.warning(f"Could not extract content from Ollama response: {response_data}")
        return ""


# --- Provider Factory ---
class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers = {
        LLMProvider.GEMINI: GeminiProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.MISTRAL: MistralProvider,
        LLMProvider.OLLAMA: OllamaProvider,
    }

    @classmethod
    def create_provider(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create a provider instance from configuration."""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise LLMConfigurationError(f"Unknown provider: {config.provider}")

        return provider_class(config)

    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseLLMProvider:
        """Create a provider from a dictionary configuration."""
        provider_name = config_dict.get('provider')
        if not provider_name:
            raise LLMConfigurationError("Provider name not specified in configuration")

        try:
            provider_enum = LLMProvider(provider_name.lower())
        except ValueError:
            raise LLMConfigurationError(f"Invalid provider name: {provider_name}")

        # Get API key from environment or config
        api_key = config_dict.get('api_key')
        if not api_key:
            # Try to get from environment based on provider
            env_key_map = {
                LLMProvider.GEMINI: ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
                LLMProvider.OPENAI: ['OPENAI_API_KEY'],
                LLMProvider.ANTHROPIC: ['ANTHROPIC_API_KEY'],
                LLMProvider.MISTRAL: ['MISTRAL_API_KEY'],
                LLMProvider.OLLAMA: ['OLLAMA_API_KEY'],  # May not be needed
            }

            for env_var in env_key_map.get(provider_enum, []):
                api_key = os.getenv(env_var)
                if api_key:
                    break

        if not api_key and provider_enum != LLMProvider.OLLAMA:
            raise LLMConfigurationError(f"API key not found for provider {provider_name}")

        config = LLMConfig(
            provider=provider_enum,
            model=config_dict.get('model', 'default'),
            api_key=api_key or 'local',  # For Ollama
            base_url=config_dict.get('base_url', ''),
            max_tokens=config_dict.get('max_tokens', 2000),
            temperature=config_dict.get('temperature', 0.1),
            timeout=config_dict.get('timeout', 60),
            extra_params=config_dict.get('extra_params', {})
        )

        return cls.create_provider(config)


# --- Enhanced LLM Manager ---
class LLMManager:
    """Manager for handling multiple LLM providers with fallback support."""

    def __init__(self, primary_provider: BaseLLMProvider, fallback_providers: List[BaseLLMProvider] = None):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.session = None

    def set_session(self, session: requests.Session):
        """Set requests session for all providers."""
        self.session = session
        self.primary_provider.set_session(session)
        for provider in self.fallback_providers:
            provider.set_session(session)

    def call_llm(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Call LLM with fallback support."""
        providers = [self.primary_provider] + self.fallback_providers

        last_error = None
        for i, provider in enumerate(providers):
            try:
                logger.debug(f"Attempting LLM call with {provider.config.provider.value} (attempt {i + 1})")
                result = provider.call_llm(user_prompt, system_prompt, **kwargs)

                if i > 0:  # Used fallback
                    logger.info(f"Successfully used fallback provider {provider.config.provider.value}")

                return result

            except QuotaError as e:
                logger.warning(f"Quota error with {provider.config.provider.value}: {e}")
                last_error = e
                continue

            except Exception as e:
                logger.warning(f"Error with {provider.config.provider.value}: {e}")
                last_error = e
                continue

        # All providers failed
        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise LLMProviderError(error_msg)


# --- Utility Functions ---
def create_llm_config_from_env(provider_name: str, model: str = None, base_url: str = None) -> LLMConfig:
    """Create LLM config using environment variables."""
    try:
        provider_enum = LLMProvider(provider_name.lower())
    except ValueError:
        raise LLMConfigurationError(f"Invalid provider name: {provider_name}")

    # Default configurations
    defaults = {
        LLMProvider.GEMINI: {
            'model': 'gemini-1.5-flash-latest',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta/models',
            'env_keys': ['GOOGLE_API_KEY', 'GEMINI_API_KEY']
        },
        LLMProvider.OPENAI: {
            'model': 'gpt-4',
            'base_url': 'https://api.openai.com/v1/chat/completions',
            'env_keys': ['OPENAI_API_KEY']
        },
        LLMProvider.ANTHROPIC: {
            'model': 'claude-3-5-sonnet-20241022',
            'base_url': 'https://api.anthropic.com/v1/messages',
            'env_keys': ['ANTHROPIC_API_KEY']
        },
        LLMProvider.MISTRAL: {
            'model': 'mistral-large-latest',
            'base_url': 'https://api.mistral.ai/v1/chat/completions',
            'env_keys': ['MISTRAL_API_KEY']
        },
        LLMProvider.OLLAMA: {
            'model': 'llama2',
            'base_url': 'http://localhost:11434/v1/chat/completions',
            'env_keys': ['OLLAMA_API_KEY']
        }
    }

    config_defaults = defaults[provider_enum]

    # Get API key from environment
    api_key = None
    for env_var in config_defaults['env_keys']:
        api_key = os.getenv(env_var)
        if api_key:
            break

    if not api_key and provider_enum != LLMProvider.OLLAMA:
        raise LLMConfigurationError(f"No API key found in environment for {provider_name}")

    return LLMConfig(
        provider=provider_enum,
        model=model or config_defaults['model'],
        api_key=api_key or 'local',
        base_url=base_url or config_defaults['base_url']
    )


# --- Backward Compatibility Function ---
def call_llm(
        model: str,
        user_prompt: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None
) -> str:
    """
    Backward compatibility function for existing code.
    Automatically detects provider based on base_url.
    """
    if not base_url:
        raise ValueError("base_url must be provided")

    # Auto-detect provider from base_url
    if "generativelanguage.googleapis.com" in base_url:
        provider = LLMProvider.GEMINI
    elif "api.openai.com" in base_url:
        provider = LLMProvider.OPENAI
    elif "api.anthropic.com" in base_url:
        provider = LLMProvider.ANTHROPIC
    elif "api.mistral.ai" in base_url:
        provider = LLMProvider.MISTRAL
    elif "localhost" in base_url or "ollama" in base_url:
        provider = LLMProvider.OLLAMA
    else:
        # Default to OpenAI-compatible
        provider = LLMProvider.OPENAI

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature
    )

    llm_provider = LLMProviderFactory.create_provider(config)
    if session:
        llm_provider.set_session(session)

    return llm_provider.call_llm(user_prompt, system_prompt, max_tokens=max_tokens, temperature=temperature)


# --- JSON Extraction (unchanged) ---
def fix_malformed_json(text: str) -> str:
    """Fix common JSON formatting issues."""
    text = text.replace(""", "\"").replace(""", "\"")
    text = text.replace("```", "")
    text = re.sub(r'"([^"]*)$', r'"\1"', text)
    return text.strip()

def extract_json_from_text(text):
    """
    Extract JSON array from text that might contain extra content, noise, or markdown.

    Args:
        text: Text that may contain JSON

    Returns:
        The parsed JSON if found, else None
    """

    import json
    import re

    if not text or not isinstance(text, str):
        logger.warning("extract_json_from_text received empty or non-string input.")
        return None

    logger.debug(f"Attempting to extract JSON from text starting with: {text[:100]}...") # Log beginning of text

    # Auto-fix: Replace curly quotes with straight quotes
    text = text.replace("“", "\"").replace("”", "\"")

    # Auto-fix: Remove any unexpected trailing backticks or markdown formatting
    text = re.sub(r'^```(?:json)?\s*', '', text) # Remove starting ```json or ```
    text = re.sub(r'\s*```$', '', text) # Remove ending ```

    # Find the start of the first JSON array or object
    start_bracket = text.find('[')
    start_brace = text.find('{')

    if start_bracket == -1 and start_brace == -1:
        logger.warning("No JSON array '[' or object '{' start found in text.")
        return None

    # Determine if we expect an array or object based on which comes first
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        # Looks like a single JSON object, not an array
        start_char = '{'
        end_char = '}'
        start_idx = start_brace
        logger.debug("Detected potential JSON object start '{'.")
    else:
        # Looks like a JSON array
        start_char = '['
        end_char = ']'
        start_idx = start_bracket
        logger.debug("Detected potential JSON array start '['.")

    # Try to find the matching end character
    bracket_count = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == start_char:
            bracket_count += 1
        elif text[i] == end_char:
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i
                break # Found the matching end

    if end_idx != -1:
        json_str = text[start_idx : end_idx + 1]
        logger.debug(f"Extracted potential JSON string: {json_str[:100]}...")
        try:
            # Basic cleaning before parsing
            json_str = json_str.replace('½', 'half')
            json_str = json_str.replace('°', ' degrees')
            json_str = json_str.replace('@', ' at ')
            # Attempt direct parsing of the extracted string
            parsed_json = json.loads(json_str)
            logger.info("Successfully parsed extracted JSON.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse extracted JSON string: {e}")
            logger.debug(f"Problematic JSON string: {json_str}")
            # Optionally add more fixing logic here if needed, like regex fixes
            # For now, just return None if direct parse fails
            return None
    else:
        logger.warning(f"Could not find matching end character '{end_char}' for start at index {start_idx}.")
        # Attempt recovery only if it started with '[' (expecting an array of objects)
        if start_char == '[':
            logger.info("Attempting recovery of JSON objects within the text...")
            objects = []
            # Regex to find structures that look like {"key": "value", ...}
            # This is a simplified regex and might need refinement
            obj_pattern = re.compile(r'\{\s*"[^"]+"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]+"\s*:\s*"[^"]*")*\s*\}')
            for match in obj_pattern.finditer(text):
                objects.append(match.group(0))

            if objects:
                reconstructed_json_str = "[" + ",".join(objects) + "]"
                logger.info(f"Reconstructed JSON array with {len(objects)} objects.")
                try:
                    parsed_json = json.loads(reconstructed_json_str)
                    logger.info("Successfully parsed reconstructed JSON array.")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"Could not parse reconstructed JSON: {e}")
                    logger.debug(f"Reconstructed JSON string: {reconstructed_json_str}")
                    return None
            else:
                 logger.error("Recovery failed: No valid JSON objects found within the text.")
                 return None
        else:
             logger.error("Could not extract a complete JSON structure.")
             return None
