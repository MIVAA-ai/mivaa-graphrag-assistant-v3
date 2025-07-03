import logging
from typing import Optional, List, Dict, Any

# ENHANCED: Import multi-provider LLM system with fallback
try:
    from src.knowledge_graph.llm import (
        LLMManager,
        LLMProviderFactory,
        LLMConfig,
        LLMProvider,
        LLMProviderError,
        QuotaError
    )

    NEW_LLM_SYSTEM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Multi-provider LLM system available for Cypher correction")
except ImportError as e:
    NEW_LLM_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Multi-provider LLM system not available: {e}. Using legacy system.")

# LlamaIndex imports for type hinting (adjust if your specific types differ)
# Using base types for broader compatibility, replace with specific classes if known
# e.g., from llama_index.llms.openai import OpenAI
# e.g., from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.llms.llm import LLM
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.prompts import ChatPromptTemplate  # Corrected import path

# Setup logger for this module
# Configure logging level and format in your main application entry point
logger = logging.getLogger(__name__)

# System prompt defining the LLM's role as a Cypher expert correcting errors.
CORRECT_CYPHER_SYSTEM_TEMPLATE = """You are a Cypher expert reviewing a statement written by a junior developer.
You need to correct the Cypher statement based on the provided errors. No pre-amble."
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

# User prompt providing the context (schema, original query, errors) for correction.
CORRECT_CYPHER_USER_TEMPLATE = """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """

# ENHANCED: Global configuration for multi-provider LLM
_global_llm_config = None
_cypher_correction_llm_manager = None


def initialize_cypher_correction_llm(config: Dict[str, Any]) -> bool:
    """
    ENHANCED: Initialize multi-provider LLM system for Cypher correction.

    Args:
        config: Configuration dictionary for multi-provider LLM

    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _global_llm_config, _cypher_correction_llm_manager

    if not NEW_LLM_SYSTEM_AVAILABLE:
        logger.warning("Multi-provider LLM system not available for Cypher correction")
        return False

    try:
        _global_llm_config = config

        # Import the main LLM configuration manager
        from GraphRAG_Document_AI_Platform import get_llm_config_manager

        main_llm_manager = get_llm_config_manager(config)
        _cypher_correction_llm_manager = main_llm_manager.get_llm_manager('cypher_correction')

        logger.info("✅ Multi-provider LLM system initialized for Cypher correction")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize multi-provider LLM for Cypher correction: {e}")
        _cypher_correction_llm_manager = None
        return False


def _enhanced_llm_call(prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
    """
    ENHANCED: Call multi-provider LLM with fallback to None if not available.

    Args:
        prompt: User prompt
        system_prompt: System prompt (optional)
        **kwargs: Additional LLM parameters

    Returns:
        str: LLM response or None if failed
    """
    if _cypher_correction_llm_manager and NEW_LLM_SYSTEM_AVAILABLE:
        try:
            logger.debug("🎯 Using enhanced LLM system for Cypher correction")
            return _cypher_correction_llm_manager.call_llm(
                user_prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Enhanced LLM failed for Cypher correction: {e}")

    logger.debug("🔄 Enhanced LLM not available for Cypher correction")
    return None


async def correct_cypher_step(
        llm: LLM,
        graph_store: GraphStore,
        subquery: str,
        cypher: str,
        errors: str,
        schema_exclude_types: Optional[List[str]] = None,  # Default to None for broader compatibility
        config: Optional[Dict[str, Any]] = None,  # ENHANCED: Optional config for multi-provider LLM
        use_enhanced_llm: bool = True  # ENHANCED: Whether to try enhanced LLM first
) -> Optional[str]:
    """
    ENHANCED: Uses an LLM to correct a given Cypher query based on schema information and error messages.
    Now supports multi-provider LLM with fallback to original LlamaIndex LLM.

    Args:
        llm: The LlamaIndex LLM instance to use for correction (fallback).
        graph_store: The LlamaIndex GraphStore instance providing schema access.
        subquery: The original natural language question or subquery that led to the Cypher.
        cypher: The incorrect Cypher query string.
        errors: The error message(s) received when executing the incorrect query.
        schema_exclude_types: Optional list of node labels to exclude from the schema string.
                              Defaults to None (include all types). Pass ["Actor", "Director"]
                              if you want to replicate the original behavior.
        config: ENHANCED - Optional configuration for multi-provider LLM
        use_enhanced_llm: ENHANCED - Whether to try enhanced LLM first

    Returns:
        Optional[str]: The corrected Cypher query suggested by the LLM, or None if correction fails.
    """
    # ENHANCED: Initialize enhanced LLM if config provided
    if config and use_enhanced_llm and NEW_LLM_SYSTEM_AVAILABLE:
        initialize_cypher_correction_llm(config)

    # --- Input Validation ---
    # Check if essential arguments are provided and have basic validity
    if not llm:
        logger.error("correct_cypher_step: LLM instance is required.")
        return None
    if not graph_store:
        logger.error("correct_cypher_step: GraphStore instance is required.")
        return None
    if not subquery or not isinstance(subquery, str) or not subquery.strip():
        logger.error("correct_cypher_step: Valid 'subquery' string is required.")
        return None
    if not cypher or not isinstance(cypher, str) or not cypher.strip():
        logger.error("correct_cypher_step: Valid 'cypher' query string is required.")
        return None
    if not errors:  # Allow empty error string, but log if None
        logger.warning("correct_cypher_step: 'errors' argument is missing or empty.")
        errors = ""  # Ensure it's at least an empty string
    elif not isinstance(errors, str):
        logger.warning(f"correct_cypher_step: 'errors' argument is not a string ({type(errors)}). Converting.")
        errors = str(errors)  # Convert non-string errors

    logger.info(f"Attempting to correct Cypher for subquery: '{subquery}'")
    logger.debug(f"Original Cypher:\n{cypher}")
    logger.debug(f"Errors:\n{errors}")

    try:
        # --- Schema Retrieval ---
        # Retrieve the graph schema string using the provided graph_store instance.
        # Pass the configurable exclude_types list.
        try:
            schema = graph_store.get_schema()
            logger.debug(f"Using schema (excluding {schema_exclude_types}):\n{schema[:500]}...")  # Log start of schema
        except Exception as schema_e:
            logger.error(f"Failed to retrieve schema from graph_store: {schema_e}", exc_info=True)
            return None  # Cannot proceed without schema

        # ENHANCED: Try multi-provider LLM first
        if use_enhanced_llm and _cypher_correction_llm_manager and NEW_LLM_SYSTEM_AVAILABLE:
            try:
                logger.debug("Attempting Cypher correction with enhanced multi-provider LLM")

                # Format prompt for enhanced LLM
                enhanced_prompt = CORRECT_CYPHER_USER_TEMPLATE.format(
                    question=subquery,
                    schema=schema,
                    errors=errors,
                    cypher=cypher
                )

                # Call enhanced LLM
                enhanced_response = _enhanced_llm_call(
                    prompt=enhanced_prompt,
                    system_prompt=CORRECT_CYPHER_SYSTEM_TEMPLATE,
                    max_tokens=500,
                    temperature=0.1
                )

                if enhanced_response and enhanced_response.strip():
                    corrected_query = enhanced_response.strip()

                    # Validate enhanced response
                    if _validate_cypher_response(corrected_query, cypher):
                        logger.info(f"✅ Enhanced LLM successfully corrected Cypher:\n{corrected_query}")
                        return corrected_query
                    else:
                        logger.warning("Enhanced LLM response validation failed, falling back to original LLM")
                else:
                    logger.warning("Enhanced LLM returned empty response, falling back to original LLM")

            except Exception as enhanced_e:
                logger.warning(f"Enhanced LLM correction failed: {enhanced_e}, falling back to original LLM")

        # --- Fallback to Original LlamaIndex LLM ---
        logger.debug("Using original LlamaIndex LLM for Cypher correction")

        # --- Prompt Formatting ---
        # Define the message structure for the LLM.
        correct_cypher_messages = [
            ("system", CORRECT_CYPHER_SYSTEM_TEMPLATE),
            ("user", CORRECT_CYPHER_USER_TEMPLATE),
        ]
        # Create the prompt template object.
        correct_cypher_prompt = ChatPromptTemplate.from_messages(correct_cypher_messages)

        # --- LLM Call with Error Handling ---
        try:
            # Format the prompt with the specific details (question, schema, errors, original cypher).
            # Call the LLM asynchronously to get the corrected query.
            response = await llm.achat(
                correct_cypher_prompt.format_messages(
                    question=subquery,
                    schema=schema,
                    errors=errors,
                    cypher=cypher,
                )
            )
            # Extract the text content from the response message, handling potential None values.
            corrected_query = response.message.content.strip() if response and response.message and response.message.content else None

        except Exception as llm_e:
            # Log any errors encountered during the LLM API call.
            logger.error(f"LLM call failed during Cypher correction: {llm_e}", exc_info=True)
            return None  # Indicate correction failure due to LLM error

        # --- Output Validation ---
        if corrected_query and _validate_cypher_response(corrected_query, cypher):
            logger.info(f"🔄 Original LLM successfully corrected Cypher:\n{corrected_query}")
            return corrected_query
        else:
            logger.warning("Original LLM correction failed validation")
            return None

    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"Unexpected error in correct_cypher_step: {e}", exc_info=True)
        return None


def _validate_cypher_response(corrected_query: str, original_query: str) -> bool:
    """
    ENHANCED: Validate the corrected Cypher response.

    Args:
        corrected_query: The corrected query from LLM
        original_query: The original query for comparison

    Returns:
        bool: True if response is valid, False otherwise
    """
    # Check if the LLM returned any content.
    if not corrected_query:
        logger.warning("LLM response for correction was empty.")
        return False

    # Basic check if the response looks like a Cypher query.
    # A more robust validation could involve a Cypher parser if available.
    if "MATCH" not in corrected_query.upper() or "RETURN" not in corrected_query.upper():
        logger.warning(
            f"LLM correction response doesn't seem like a valid query (missing MATCH/RETURN): {corrected_query}")
        # Returning False might be safer than returning potentially non-Cypher text.
        return False

    # Check if the LLM simply returned the original query.
    if corrected_query == original_query.strip():
        logger.warning("LLM correction returned the original query. No change made.")
        # Returning False signifies no *useful* correction was made.
        return False

    return True


# ENHANCED: New function for multi-provider Cypher correction
async def correct_cypher_with_multi_provider(
        graph_store: GraphStore,
        subquery: str,
        cypher: str,
        errors: str,
        config: Dict[str, Any],
        schema_exclude_types: Optional[List[str]] = None
) -> Optional[str]:
    """
    ENHANCED: Correct Cypher using only multi-provider LLM (no LlamaIndex fallback).

    Args:
        graph_store: The GraphStore instance providing schema access
        subquery: The original natural language question
        cypher: The incorrect Cypher query string
        errors: The error message(s) received
        config: Configuration for multi-provider LLM
        schema_exclude_types: Optional list of node labels to exclude from schema

    Returns:
        Optional[str]: The corrected Cypher query or None if correction fails
    """
    if not NEW_LLM_SYSTEM_AVAILABLE:
        logger.error("Multi-provider LLM system not available for Cypher correction")
        return None

    # Initialize enhanced LLM
    if not initialize_cypher_correction_llm(config):
        logger.error("Failed to initialize multi-provider LLM for Cypher correction")
        return None

    try:
        # Get schema
        schema = graph_store.get_schema()

        # Format prompt
        prompt = CORRECT_CYPHER_USER_TEMPLATE.format(
            question=subquery,
            schema=schema,
            errors=errors,
            cypher=cypher
        )

        # Call enhanced LLM
        response = _enhanced_llm_call(
            prompt=prompt,
            system_prompt=CORRECT_CYPHER_SYSTEM_TEMPLATE,
            max_tokens=500,
            temperature=0.1
        )

        if response and _validate_cypher_response(response, cypher):
            logger.info(f"✅ Multi-provider LLM corrected Cypher:\n{response}")
            return response
        else:
            logger.warning("Multi-provider LLM correction failed or invalid")
            return None

    except Exception as e:
        logger.error(f"Error in multi-provider Cypher correction: {e}", exc_info=True)
        return None


# ENHANCED: Utility functions for multi-provider LLM management

def get_cypher_correction_provider_info() -> Dict[str, Any]:
    """
    ENHANCED: Get information about the configured LLM provider for Cypher correction.

    Returns:
        Dict containing provider information
    """
    if not NEW_LLM_SYSTEM_AVAILABLE or not _cypher_correction_llm_manager:
        return {
            'multi_provider_enabled': False,
            'system_available': NEW_LLM_SYSTEM_AVAILABLE,
            'provider_info': None
        }

    try:
        primary_provider = _cypher_correction_llm_manager.primary_provider
        fallback_providers = _cypher_correction_llm_manager.fallback_providers

        return {
            'multi_provider_enabled': True,
            'system_available': NEW_LLM_SYSTEM_AVAILABLE,
            'provider_info': {
                'primary_provider': {
                    'name': primary_provider.config.provider.value,
                    'model': primary_provider.config.model,
                    'ready': True
                },
                'fallback_providers': [
                    {
                        'name': fp.config.provider.value,
                        'model': fp.config.model,
                        'ready': True
                    }
                    for fp in fallback_providers
                ]
            }
        }
    except Exception as e:
        return {
            'multi_provider_enabled': False,
            'system_available': NEW_LLM_SYSTEM_AVAILABLE,
            'error': str(e)
        }


def is_enhanced_cypher_correction_available() -> bool:
    """
    ENHANCED: Check if enhanced multi-provider LLM is available for Cypher correction.

    Returns:
        bool: True if enhanced system is available, False otherwise
    """
    return NEW_LLM_SYSTEM_AVAILABLE and _cypher_correction_llm_manager is not None


def reset_cypher_correction_llm():
    """
    ENHANCED: Reset the multi-provider LLM configuration for Cypher correction.
    Useful for testing or reconfiguration.
    """
    global _cypher_correction_llm_manager, _global_llm_config
    _cypher_correction_llm_manager = None
    _global_llm_config = None
    logger.info("Reset multi-provider LLM configuration for Cypher correction")


# ENHANCED: Factory functions for creating enhanced correction functions

def create_enhanced_cypher_corrector(config: Dict[str, Any]):
    """
    ENHANCED: Factory function to create an enhanced Cypher corrector with pre-configured LLM.

    Args:
        config: Configuration for multi-provider LLM

    Returns:
        Function that can be used for Cypher correction
    """
    # Initialize the LLM
    initialize_cypher_correction_llm(config)

    async def corrector(graph_store: GraphStore, subquery: str, cypher: str, errors: str,
                        schema_exclude_types: Optional[List[str]] = None) -> Optional[str]:
        """Pre-configured Cypher corrector function."""
        return await correct_cypher_with_multi_provider(
            graph_store=graph_store,
            subquery=subquery,
            cypher=cypher,
            errors=errors,
            config=config,
            schema_exclude_types=schema_exclude_types
        )

    return corrector


# ENHANCED: Backward compatibility wrapper
async def correct_cypher_step_enhanced(
        llm: LLM,
        graph_store: GraphStore,
        subquery: str,
        cypher: str,
        errors: str,
        schema_exclude_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    ENHANCED: Backward compatible wrapper that automatically enables multi-provider LLM if config is provided.

    This is a drop-in replacement for the original correct_cypher_step function.
    """
    return await correct_cypher_step(
        llm=llm,
        graph_store=graph_store,
        subquery=subquery,
        cypher=cypher,
        errors=errors,
        schema_exclude_types=schema_exclude_types,
        config=config,
        use_enhanced_llm=True
    )


if __name__ == "__main__":
    print("Enhanced Cypher Correction with Multi-Provider LLM Support")
    print("Provides Cypher query correction with multi-provider LLM enhancements")
    print("Backward compatible with existing LlamaIndex-based correction")