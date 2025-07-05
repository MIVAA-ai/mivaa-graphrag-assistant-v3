# enhanced_graph_rag_qa.py - COMPLETE INTEGRATION LAYER WITH MULTI-PROVIDER LLM SUPPORT AND CYPHER CLEANING

import logging
import time
from typing import Dict, List, Optional, Any

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
    logger.info("✅ Multi-provider LLM system available for enhanced GraphRAG QA")
except ImportError as e:
    NEW_LLM_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Multi-provider LLM system not available: {e}. Using legacy system.")

# Import the universal system
from universal_asset_patterns import (
    UniversalAssetManagementEngine,
    IndustryType,
    DomainAdaptivePatternMatcher
)

# Import your existing GraphRAG system
from graph_rag_qa import GraphRAGQA

logger = logging.getLogger(__name__)


class EnhancedGraphRAGQA(GraphRAGQA):
    """
    Enhanced version that adds universal pattern support AND multi-provider LLM support
    to your existing GraphRAGQA. Completely backward compatible - your existing code
    continues to work unchanged.
    """

    def __init__(self, *args, **kwargs):
        # Extract new enhancement parameters
        self.enable_universal_patterns = kwargs.pop('enable_universal_patterns', True)
        self.manual_industry = kwargs.pop('manual_industry', None)
        self.pattern_confidence_threshold = kwargs.pop('pattern_confidence_threshold', 0.6)

        # ENHANCED: Extract multi-provider LLM parameters
        self.enable_multi_provider_llm = kwargs.pop('enable_multi_provider_llm', True)
        self.config = kwargs.pop('config', {})  # Configuration for multi-provider LLM

        # Initialize the base GraphRAGQA system first (unchanged)
        super().__init__(*args, **kwargs)

        # Add universal enhancements
        self.universal_engine = None
        if self.enable_universal_patterns:
            self._initialize_universal_engine()

        # ENHANCED: Initialize multi-provider LLM system
        self.llm_managers = {}
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            self._initialize_multi_provider_llm()

        logger.info(
            f"Enhanced GraphRAG QA initialized. Universal patterns: {self.enable_universal_patterns}, Multi-provider LLM: {self.enable_multi_provider_llm}")

    def _initialize_universal_engine(self):
        """Initialize the universal pattern engine"""
        try:
            if not self.driver:
                logger.warning("No Neo4j driver available for universal engine")
                return

            self.universal_engine = UniversalAssetManagementEngine(self.driver)

            # Manual industry override if specified
            if self.manual_industry:
                try:
                    industry_enum = IndustryType(self.manual_industry.lower())
                    patterns = self.universal_engine.get_patterns_for_industry(industry_enum)
                    self.universal_engine.adapted_patterns = patterns
                    logger.info(f"Set manual industry context: {self.manual_industry}")
                except ValueError:
                    logger.warning(f"Invalid manual industry: {self.manual_industry}")

            logger.info("Universal pattern engine initialized successfully")

        except Exception as e:
            logger.error(f"Universal engine initialization failed: {e}")
            self.universal_engine = None

    def _initialize_multi_provider_llm(self):
        """ENHANCED: Initialize multi-provider LLM system for GraphRAG QA tasks"""
        try:
            # Import the main LLM configuration manager
            from GraphRAG_Document_AI_Platform import get_llm_config_manager

            main_llm_manager = get_llm_config_manager(self.config)

            # Create task-specific LLM managers for GraphRAG QA
            qa_tasks = ['cypher_generation', 'entity_linking', 'answer_generation', 'query_correction']

            for task in qa_tasks:
                try:
                    self.llm_managers[task] = main_llm_manager.get_llm_manager(task)
                    logger.info(f"✅ Initialized LLM manager for {task}")
                except Exception as e:
                    logger.warning(f"Could not initialize LLM manager for {task}: {e}")
                    self.llm_managers[task] = None

        except ImportError as e:
            logger.warning(f"Could not import main LLM configuration manager: {e}")
            self.llm_managers = {}

    def clean_cypher_query(self, cypher_text: str) -> str:
        """
        FIXED: Clean LLM-generated Cypher query to remove markdown formatting and ensure pure Cypher.

        Args:
            cypher_text: Raw output from LLM that may contain markdown

        Returns:
            Clean Cypher query ready for Neo4j execution
        """
        if not cypher_text or not isinstance(cypher_text, str):
            logger.warning("Empty or invalid Cypher input")
            return ""

        original_cypher = cypher_text
        logger.debug(f"🔧 Cleaning Cypher input: {original_cypher[:100]}...")

        # Remove markdown code blocks
        cypher_text = cypher_text.replace('```cypher\n', '').replace('```cypher', '')
        cypher_text = cypher_text.replace('\n```', '').replace('```', '')

        # Remove any leading/trailing whitespace
        cypher_text = cypher_text.strip()

        # Remove quotes if the entire query is wrapped in quotes
        if ((cypher_text.startswith('"') and cypher_text.endswith('"')) or
                (cypher_text.startswith("'") and cypher_text.endswith("'"))):
            cypher_text = cypher_text[1:-1]

        # Remove any remaining backticks
        cypher_text = cypher_text.replace('`', '')

        # Ensure the query doesn't start with explanatory text
        lines = cypher_text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            # Skip explanatory text - look for actual Cypher keywords
            if any(line.upper().startswith(keyword) for keyword in [
                'MATCH', 'RETURN', 'WHERE', 'WITH', 'CREATE', 'MERGE', 'DELETE',
                'SET', 'REMOVE', 'FOREACH', 'CALL', 'UNION', 'OPTIONAL', 'UNWIND'
            ]):
                cleaned_lines.append(line)
            elif cleaned_lines:  # Only add non-keyword lines if we've already started the query
                cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # Final validation - ensure we have a valid Cypher query
        if not result:
            logger.warning("Cypher cleaning resulted in empty query")
            return ""

        # Basic validation - should start with a Cypher keyword
        first_word = result.split()[0].upper() if result.split() else ""
        valid_start_keywords = [
            'MATCH', 'RETURN', 'WITH', 'CREATE', 'MERGE', 'DELETE',
            'SET', 'REMOVE', 'CALL', 'UNION', 'OPTIONAL', 'UNWIND'
        ]

        if first_word not in valid_start_keywords:
            logger.warning(f"Cypher query doesn't start with valid keyword: {first_word}")
            # Try to find the first valid line
            for line in result.split('\n'):
                first_word_line = line.strip().split()[0].upper() if line.strip().split() else ""
                if first_word_line in valid_start_keywords:
                    result = line.strip()
                    break

        logger.info(f"🔧 Original: {original_cypher[:50]}... → ✅ Cleaned: {result}")
        return result

    def _enhanced_llm_call(self, task_name: str, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        ENHANCED: Enhanced LLM call with multi-provider support.
        Falls back to base system if enhanced LLM is not available.
        """
        # Try enhanced system first
        if (self.enable_multi_provider_llm and
                NEW_LLM_SYSTEM_AVAILABLE and
                task_name in self.llm_managers and
                self.llm_managers[task_name]):

            try:
                logger.debug(f"🎯 Using enhanced LLM system for {task_name}")
                return self.llm_managers[task_name].call_llm(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Enhanced LLM failed for {task_name}: {e}, falling back to base system")

        # Fall back to base GraphRAG system
        logger.debug(f"🔄 Using base GraphRAG LLM system for {task_name}")

        # Call the base system's LLM method
        if hasattr(self, 'llm') and self.llm:
            try:
                # Try to use the base system's LLM call method
                if hasattr(self.llm, 'call'):
                    return self.llm.call(prompt, system_prompt=system_prompt, **kwargs)
                elif hasattr(self.llm, 'invoke'):
                    return self.llm.invoke(prompt, **kwargs)
                elif hasattr(self.llm, 'generate'):
                    response = self.llm.generate([prompt], **kwargs)
                    return response.generations[0][0].text if response.generations else ""
                else:
                    # Direct call if it's a callable
                    return self.llm(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Base LLM call failed for {task_name}: {e}")
                return ""

        logger.warning(f"No LLM available for {task_name}")
        return ""

    def _generate_cypher_query(self, question: str, linked_entities: Dict[str, Optional[str]]) -> Optional[str]:
        """
        FIXED: Cypher generation with multi-provider LLM support AND Cypher cleaning.
        This overrides the base method but maintains full compatibility.
        """

        # Try universal patterns first if available
        if self.universal_engine and self.enable_universal_patterns:
            try:
                logger.debug("Attempting universal pattern generation")

                adaptive_result = self.universal_engine.generate_adaptive_cypher(question, linked_entities)

                if adaptive_result.get('cypher_query'):
                    confidence = adaptive_result.get('confidence_score', 0.0)

                    if confidence >= self.pattern_confidence_threshold:
                        logger.info(f"Universal pattern successful (confidence: {confidence:.3f})")

                        # Store metadata for later use
                        self._last_universal_result = adaptive_result

                        # FIXED: Clean the universal pattern cypher too
                        clean_cypher = self.clean_cypher_query(adaptive_result['cypher_query'])
                        if clean_cypher:
                            return clean_cypher
                    else:
                        logger.debug(f"Universal pattern confidence too low: {confidence:.3f}")

            except Exception as e:
                logger.warning(f"Universal pattern generation failed: {e}")

        # FIXED: Try multi-provider LLM for Cypher generation with proper cleaning
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            try:
                logger.debug("Attempting multi-provider LLM Cypher generation")

                # Create enhanced prompt for Cypher generation
                entity_context = ""
                if linked_entities:
                    entity_context = f"Linked entities: {linked_entities}"

                cypher_prompt = f"""
                Generate a Cypher query to answer the following question:
                Question: {question}
                {entity_context}

                CRITICAL REQUIREMENTS:
                1. Return ONLY pure Cypher code
                2. NO markdown formatting (no ```cypher or ```)
                3. NO explanations or comments
                4. NO quotes around the entire query
                5. Start directly with Cypher keywords (MATCH, RETURN, etc.)

                Example correct response:
                MATCH (n:Entity) WHERE n.name = "example" RETURN n

                Provide only the Cypher query without any additional text.
                """

                system_prompt = """You are a Neo4j Cypher query expert. Generate ONLY pure Cypher code without any markdown formatting, explanations, or additional text. The query should be ready to execute directly in Neo4j."""

                raw_cypher_response = self._enhanced_llm_call(
                    task_name='cypher_generation',
                    prompt=cypher_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.1  # Low temperature for consistent formatting
                )

                if raw_cypher_response and raw_cypher_response.strip():
                    # FIXED: Clean the generated Cypher
                    clean_cypher = self.clean_cypher_query(raw_cypher_response)

                    if clean_cypher:
                        logger.info("Multi-provider LLM Cypher generation successful")
                        return clean_cypher
                    else:
                        logger.warning("Multi-provider LLM generated Cypher but cleaning failed")

            except Exception as e:
                logger.warning(f"Multi-provider LLM Cypher generation failed: {e}")

        # Fallback to original method
        logger.debug("Using base GraphRAG Cypher generation")
        base_cypher = super()._generate_cypher_query(question, linked_entities)

        # FIXED: Also clean the base system's cypher output
        if base_cypher:
            clean_base_cypher = self.clean_cypher_query(base_cypher)
            return clean_base_cypher if clean_base_cypher else base_cypher

        return base_cypher

    def _link_entities(self, question: str) -> Dict[str, Optional[str]]:
        """
        ENHANCED: Entity linking with multi-provider LLM support.
        """
        # ENHANCED: Try multi-provider LLM for entity linking
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            try:
                logger.debug("Attempting multi-provider LLM entity linking")

                entity_linking_prompt = f"""
                Identify and extract named entities from the following question:
                Question: {question}

                Please identify entities that might be relevant for a knowledge graph query.
                Return the entities in a simple format like: entity1, entity2, entity3
                """

                system_prompt = """You are an expert at identifying named entities in questions for knowledge graph queries. Focus on entities that would be nodes in a graph database."""

                entity_response = self._enhanced_llm_call(
                    task_name='entity_linking',
                    prompt=entity_linking_prompt,
                    system_prompt=system_prompt,
                    max_tokens=200,
                    temperature=0.1
                )

                if entity_response and entity_response.strip():
                    # Parse the response to extract entities
                    entities = {}
                    for entity in entity_response.strip().split(','):
                        entity = entity.strip()
                        if entity:
                            entities[entity] = None  # Neo4j ID would be resolved later

                    if entities:
                        logger.info(f"Multi-provider LLM entity linking successful: {entities}")
                        return entities

            except Exception as e:
                logger.warning(f"Multi-provider LLM entity linking failed: {e}")

        # Fallback to original method
        logger.debug("Using base GraphRAG entity linking")
        return super()._link_entities(question)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        ENHANCED: Answer generation with multi-provider LLM support.
        """
        # ENHANCED: Try multi-provider LLM for answer generation
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            try:
                logger.debug("Attempting multi-provider LLM answer generation")

                answer_prompt = f"""
                Based on the following context from a knowledge graph, answer the user's question:

                Question: {question}
                Context: {context}

                Please provide a clear, concise answer based on the context provided.
                """

                system_prompt = """You are an expert at answering questions based on knowledge graph data. Provide accurate, informative answers based on the given context."""

                answer = self._enhanced_llm_call(
                    task_name='answer_generation',
                    prompt=answer_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.3
                )

                if answer and answer.strip():
                    logger.info("Multi-provider LLM answer generation successful")
                    return answer.strip()

            except Exception as e:
                logger.warning(f"Multi-provider LLM answer generation failed: {e}")

        # Fallback to original method
        logger.debug("Using base GraphRAG answer generation")
        return super()._generate_answer(question, context)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        ENHANCED: Question answering with multi-provider LLM and universal pattern metadata.
        Maintains full compatibility with base GraphRAGQA.
        """

        logger.info(f"=== Enhanced GraphRAG Processing: {question} ===")

        # Clear previous universal result
        self._last_universal_result = None

        # Use the enhanced base method (which will call our enhanced methods)
        result = super().answer_question(question)

        # ENHANCED: Add multi-provider LLM metadata
        llm_providers_used = []
        for task_name, manager in self.llm_managers.items():
            if manager:
                try:
                    primary_provider = manager.primary_provider
                    fallback_providers = manager.fallback_providers

                    providers = [primary_provider.config.provider.value]
                    providers.extend([fp.config.provider.value for fp in fallback_providers])

                    llm_providers_used.append({
                        'task': task_name,
                        'providers': providers
                    })
                except Exception as e:
                    logger.debug(f"Could not get provider info for {task_name}: {e}")

        # Add universal enhancement metadata if available
        if hasattr(self, '_last_universal_result') and self._last_universal_result:
            universal_data = self._last_universal_result

            result.update({
                'cypher_confidence': universal_data.get('confidence_score', 0.0),
                'generation_approach': universal_data.get('approach_used', 'base_system'),
                'pattern_used': universal_data.get('pattern_used'),
                'domain_detected': universal_data.get('domain_detected', 'unknown'),
                'pattern_category': universal_data.get('pattern_category'),
                'llm_providers_used': llm_providers_used,
                'universal_enhancement': {
                    'enabled': self.enable_universal_patterns,
                    'industry_detection': True,
                    'pattern_adaptation': True,
                    'confidence_threshold': self.pattern_confidence_threshold
                },
                'multi_provider_llm': {
                    'enabled': self.enable_multi_provider_llm,
                    'system_available': NEW_LLM_SYSTEM_AVAILABLE,
                    'providers_configured': len(llm_providers_used)
                }
            })
        else:
            # Base system was used
            result.update({
                'generation_approach': 'base_system',
                'llm_providers_used': llm_providers_used,
                'universal_enhancement': {
                    'enabled': self.enable_universal_patterns,
                    'pattern_matched': False,
                    'fallback_used': True
                },
                'multi_provider_llm': {
                    'enabled': self.enable_multi_provider_llm,
                    'system_available': NEW_LLM_SYSTEM_AVAILABLE,
                    'providers_configured': len(llm_providers_used)
                }
            })

        logger.info("=== Enhanced GraphRAG Processing Complete ===")
        return result

    def get_industry_info(self):
        """Get comprehensive industry and system information"""
        try:
            # Basic info from universal engine
            industry_info = {
                'detected_industry': getattr(self.universal_engine, 'detected_industry', 'unknown'),
                'available_patterns': len(getattr(self.universal_engine, 'patterns', [])),
                'detection_confidence': getattr(self.universal_engine, 'confidence', 0.8),
                'schema_entities': 0,
                'schema_relationships': 0,
                'entity_types': [],
                'relationship_types': []
            }

            # Get actual counts from Neo4j
            if hasattr(self, 'graph_store') and self.graph_store:
                try:
                    # Count entities
                    entity_query = "MATCH (n:Entity) RETURN count(n) as count"
                    entity_result = self.graph_store.query(entity_query)
                    if entity_result:
                        industry_info['schema_entities'] = entity_result[0].get('count', 0)

                    # Count relationships
                    rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
                    rel_result = self.graph_store.query(rel_query)
                    if rel_result:
                        industry_info['schema_relationships'] = rel_result[0].get('count', 0)

                    # Get entity types (try multiple possible property names)
                    entity_types_queries = [
                        # Try common property names for entity types
                        "MATCH (n:Entity) WHERE n.type IS NOT NULL RETURN DISTINCT n.type as entity_type LIMIT 20",
                        "MATCH (n:Entity) WHERE n.entity_type IS NOT NULL RETURN DISTINCT n.entity_type as entity_type LIMIT 20",
                        "MATCH (n:Entity) WHERE n.category IS NOT NULL RETURN DISTINCT n.category as entity_type LIMIT 20",
                        "MATCH (n:Entity) WHERE n.label IS NOT NULL RETURN DISTINCT n.label as entity_type LIMIT 20",
                        # Fallback: get sample entity names to understand structure
                        "MATCH (n:Entity) WHERE n.name IS NOT NULL RETURN DISTINCT n.name as entity_type LIMIT 10"
                    ]

                    entity_types = []
                    for query in entity_types_queries:
                        try:
                            result = self.graph_store.query(query)
                            if result:
                                entity_types = [r.get('entity_type') for r in result if r.get('entity_type')]
                                if entity_types:  # If we found results, stop trying other queries
                                    break
                        except Exception as e:
                            continue  # Try next query

                    industry_info['entity_types'] = entity_types

                    # Get relationship types
                    rel_types_query = """
                    MATCH ()-[r]->() 
                    RETURN DISTINCT type(r) as rel_type 
                    LIMIT 20
                    """
                    rel_types_result = self.graph_store.query(rel_types_query)
                    if rel_types_result:
                        industry_info['relationship_types'] = [r.get('rel_type') for r in rel_types_result if
                                                               r.get('rel_type')]

                except Exception as e:
                    logger.warning(f"Could not get detailed schema info: {e}")

            return industry_info

        except Exception as e:
            logger.error(f"Error getting industry info: {e}")
            return {
                'detected_industry': 'unknown',
                'available_patterns': 0,
                'detection_confidence': 0.0,
                'schema_entities': 0,
                'schema_relationships': 0,
                'entity_types': [],
                'relationship_types': []
            }

    def get_pattern_stats(self):
        """Get pattern usage statistics"""
        if not hasattr(self, 'universal_engine') or not self.universal_engine:
            return {}

        try:
            patterns = getattr(self.universal_engine, 'patterns', [])
            return {
                'total_patterns': len(patterns),
                'pattern_names': [p.get('name', 'Unknown') for p in patterns] if patterns else [],
                'industry_specific': True if self.universal_engine.detected_industry != 'general' else False,
                'confidence_threshold': getattr(self, 'pattern_confidence_threshold', 0.7)
            }
        except Exception as e:
            logger.warning(f"Error getting pattern stats: {e}")
            return {}

    def get_system_health(self):
        """ENHANCED: Get system health metrics including multi-provider LLM status"""
        health = {
            'neo4j_connected': False,
            'vector_db_ready': False,
            'llm_available': False,
            'universal_patterns_active': False,
            'multi_provider_llm_active': False,
            'llm_providers_ready': []
        }

        try:
            # Check Neo4j
            if hasattr(self, 'graph_store') and self.graph_store:
                test_query = "RETURN 1 as test"
                result = self.graph_store.query(test_query)
                health['neo4j_connected'] = bool(result)

            # Check vector DB
            if hasattr(self, 'vector_store') and self.vector_store:
                health['vector_db_ready'] = True

            # Check LLM
            if hasattr(self, 'llm') and self.llm:
                health['llm_available'] = True

            # Check universal patterns
            if hasattr(self, 'universal_engine') and self.universal_engine:
                health['universal_patterns_active'] = True

            # ENHANCED: Check multi-provider LLM
            if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
                health['multi_provider_llm_active'] = True

                # Check each LLM manager
                for task_name, manager in self.llm_managers.items():
                    if manager:
                        try:
                            primary_provider = manager.primary_provider
                            health['llm_providers_ready'].append({
                                'task': task_name,
                                'provider': primary_provider.config.provider.value,
                                'ready': True
                            })
                        except Exception as e:
                            health['llm_providers_ready'].append({
                                'task': task_name,
                                'provider': 'unknown',
                                'ready': False,
                                'error': str(e)
                            })

        except Exception as e:
            logger.warning(f"Error checking system health: {e}")

        return health

    def get_llm_provider_info(self) -> Dict[str, Any]:
        """ENHANCED: Get detailed information about configured LLM providers"""
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {
                'multi_provider_enabled': False,
                'system_available': NEW_LLM_SYSTEM_AVAILABLE,
                'providers': []
            }

        provider_info = {
            'multi_provider_enabled': True,
            'system_available': NEW_LLM_SYSTEM_AVAILABLE,
            'providers': []
        }

        for task_name, manager in self.llm_managers.items():
            if manager:
                try:
                    primary_provider = manager.primary_provider
                    fallback_providers = manager.fallback_providers

                    task_providers = {
                        'task': task_name,
                        'primary_provider': {
                            'name': primary_provider.config.provider.value,
                            'model': primary_provider.config.model,
                            'ready': True
                        },
                        'fallback_providers': []
                    }

                    for fp in fallback_providers:
                        task_providers['fallback_providers'].append({
                            'name': fp.config.provider.value,
                            'model': fp.config.model,
                            'ready': True
                        })

                    provider_info['providers'].append(task_providers)

                except Exception as e:
                    provider_info['providers'].append({
                        'task': task_name,
                        'error': str(e),
                        'ready': False
                    })

        return provider_info

    def switch_industry(self, industry_name: str) -> bool:
        """Manually switch industry context"""
        if not self.universal_engine:
            logger.warning("Universal engine not available for industry switching")
            return False

        try:
            industry_enum = IndustryType(industry_name.lower())
            patterns = self.universal_engine.get_patterns_for_industry(industry_enum)
            self.universal_engine.adapted_patterns = patterns
            logger.info(f"Switched to {industry_name} industry context")
            return True
        except ValueError:
            logger.error(f"Unknown industry: {industry_name}")
            return False

    def get_available_industries(self) -> List[str]:
        """Get list of supported industries"""
        return [industry.value.replace('_', ' ').title() for industry in IndustryType]

    def is_enhanced(self) -> bool:
        """Check if universal enhancements are active"""
        return self.enable_universal_patterns and self.universal_engine is not None

    def is_multi_provider_enabled(self) -> bool:
        """ENHANCED: Check if multi-provider LLM is active"""
        return self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE and bool(self.llm_managers)


# Backward compatibility function - drop-in replacement
def create_enhanced_graphrag_qa(*args, **kwargs) -> EnhancedGraphRAGQA:
    """
    Factory function for creating enhanced GraphRAG QA.
    Use this as a drop-in replacement for GraphRAGQA.
    """
    return EnhancedGraphRAGQA(*args, **kwargs)


# ENHANCED: New factory function with multi-provider support
def create_multi_provider_graphrag_qa(config: Dict[str, Any], *args, **kwargs) -> EnhancedGraphRAGQA:
    """
    Factory function for creating enhanced GraphRAG QA with multi-provider LLM support.

    Args:
        config: Configuration dictionary for multi-provider LLM
        *args, **kwargs: Additional arguments for GraphRAG QA
    """
    kwargs['config'] = config
    kwargs['enable_multi_provider_llm'] = True
    return EnhancedGraphRAGQA(*args, **kwargs)


# Example integration helper
def upgrade_existing_graphrag(existing_qa_engine) -> EnhancedGraphRAGQA:
    """
    Helper to upgrade an existing GraphRAGQA instance.
    (Note: This would require re-initialization with same parameters)
    """
    logger.info("For upgrading existing instances, please re-initialize with EnhancedGraphRAGQA")
    return None


if __name__ == "__main__":
    print("Enhanced GraphRAG QA Integration Layer with Multi-Provider LLM Support and Cypher Cleaning")
    print("✅ FIXED: Cypher query generation now properly cleans markdown formatting")
    print("✅ ENHANCED: Multi-provider LLM support with fallback mechanisms")
    print("✅ ENHANCED: Universal pattern support with industry detection")
    print("Ready for integration with your existing system!")