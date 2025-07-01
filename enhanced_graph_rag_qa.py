# enhanced_graph_rag_qa.py - COMPLETE INTEGRATION LAYER

import logging
import time
from typing import Dict, List, Optional, Any

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
    Enhanced version that adds universal pattern support to your existing GraphRAGQA.
    Completely backward compatible - your existing code continues to work unchanged.
    """

    def __init__(self, *args, **kwargs):
        # Extract new enhancement parameters
        self.enable_universal_patterns = kwargs.pop('enable_universal_patterns', True)
        self.manual_industry = kwargs.pop('manual_industry', None)
        self.pattern_confidence_threshold = kwargs.pop('pattern_confidence_threshold', 0.6)

        # Initialize the base GraphRAGQA system first (unchanged)
        super().__init__(*args, **kwargs)

        # Add universal enhancements
        self.universal_engine = None
        if self.enable_universal_patterns:
            self._initialize_universal_engine()

        logger.info(f"Enhanced GraphRAG QA initialized. Universal patterns: {self.enable_universal_patterns}")

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

    def _generate_cypher_query(self, question: str, linked_entities: Dict[str, Optional[str]]) -> Optional[str]:
        """
        Enhanced Cypher generation that tries universal patterns first, then falls back to base system.
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

                        return adaptive_result['cypher_query']
                    else:
                        logger.debug(f"Universal pattern confidence too low: {confidence:.3f}")

            except Exception as e:
                logger.warning(f"Universal pattern generation failed: {e}")

        # Fallback to original method
        logger.debug("Using base GraphRAG Cypher generation")
        return super()._generate_cypher_query(question, linked_entities)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Enhanced question answering that includes universal pattern metadata.
        Maintains full compatibility with base GraphRAGQA.
        """

        logger.info(f"=== Enhanced GraphRAG Processing: {question} ===")

        # Clear previous universal result
        self._last_universal_result = None

        # Use the enhanced base method (which will call our enhanced _generate_cypher_query)
        result = super().answer_question(question)

        # Add universal enhancement metadata if available
        if hasattr(self, '_last_universal_result') and self._last_universal_result:
            universal_data = self._last_universal_result

            result.update({
                'cypher_confidence': universal_data.get('confidence_score', 0.0),
                'generation_approach': universal_data.get('approach_used', 'base_system'),
                'pattern_used': universal_data.get('pattern_used'),
                'domain_detected': universal_data.get('domain_detected', 'unknown'),
                'pattern_category': universal_data.get('pattern_category'),
                'universal_enhancement': {
                    'enabled': self.enable_universal_patterns,
                    'industry_detection': True,
                    'pattern_adaptation': True,
                    'confidence_threshold': self.pattern_confidence_threshold
                }
            })
        else:
            # Base system was used
            result.update({
                'generation_approach': 'base_system',
                'universal_enhancement': {
                    'enabled': self.enable_universal_patterns,
                    'pattern_matched': False,
                    'fallback_used': True
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

    # Also add this method if it doesn't exist
    def is_enhanced(self):
        """Check if this is an enhanced GraphRAG engine"""
        return hasattr(self, 'universal_engine') and self.universal_engine is not None

    # And add these additional methods for completeness
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
        """Get system health metrics"""
        health = {
            'neo4j_connected': False,
            'vector_db_ready': False,
            'llm_available': False,
            'universal_patterns_active': False
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

        except Exception as e:
            logger.warning(f"Error checking system health: {e}")

        return health

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


# Backward compatibility function - drop-in replacement
def create_enhanced_graphrag_qa(*args, **kwargs) -> EnhancedGraphRAGQA:
    """
    Factory function for creating enhanced GraphRAG QA.
    Use this as a drop-in replacement for GraphRAGQA.
    """
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
    print("Enhanced GraphRAG QA Integration Layer")
    print("Provides universal pattern support with full backward compatibility")
    print("Ready for integration with your existing system!")