# advanced_cypher_generation.py - ENHANCED CYPHER GENERATION SYSTEM

import logging
import re
import json
import hashlib
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict, Counter
import networkx as nx

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    SIMPLE = "simple"  # Single node/relationship
    MODERATE = "moderate"  # 2-3 hops
    COMPLEX = "complex"  # 4+ hops, aggregations
    RECURSIVE = "recursive"  # Variable length paths


class QueryCategory(Enum):
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_EXPLORATION = "relationship_exploration"
    PATH_FINDING = "path_finding"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    MAINTENANCE_WORKFLOW = "maintenance_workflow"
    ASSET_HIERARCHY = "asset_hierarchy"
    COMPLIANCE = "compliance"


@dataclass
class QueryPattern:
    """Represents a reusable Cypher query pattern"""
    category: QueryCategory
    complexity: QueryComplexity
    template: str
    description: str
    example_nl: str
    example_cypher: str
    parameters: List[str]
    confidence_score: float = 1.0


@dataclass
class SchemaElement:
    """Enhanced schema representation"""
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: Dict[str, List[str]]
    relationship_properties: Dict[str, List[str]]
    common_patterns: List[str]
    domain_specific_terms: List[str]


class EnhancedCypherGenerator:
    """
    Advanced Cypher generation with pattern-based approach,
    confidence scoring, and domain-specific optimizations.
    """

    def __init__(self, neo4j_driver, llm_config: Dict[str, Any]):
        self.driver = neo4j_driver
        self.llm_config = llm_config

        # Core components
        self.schema_analyzer = SchemaAnalyzer(neo4j_driver)
        self.pattern_library = PatternLibrary()
        self.confidence_scorer = ConfidenceScorer()
        self.query_validator = QueryValidator(neo4j_driver)
        self.few_shot_manager = EnhancedFewShotManager()

        # Performance tracking
        self.query_stats = QueryStatistics()

        # Initialize pattern library with domain-specific patterns
        self._initialize_asset_management_patterns()

        logger.info("Enhanced Cypher Generator initialized")

    def _initialize_asset_management_patterns(self):
        """Initialize patterns specific to Physical Asset Management"""

        # Asset Lookup Patterns
        self.pattern_library.add_pattern(QueryPattern(
            category=QueryCategory.ENTITY_LOOKUP,
            complexity=QueryComplexity.SIMPLE,
            template="MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('{entity}') RETURN e.name, labels(e) LIMIT 10",
            description="Find assets by name or partial match",
            example_nl="Find equipment named pump",
            example_cypher="MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'pump' RETURN e.name, labels(e)",
            parameters=["entity"]
        ))

        # Maintenance Workflow Patterns
        self.pattern_library.add_pattern(QueryPattern(
            category=QueryCategory.MAINTENANCE_WORKFLOW,
            complexity=QueryComplexity.MODERATE,
            template="""MATCH (asset:Entity)-[r1]->(wo:Entity)
                       WHERE toLower(asset.name) CONTAINS toLower('{asset}') 
                       AND 'WorkOrder' IN labels(wo)
                       OPTIONAL MATCH (wo)-[r2]->(tech:Entity)
                       WHERE 'Technician' IN labels(tech)
                       RETURN asset.name, wo.name, tech.name, type(r1), type(r2)""",
            description="Find work orders and assigned technicians for asset",
            example_nl="Who is assigned to maintain the main compressor?",
            example_cypher="MATCH (asset:Entity)-[r1]->(wo:Entity)-[r2]->(tech:Entity) WHERE asset.name CONTAINS 'compressor' AND 'WorkOrder' IN labels(wo) RETURN wo.name, tech.name",
            parameters=["asset"]
        ))

        # Asset Hierarchy Patterns
        self.pattern_library.add_pattern(QueryPattern(
            category=QueryCategory.ASSET_HIERARCHY,
            complexity=QueryComplexity.COMPLEX,
            template="""MATCH path = (parent:Entity)-[:contains*1..3]->(child:Entity)
                       WHERE toLower(parent.name) CONTAINS toLower('{parent}')
                       RETURN parent.name, child.name, length(path) as depth
                       ORDER BY depth, child.name""",
            description="Find asset hierarchy and containment relationships",
            example_nl="What equipment is contained in Building A?",
            example_cypher="MATCH (building:Entity)-[:contains*1..3]->(equipment:Entity) WHERE building.name = 'Building A' RETURN equipment.name",
            parameters=["parent"]
        ))

        # Compliance Patterns
        self.pattern_library.add_pattern(QueryPattern(
            category=QueryCategory.COMPLIANCE,
            complexity=QueryComplexity.MODERATE,
            template="""MATCH (asset:Entity)-[r]->(inspection:Entity)
                       WHERE 'Inspection' IN labels(inspection)
                       AND toLower(asset.name) CONTAINS toLower('{asset}')
                       RETURN asset.name, inspection.name, r.date, r.status
                       ORDER BY r.date DESC""",
            description="Find compliance inspections for assets",
            example_nl="When was the last inspection of the safety valve?",
            example_cypher="MATCH (valve:Entity)-[r]->(insp:Entity) WHERE valve.name CONTAINS 'safety valve' AND 'Inspection' IN labels(insp) RETURN insp.name, r.date ORDER BY r.date DESC LIMIT 1",
            parameters=["asset"]
        ))

    def generate_cypher(self, question: str, linked_entities: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Enhanced Cypher generation with multi-stage approach
        """
        start_time = time.time()

        # Step 1: Analyze question complexity and category
        analysis = self._analyze_question(question)

        # Step 2: Pattern matching
        matched_patterns = self._match_patterns(question, analysis)

        # Step 3: Schema-aware generation
        schema_context = self.schema_analyzer.get_relevant_schema(question, linked_entities)

        # Step 4: Multi-approach generation
        cypher_candidates = []

        # Approach A: Pattern-based generation
        if matched_patterns:
            pattern_query = self._generate_from_patterns(question, matched_patterns, linked_entities)
            if pattern_query:
                cypher_candidates.append({
                    'query': pattern_query,
                    'approach': 'pattern_based',
                    'confidence': self._calculate_pattern_confidence(question, matched_patterns)
                })

        # Approach B: LLM with enhanced prompts
        llm_query = self._generate_with_enhanced_llm(question, schema_context, linked_entities, matched_patterns)
        if llm_query:
            cypher_candidates.append({
                'query': llm_query,
                'approach': 'llm_enhanced',
                'confidence': self._calculate_llm_confidence(question, llm_query, schema_context)
            })

        # Approach C: Hybrid approach
        if len(cypher_candidates) > 1:
            hybrid_query = self._create_hybrid_query(cypher_candidates, question, schema_context)
            if hybrid_query:
                cypher_candidates.append({
                    'query': hybrid_query,
                    'approach': 'hybrid',
                    'confidence': self._calculate_hybrid_confidence(cypher_candidates)
                })

        # Step 5: Validation and selection
        best_candidate = self._select_best_candidate(cypher_candidates, question)

        # Step 6: Post-processing and optimization
        if best_candidate:
            optimized_query = self._optimize_query(best_candidate['query'], schema_context)
            best_candidate['query'] = optimized_query

            # Validate before returning
            validation_result = self.query_validator.validate_query(optimized_query)
            best_candidate['validation'] = validation_result

        generation_time = time.time() - start_time

        result = {
            'cypher_query': best_candidate['query'] if best_candidate else None,
            'approach_used': best_candidate['approach'] if best_candidate else 'failed',
            'confidence_score': best_candidate['confidence'] if best_candidate else 0.0,
            'question_analysis': analysis,
            'matched_patterns': [p.description for p in matched_patterns],
            'schema_elements_used': schema_context.get('elements_used', []),
            'generation_time': generation_time,
            'validation_result': best_candidate.get('validation') if best_candidate else None,
            'all_candidates': len(cypher_candidates)
        }

        # Store for learning
        self._store_generation_result(question, result)

        return result

    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine complexity and category"""

        analysis = {
            'complexity': QueryComplexity.SIMPLE,
            'category': QueryCategory.ENTITY_LOOKUP,
            'entities_mentioned': [],
            'temporal_indicators': [],
            'aggregation_needed': False,
            'relationship_focus': False
        }

        # Entity extraction
        entity_patterns = [
            r'\b([A-Z][a-z]+ [A-Z0-9]+)\b',  # Equipment names
            r'\b([A-Z]{2,}\d*)\b',  # Codes/IDs
            r'\b(building|room|floor|site) [A-Za-z0-9]+\b',  # Locations
        ]

        for pattern in entity_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            analysis['entities_mentioned'].extend(matches)

        # Complexity indicators
        complexity_indicators = {
            QueryComplexity.SIMPLE: ['what is', 'find', 'show me', 'list'],
            QueryComplexity.MODERATE: ['how many', 'who is assigned', 'what equipment', 'which technician'],
            QueryComplexity.COMPLEX: ['all equipment in', 'maintenance history', 'compliance status', 'hierarchy'],
            QueryComplexity.RECURSIVE: ['all connected', 'full path', 'entire network', 'all dependencies']
        }

        for complexity, indicators in complexity_indicators.items():
            if any(indicator in question.lower() for indicator in indicators):
                analysis['complexity'] = complexity
                break

        # Category classification
        category_keywords = {
            QueryCategory.ENTITY_LOOKUP: ['find', 'what is', 'show', 'list'],
            QueryCategory.MAINTENANCE_WORKFLOW: ['work order', 'maintenance', 'repair', 'service', 'technician'],
            QueryCategory.ASSET_HIERARCHY: ['contains', 'inside', 'part of', 'hierarchy', 'structure'],
            QueryCategory.COMPLIANCE: ['inspection', 'compliance', 'audit', 'certification', 'regulation'],
            QueryCategory.TEMPORAL: ['when', 'last', 'next', 'schedule', 'history', 'recent'],
            QueryCategory.AGGREGATION: ['how many', 'total', 'count', 'average', 'sum']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                analysis['category'] = category
                break

        # Temporal indicators
        temporal_patterns = [
            r'\b(last|next|recent|upcoming)\b',
            r'\b(\d{4}|\d{1,2}/\d{1,2})\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]

        for pattern in temporal_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            analysis['temporal_indicators'].extend(matches)

        # Aggregation detection
        aggregation_words = ['how many', 'count', 'total', 'sum', 'average', 'maximum', 'minimum']
        analysis['aggregation_needed'] = any(word in question.lower() for word in aggregation_words)

        # Relationship focus
        relationship_words = ['connected to', 'assigned to', 'maintained by', 'contains', 'part of', 'related to']
        analysis['relationship_focus'] = any(word in question.lower() for word in relationship_words)

        return analysis

    def _match_patterns(self, question: str, analysis: Dict[str, Any]) -> List[QueryPattern]:
        """Match question against pattern library"""

        matched_patterns = []

        # Get patterns by category and complexity
        category_patterns = self.pattern_library.get_patterns_by_category(analysis['category'])
        complexity_patterns = self.pattern_library.get_patterns_by_complexity(analysis['complexity'])

        # Score pattern matches
        all_patterns = category_patterns + complexity_patterns

        for pattern in all_patterns:
            score = self._calculate_pattern_match_score(question, pattern)
            if score > 0.5:  # Threshold for pattern matching
                pattern.confidence_score = score
                matched_patterns.append(pattern)

        # Sort by confidence
        matched_patterns.sort(key=lambda p: p.confidence_score, reverse=True)

        return matched_patterns[:3]  # Top 3 patterns

    def _calculate_pattern_match_score(self, question: str, pattern: QueryPattern) -> float:
        """Calculate how well a pattern matches the question"""

        score = 0.0
        question_lower = question.lower()

        # Example natural language similarity
        example_words = set(pattern.example_nl.lower().split())
        question_words = set(question_lower.split())

        if example_words and question_words:
            overlap = len(example_words.intersection(question_words))
            score += (overlap / len(example_words.union(question_words))) * 0.4

        # Description similarity
        desc_words = set(pattern.description.lower().split())
        if desc_words and question_words:
            desc_overlap = len(desc_words.intersection(question_words))
            score += (desc_overlap / len(desc_words.union(question_words))) * 0.3

        # Parameter presence
        param_present = 0
        for param in pattern.parameters:
            if param.lower() in question_lower:
                param_present += 1

        if pattern.parameters:
            score += (param_present / len(pattern.parameters)) * 0.3

        return min(score, 1.0)

    def _generate_from_patterns(self, question: str, patterns: List[QueryPattern], entities: Dict[str, str]) -> str:
        """Generate Cypher using matched patterns"""

        if not patterns:
            return None

        best_pattern = patterns[0]  # Highest confidence

        # Extract parameter values from question and entities
        param_values = {}

        for param in best_pattern.parameters:
            if param == 'entity' or param == 'asset':
                # Try to find entity mention in question or linked entities
                if entities:
                    # Use first linked entity as primary
                    param_values[param] = list(entities.values())[0]
                else:
                    # Extract from question
                    entity_match = self._extract_entity_from_question(question)
                    if entity_match:
                        param_values[param] = entity_match
            elif param == 'parent':
                # Look for location/container entities
                location_words = ['building', 'facility', 'site', 'room', 'floor']
                for word in location_words:
                    pattern_match = re.search(rf'{word}\s+([A-Za-z0-9]+)', question, re.IGNORECASE)
                    if pattern_match:
                        param_values[param] = f"{word} {pattern_match.group(1)}"
                        break

        # Fill template with parameters
        try:
            filled_query = best_pattern.template.format(**param_values)
            return filled_query
        except KeyError as e:
            logger.warning(f"Missing parameter for pattern: {e}")
            return None

    def _generate_with_enhanced_llm(self, question: str, schema_context: Dict, entities: Dict,
                                    patterns: List[QueryPattern]) -> str:
        """Generate Cypher using LLM with enhanced prompts"""

        # Build enhanced system prompt
        system_prompt = self._build_enhanced_system_prompt(schema_context, patterns)

        # Build user prompt with context
        user_prompt = self._build_enhanced_user_prompt(question, schema_context, entities, patterns)

        try:
            from src.knowledge_graph.llm import call_llm

            response = call_llm(
                model=self.llm_config['model'],
                api_key=self.llm_config['api_key'],
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=800,
                base_url=self.llm_config.get('base_url')
            )

            return self._extract_cypher_from_response(response)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _build_enhanced_system_prompt(self, schema_context: Dict, patterns: List[QueryPattern]) -> str:
        """Build comprehensive system prompt with schema and patterns"""

        prompt_parts = [
            "You are an expert Neo4j Cypher query generator specialized in Physical Asset Management systems.",
            "",
            "CRITICAL RULES:",
            "1. Generate syntactically correct Cypher queries only",
            "2. Use exact node labels and relationship types from the schema",
            "3. Apply appropriate WHERE clauses for entity matching",
            "4. Use toLower() for case-insensitive string matching",
            "5. Always RETURN relevant data (names, properties, relationships)",
            "6. Use LIMIT for potentially large result sets",
            "",
            "SCHEMA INFORMATION:",
        ]

        # Add schema details
        if schema_context.get('node_labels'):
            prompt_parts.append(f"Node Labels: {', '.join(schema_context['node_labels'])}")

        if schema_context.get('relationship_types'):
            prompt_parts.append(f"Relationship Types: {', '.join(schema_context['relationship_types'])}")

        if schema_context.get('common_patterns'):
            prompt_parts.append("Common Patterns:")
            for pattern in schema_context['common_patterns'][:5]:
                prompt_parts.append(f"  - {pattern}")

        # Add relevant query patterns
        if patterns:
            prompt_parts.extend([
                "",
                "RELEVANT QUERY PATTERNS:",
            ])
            for pattern in patterns[:2]:  # Top 2 patterns
                prompt_parts.extend([
                    f"Pattern: {pattern.description}",
                    f"Example: {pattern.example_nl} → {pattern.example_cypher}",
                    ""
                ])

        prompt_parts.extend([
            "",
            "OUTPUT FORMAT:",
            "Return ONLY the Cypher query, no explanations or markdown formatting.",
            "Ensure the query is complete and executable."
        ])

        return "\n".join(prompt_parts)

    def _build_enhanced_user_prompt(self, question: str, schema_context: Dict, entities: Dict,
                                    patterns: List[QueryPattern]) -> str:
        """Build comprehensive user prompt with all context"""

        prompt_parts = [f"Question: {question}", ""]

        # Add linked entities context
        if entities:
            prompt_parts.append("Linked Entities:")
            for mention, canonical in entities.items():
                prompt_parts.append(f"  '{mention}' → '{canonical}'")
            prompt_parts.append("")

        # Add schema context
        if schema_context.get('relevant_elements'):
            prompt_parts.append("Most Relevant Schema Elements:")
            for element in schema_context['relevant_elements'][:10]:
                prompt_parts.append(f"  - {element}")
            prompt_parts.append("")

        # Add domain-specific hints
        domain_hints = self._get_domain_hints(question)
        if domain_hints:
            prompt_parts.append("Domain Context:")
            for hint in domain_hints:
                prompt_parts.append(f"  - {hint}")
            prompt_parts.append("")

        prompt_parts.append("Generate the Cypher query:")

        return "\n".join(prompt_parts)

    def _get_domain_hints(self, question: str) -> List[str]:
        """Get domain-specific hints for asset management"""

        hints = []
        question_lower = question.lower()

        if 'maintenance' in question_lower or 'repair' in question_lower:
            hints.append("Look for WorkOrder entities and their relationships to Equipment and Technician entities")

        if 'technician' in question_lower or 'assigned' in question_lower:
            hints.append("Use Person or Technician node labels for personnel")

        if 'building' in question_lower or 'location' in question_lower:
            hints.append("Use location hierarchy: Site → Building → Floor → Room")

        if 'equipment' in question_lower or 'asset' in question_lower:
            hints.append("Equipment entities often have manufacturer, model, and serial_number properties")

        if 'when' in question_lower or 'last' in question_lower:
            hints.append("Look for date properties and temporal relationships")

        return hints

    def _select_best_candidate(self, candidates: List[Dict], question: str) -> Dict:
        """Select the best Cypher candidate based on multiple criteria"""

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate
        for candidate in candidates:
            score = candidate['confidence']

            # Bonus for pattern-based approaches (more reliable)
            if candidate['approach'] == 'pattern_based':
                score += 0.1

            # Bonus for hybrid approaches (best of both worlds)
            elif candidate['approach'] == 'hybrid':
                score += 0.05

            # Penalty for very complex queries (potential overfitting)
            query_complexity = self._assess_query_complexity(candidate['query'])
            if query_complexity > 0.8:
                score -= 0.1

            candidate['final_score'] = score

        # Return highest scoring candidate
        return max(candidates, key=lambda c: c['final_score'])

    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a Cypher query (0-1 scale)"""

        complexity_score = 0.0

        # Count various complexity indicators
        indicators = {
            'MATCH': 0.1,
            'OPTIONAL MATCH': 0.15,
            'WITH': 0.2,
            'UNION': 0.3,
            'subquery': 0.25,
            'collect(': 0.15,
            'count(': 0.1,
            'GROUP BY': 0.2,
            '*1..': 0.25,  # Variable length paths
            'ORDER BY': 0.05,
            'LIMIT': -0.05  # Actually reduces complexity by limiting results
        }

        query_upper = query.upper()

        for indicator, weight in indicators.items():
            count = query_upper.count(indicator.upper())
            complexity_score += count * weight

        return min(complexity_score, 1.0)

    def _store_generation_result(self, question: str, result: Dict):
        """Store generation result for learning and analytics"""

        self.query_stats.record_query(
            question=question,
            approach=result['approach_used'],
            confidence=result['confidence_score'],
            generation_time=result['generation_time'],
            success=result['cypher_query'] is not None
        )

        # Store in few-shot manager if successful
        if result['cypher_query'] and result['confidence_score'] > 0.7:
            self.few_shot_manager.add_example(
                question=question,
                cypher=result['cypher_query'],
                confidence=result['confidence_score'],
                approach=result['approach_used']
            )


class SchemaAnalyzer:
    """Enhanced schema analysis for better context"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self._schema_cache = {}
        self._last_schema_update = 0

    def get_relevant_schema(self, question: str, entities: Dict[str, str] = None) -> Dict[str, Any]:
        """Get schema elements relevant to the question"""

        # Get full schema (cached)
        full_schema = self._get_cached_schema()

        # Filter to relevant elements
        relevant_elements = []
        question_lower = question.lower()

        # Add entity-specific schema elements
        if entities:
            for canonical_name in entities.values():
                relevant_elements.extend(self._get_entity_schema_elements(canonical_name, full_schema))

        # Add keyword-based relevant elements
        for keyword in question_lower.split():
            if len(keyword) > 3:  # Skip short words
                relevant_elements.extend(self._find_schema_elements_by_keyword(keyword, full_schema))

        # Deduplicate and prioritize
        relevant_elements = list(set(relevant_elements))

        return {
            'node_labels': full_schema.get('node_labels', []),
            'relationship_types': full_schema.get('relationship_types', []),
            'relevant_elements': relevant_elements[:15],  # Top 15 most relevant
            'common_patterns': full_schema.get('common_patterns', []),
            'elements_used': relevant_elements
        }

    def _get_cached_schema(self) -> Dict[str, Any]:
        """Get schema with caching"""

        current_time = time.time()

        # Refresh cache every 5 minutes
        if current_time - self._last_schema_update > 300:
            self._schema_cache = self._extract_full_schema()
            self._last_schema_update = current_time

        return self._schema_cache

    def _extract_full_schema(self) -> Dict[str, Any]:
        """Extract comprehensive schema information"""

        schema = {
            'node_labels': [],
            'relationship_types': [],
            'node_properties': {},
            'relationship_properties': {},
            'common_patterns': []
        }

        try:
            with self.driver.session() as session:
                # Get node labels
                result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                record = result.single()
                if record:
                    schema['node_labels'] = record['labels']

                # Get relationship types
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                record = result.single()
                if record:
                    schema['relationship_types'] = record['types']

                # Get node properties for each label
                for label in schema['node_labels'][:10]:  # Limit to prevent timeout
                    try:
                        result = session.run(
                            f"MATCH (n:{label}) RETURN keys(n) as props LIMIT 100"
                        )
                        all_props = set()
                        for record in result:
                            if record['props']:
                                all_props.update(record['props'])
                        schema['node_properties'][label] = list(all_props)
                    except Exception as e:
                        logger.warning(f"Could not get properties for label {label}: {e}")

                # Get common patterns
                schema['common_patterns'] = self._extract_common_patterns(session)

        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")

        return schema

    def _extract_common_patterns(self, session) -> List[str]:
        """Extract common relationship patterns from the graph"""

        patterns = []

        try:
            # Find most common relationship patterns
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN labels(a)[0] as source_label, type(r) as rel_type, labels(b)[0] as target_label, count(*) as frequency
                ORDER BY frequency DESC
                LIMIT 20
            """)

            for record in result:
                if record['source_label'] and record['target_label'] and record['rel_type']:
                    pattern = f"{record['source_label']}-[:{record['rel_type']}]->{record['target_label']}"
                    patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Could not extract common patterns: {e}")

        return patterns


class PatternLibrary:
    """Library of reusable query patterns"""

    def __init__(self):
        self.patterns = []
        self._pattern_index = defaultdict(list)

    def add_pattern(self, pattern: QueryPattern):
        """Add a pattern to the library"""
        self.patterns.append(pattern)

        # Index by category and complexity
        self._pattern_index[pattern.category].append(pattern)
        self._pattern_index[pattern.complexity].append(pattern)

    def get_patterns_by_category(self, category: QueryCategory) -> List[QueryPattern]:
        """Get patterns by category"""
        return self._pattern_index[category].copy()

    def get_patterns_by_complexity(self, complexity: QueryComplexity) -> List[QueryPattern]:
        """Get patterns by complexity"""
        return self._pattern_index[complexity].copy()

    def search_patterns(self, keywords: List[str]) -> List[QueryPattern]:
        """Search patterns by keywords"""
        matching_patterns = []

        for pattern in self.patterns:
            # Check description and example for keywords
            text_to_search = f"{pattern.description} {pattern.example_nl}".lower()

            for keyword in keywords:
                if keyword.lower() in text_to_search:
                    matching_patterns.append(pattern)
                    break

        return matching_patterns


class ConfidenceScorer:
    """Advanced confidence scoring for Cypher queries"""

    def __init__(self):
        self.scoring_weights = {
            'syntax_validity': 0.3,
            'schema_compliance': 0.25,
            'entity_matching': 0.2,
            'pattern_similarity': 0.15,
            'complexity_appropriateness': 0.1
        }

    def calculate_confidence(self, query: str, question: str, schema_context: Dict,
                             approach: str, validation_result: Dict = None) -> float:
        """Calculate comprehensive confidence score"""

        if not query:
            return 0.0

        scores = {}

        # Syntax validity
        scores['syntax_validity'] = self._score_syntax_validity(query, validation_result)

        # Schema compliance
        scores['schema_compliance'] = self._score_schema_compliance(query, schema_context)

        # Entity matching
        scores['entity_matching'] = self._score_entity_matching(query, question)

        # Pattern similarity
        scores['pattern_similarity'] = self._score_pattern_similarity(query, question)

        # Complexity appropriateness
        scores['complexity_appropriateness'] = self._score_complexity_appropriateness(query, question)

        # Calculate weighted average
        total_score = sum(scores[factor] * self.scoring_weights[factor]
                          for factor in scores)

        # Approach-specific adjustments
        if approach == 'pattern_based':
            total_score += 0.05  # Slight bonus for pattern-based
        elif approach == 'hybrid':
            total_score += 0.03  # Bonus for hybrid approach

        return min(total_score, 1.0)

    def _score_syntax_validity(self, query: str, validation_result: Dict = None) -> float:
        """Score syntax validity of the query"""

        if validation_result:
            return 1.0 if validation_result.get('is_valid', False) else 0.2

        # Basic syntax checks
        query_upper = query.upper().strip()

        # Must start with valid Cypher keyword
        valid_starts = ['MATCH', 'MERGE', 'CREATE', 'CALL', 'OPTIONAL MATCH']
        if not any(query_upper.startswith(start) for start in valid_starts):
            return 0.1

        # Must have RETURN clause (for most queries)
        if 'RETURN' not in query_upper and 'DELETE' not in query_upper:
            return 0.3

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return 0.2

        # Check for balanced brackets
        if query.count('[') != query.count(']'):
            return 0.2

        return 0.8  # Passed basic checks

    def _score_schema_compliance(self, query: str, schema_context: Dict) -> float:
        """Score how well query uses actual schema elements"""

        if not schema_context:
            return 0.5  # Neutral score if no schema available

        score = 0.0
        total_checks = 0

        # Check node labels
        node_labels = schema_context.get('node_labels', [])
        if node_labels:
            query_labels = re.findall(r':(\w+)', query)
            valid_labels = sum(1 for label in query_labels if label in node_labels)
            if query_labels:
                score += (valid_labels / len(query_labels)) * 0.5
                total_checks += 0.5

        # Check relationship types
        rel_types = schema_context.get('relationship_types', [])
        if rel_types:
            query_rels = re.findall(r'\[:(\w+)\]', query)
            valid_rels = sum(1 for rel in query_rels if rel in rel_types)
            if query_rels:
                score += (valid_rels / len(query_rels)) * 0.3
                total_checks += 0.3
            elif not query_rels:  # No specific relationships used (generic pattern)
                score += 0.15
                total_checks += 0.3

        # Check for common patterns usage
        common_patterns = schema_context.get('common_patterns', [])
        if common_patterns:
            pattern_matches = 0
            for pattern in common_patterns[:5]:  # Check top 5 patterns
                if any(part in query for part in pattern.split('-')):
                    pattern_matches += 1

            score += min(pattern_matches / 5, 0.2)
            total_checks += 0.2

        return score / total_checks if total_checks > 0 else 0.5

    def _score_entity_matching(self, query: str, question: str) -> float:
        """Score how well query captures entities from question"""

        # Extract potential entities from question
        question_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[A-Z0-9]+\b', question)

        if not question_entities:
            return 0.7  # No entities to match

        # Check if entities appear in query (in WHERE clauses, etc.)
        entities_in_query = 0

        for entity in question_entities:
            if entity.lower() in query.lower():
                entities_in_query += 1

        return entities_in_query / len(question_entities)

    def _score_pattern_similarity(self, query: str, question: str) -> float:
        """Score based on common Cypher patterns for question types"""

        question_lower = question.lower()
        query_upper = query.upper()

        score = 0.0

        # Question type patterns
        if any(word in question_lower for word in ['what', 'which', 'show', 'list']):
            if 'RETURN' in query_upper:
                score += 0.3

        if any(word in question_lower for word in ['how many', 'count']):
            if 'COUNT(' in query_upper or 'count(' in query:
                score += 0.4

        if any(word in question_lower for word in ['who', 'assigned', 'responsible']):
            if 'Person' in query or 'Technician' in query:
                score += 0.3

        if any(word in question_lower for word in ['when', 'last', 'recent']):
            if 'ORDER BY' in query_upper and 'date' in query.lower():
                score += 0.4

        if any(word in question_lower for word in ['all', 'every', 'entire']):
            if 'LIMIT' not in query_upper:  # No limit for "all" queries
                score += 0.2

        return min(score, 1.0)

    def _score_complexity_appropriateness(self, query: str, question: str) -> float:
        """Score whether query complexity matches question complexity"""

        # Assess question complexity
        question_complexity = self._assess_question_complexity(question)

        # Assess query complexity
        query_complexity = self._assess_cypher_complexity(query)

        # Score based on how well they match
        complexity_diff = abs(question_complexity - query_complexity)

        if complexity_diff <= 0.2:
            return 1.0  # Very well matched
        elif complexity_diff <= 0.4:
            return 0.7  # Reasonably matched
        elif complexity_diff <= 0.6:
            return 0.4  # Somewhat matched
        else:
            return 0.1  # Poorly matched

    def _assess_question_complexity(self, question: str) -> float:
        """Assess complexity of the question (0-1 scale)"""

        question_lower = question.lower()
        complexity = 0.0

        # Simple indicators
        if any(word in question_lower for word in ['what is', 'show', 'find']):
            complexity += 0.1

        # Moderate indicators
        if any(word in question_lower for word in ['how many', 'who is', 'when']):
            complexity += 0.3

        # Complex indicators
        if any(word in question_lower for word in ['all', 'entire', 'hierarchy', 'network']):
            complexity += 0.5

        # Multiple entity references
        entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', question)
        if len(entities) > 2:
            complexity += 0.2

        # Temporal complexity
        if any(word in question_lower for word in ['history', 'over time', 'trend']):
            complexity += 0.3

        return min(complexity, 1.0)

    def _assess_cypher_complexity(self, query: str) -> float:
        """Assess complexity of the Cypher query (0-1 scale)"""

        complexity = 0.0
        query_upper = query.upper()

        # Basic query elements
        complexity += query_upper.count('MATCH') * 0.1
        complexity += query_upper.count('OPTIONAL MATCH') * 0.15
        complexity += query_upper.count('WITH') * 0.2
        complexity += query_upper.count('UNION') * 0.3

        # Complex operations
        if 'COUNT(' in query_upper:
            complexity += 0.2
        if 'GROUP BY' in query_upper:
            complexity += 0.25
        if 'ORDER BY' in query_upper:
            complexity += 0.1

        # Variable length paths
        if '*' in query and '..' in query:
            complexity += 0.3

        # Subqueries
        if 'CALL' in query_upper and '{' in query:
            complexity += 0.4

        return min(complexity, 1.0)


class QueryValidator:
    """Validates Cypher queries before execution"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def validate_query(self, query: str, dry_run: bool = True) -> Dict[str, Any]:
        """Validate Cypher query syntax and structure"""

        result = {
            'is_valid': False,
            'syntax_errors': [],
            'warnings': [],
            'estimated_cost': 0,
            'safety_score': 1.0
        }

        if not query or not query.strip():
            result['syntax_errors'].append("Empty query")
            return result

        # Basic syntax validation
        syntax_issues = self._check_basic_syntax(query)
        result['syntax_errors'].extend(syntax_issues)

        # Safety checks
        safety_issues = self._check_query_safety(query)
        result['warnings'].extend(safety_issues)
        result['safety_score'] = max(0.0, 1.0 - len(safety_issues) * 0.2)

        # Try to explain query (dry run)
        if dry_run and not result['syntax_errors']:
            try:
                with self.driver.session() as session:
                    explain_result = session.run(f"EXPLAIN {query}")
                    result['is_valid'] = True
                    # Could extract cost estimation from explain plan

            except Exception as e:
                result['syntax_errors'].append(f"Query explanation failed: {str(e)}")

        return result

    def _check_basic_syntax(self, query: str) -> List[str]:
        """Check basic Cypher syntax"""

        errors = []
        query_stripped = query.strip()
        query_upper = query_stripped.upper()

        # Must start with valid keyword
        valid_starts = ['MATCH', 'MERGE', 'CREATE', 'CALL', 'OPTIONAL MATCH', 'WITH']
        if not any(query_upper.startswith(start) for start in valid_starts):
            errors.append("Query must start with MATCH, MERGE, CREATE, CALL, or OPTIONAL MATCH")

        # Parentheses balance
        if query.count('(') != query.count(')'):
            errors.append("Unbalanced parentheses")

        # Bracket balance
        if query.count('[') != query.count(']'):
            errors.append("Unbalanced brackets")

        # Brace balance
        if query.count('{') != query.count('}'):
            errors.append("Unbalanced braces")

        # RETURN clause check (for most queries)
        if ('RETURN' not in query_upper and
                'DELETE' not in query_upper and
                'SET' not in query_upper and
                'REMOVE' not in query_upper):
            errors.append("Query should have a RETURN clause")

        return errors

    def _check_query_safety(self, query: str) -> List[str]:
        """Check for potentially unsafe query patterns"""

        warnings = []
        query_upper = query.upper()

        # Check for missing LIMIT on potentially large results
        if ('MATCH' in query_upper and
                'RETURN' in query_upper and
                'LIMIT' not in query_upper and
                'COUNT(' not in query_upper):
            warnings.append("Consider adding LIMIT to prevent large result sets")

        # Check for expensive operations
        if '*' in query and '..' in query:
            warnings.append("Variable length path queries can be expensive")

        if query_upper.count('MATCH') > 3:
            warnings.append("Multiple MATCH clauses may impact performance")

        # Check for Cartesian products
        match_count = query_upper.count('MATCH')
        if match_count > 1 and 'WHERE' not in query_upper:
            warnings.append("Multiple MATCH without WHERE may create Cartesian product")

        return warnings


class EnhancedFewShotManager:
    """Enhanced few-shot example management with quality scoring"""

    def __init__(self, db_path: str = "few_shot_examples.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for few-shot examples"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS few_shot_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    cypher_query TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    approach TEXT NOT NULL,
                    success_rate REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    question_embedding BLOB,
                    category TEXT,
                    complexity TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_confidence ON few_shot_examples(confidence_score)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON few_shot_examples(category)
            """)

    def add_example(self, question: str, cypher: str, confidence: float,
                    approach: str, category: str = None, complexity: str = None):
        """Add a new few-shot example"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO few_shot_examples 
                (question, cypher_query, confidence_score, approach, category, complexity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (question, cypher, confidence, approach, category, complexity))

    def get_best_examples(self, question: str, category: str = None, limit: int = 3) -> List[Dict]:
        """Get best few-shot examples for a question"""

        query = """
            SELECT question, cypher_query, confidence_score, approach, success_rate
            FROM few_shot_examples
            WHERE confidence_score > 0.6
        """

        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        query += """
            ORDER BY 
                confidence_score * success_rate DESC,
                usage_count DESC
            LIMIT ?
        """
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            examples = []
            for row in cursor.fetchall():
                examples.append({
                    'question': row[0],
                    'cypher': row[1],
                    'confidence': row[2],
                    'approach': row[3],
                    'success_rate': row[4]
                })

        return examples

    def update_example_success(self, question: str, cypher: str, success: bool):
        """Update success rate of an example"""

        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute("""
                SELECT usage_count, success_rate FROM few_shot_examples
                WHERE question = ? AND cypher_query = ?
            """, (question, cypher))

            row = cursor.fetchone()
            if row:
                usage_count, current_success_rate = row
                new_usage_count = usage_count + 1

                # Update success rate using running average
                if success:
                    new_success_rate = ((current_success_rate * usage_count) + 1) / new_usage_count
                else:
                    new_success_rate = (current_success_rate * usage_count) / new_usage_count

                conn.execute("""
                    UPDATE few_shot_examples
                    SET usage_count = ?, success_rate = ?
                    WHERE question = ? AND cypher_query = ?
                """, (new_usage_count, new_success_rate, question, cypher))


class QueryStatistics:
    """Track query generation statistics for analytics"""

    def __init__(self):
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'approach_success': defaultdict(int),
            'approach_total': defaultdict(int),
            'avg_generation_time': 0.0,
            'confidence_distribution': defaultdict(int)
        }

    def record_query(self, question: str, approach: str, confidence: float,
                     generation_time: float, success: bool):
        """Record statistics for a query"""

        self.stats['total_queries'] += 1
        self.stats['approach_total'][approach] += 1

        if success:
            self.stats['successful_queries'] += 1
            self.stats['approach_success'][approach] += 1

        # Update average generation time
        total_time = self.stats['avg_generation_time'] * (self.stats['total_queries'] - 1)
        self.stats['avg_generation_time'] = (total_time + generation_time) / self.stats['total_queries']

        # Track confidence distribution
        confidence_bucket = int(confidence * 10) / 10  # Round to nearest 0.1
        self.stats['confidence_distribution'][confidence_bucket] += 1

    def get_success_rates(self) -> Dict[str, float]:
        """Get success rates by approach"""

        rates = {}
        for approach in self.stats['approach_total']:
            total = self.stats['approach_total'][approach]
            success = self.stats['approach_success'][approach]
            rates[approach] = success / total if total > 0 else 0.0

        return rates

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary"""

        total = self.stats['total_queries']
        successful = self.stats['successful_queries']

        return {
            'total_queries': total,
            'overall_success_rate': successful / total if total > 0 else 0.0,
            'avg_generation_time': self.stats['avg_generation_time'],
            'approach_success_rates': self.get_success_rates(),
            'confidence_distribution': dict(self.stats['confidence_distribution'])
        }


# Integration with existing GraphRAGQA class
class GraphRAGQAEnhanced:
    """Enhanced version of GraphRAGQA with advanced Cypher generation"""

    def __init__(self, base_graphrag_qa, llm_config: Dict[str, Any]):
        self.base_qa = base_graphrag_qa
        self.enhanced_generator = EnhancedCypherGenerator(
            neo4j_driver=base_graphrag_qa.driver,
            llm_config=llm_config
        )

        logger.info("Enhanced GraphRAG QA initialized with advanced Cypher generation")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Enhanced question answering with advanced Cypher generation"""

        logger.info(f"=== Enhanced GraphRAG Processing: {question} ===")

        # Step 1: Enhanced entity linking (use base implementation)
        potential_mentions = self.base_qa._extract_potential_entities(question)
        linked_entities = self.base_qa._link_entities(potential_mentions) if potential_mentions else {}

        # Step 2: Advanced Cypher generation
        generation_result = self.enhanced_generator.generate_cypher(question, linked_entities)

        cypher_query = generation_result['cypher_query']
        confidence_score = generation_result['confidence_score']

        # Step 3: Execute query if successful
        graph_results = None
        if cypher_query and confidence_score > 0.3:  # Minimum confidence threshold
            try:
                graph_results = self.base_qa._query_neo4j(cypher_query)
                logger.info(f"Query executed successfully: {len(graph_results)} results")

                # Update few-shot learning with success
                if hasattr(self.enhanced_generator, 'few_shot_manager'):
                    self.enhanced_generator.few_shot_manager.update_example_success(
                        question, cypher_query, success=True
                    )

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                graph_results = []

                # Update few-shot learning with failure
                if hasattr(self.enhanced_generator, 'few_shot_manager'):
                    self.enhanced_generator.few_shot_manager.update_example_success(
                        question, cypher_query, success=False
                    )
        else:
            logger.warning(f"Cypher generation failed or low confidence: {confidence_score}")
            graph_results = []

        # Step 4: Vector search (use base implementation)
        vector_top_k = 5
        similar_chunks = self.base_qa._query_vector_db(question, top_k=vector_top_k)

        # Step 5: Generate final answer (use base implementation)
        context_str = self.base_qa._format_context(graph_results, similar_chunks)
        answer_dict = self.base_qa._synthesize_answer(question, context_str, similar_chunks)

        # Step 6: Enhanced result with generation metadata
        answer_dict.update({
            'cypher_query': cypher_query,
            'cypher_confidence': confidence_score,
            'generation_approach': generation_result['approach_used'],
            'question_analysis': generation_result['question_analysis'],
            'matched_patterns': generation_result['matched_patterns'],
            'schema_elements_used': generation_result['schema_elements_used'],
            'generation_time': generation_result['generation_time'],
            'linked_entities': linked_entities,
            'validation_result': generation_result.get('validation_result'),
            'enhancement_metadata': {
                'total_candidates': generation_result['all_candidates'],
                'complexity_detected': generation_result['question_analysis']['complexity'].value,
                'category_detected': generation_result['question_analysis']['category'].value
            }
        })

        logger.info("=== Enhanced GraphRAG Processing Complete ===")
        return answer_dict

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        return self.enhanced_generator.query_stats.get_summary()


# Usage Example and Testing Framework
if __name__ == "__main__":
    print("=== Enhanced Cypher Generation System ===")

    # Example configuration
    test_config = {
        'model': 'gemini-1.5-flash',
        'api_key': 'your-api-key',
        'base_url': None
    }


    # Example usage with mock driver
    class MockDriver:
        def session(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query):
            # Mock result
            class MockResult:
                def single(self):
                    return {'labels': ['Entity', 'Equipment', 'Person'],
                            'types': ['contains', 'assigned_to', 'maintains']}

            return MockResult()


    # Initialize enhanced generator
    mock_driver = MockDriver()
    enhanced_generator = EnhancedCypherGenerator(mock_driver, test_config)

    # Test questions
    test_questions = [
        "Find equipment named pump in Building A",
        "Who is assigned to maintain the main compressor?",
        "What equipment is contained in Building B?",
        "When was the last inspection of safety valve SV-101?",
        "Show me all work orders for technician John Smith"
    ]

    print("\n--- Testing Enhanced Cypher Generation ---")

    for question in test_questions:
        print(f"\nQuestion: {question}")

        result = enhanced_generator.generate_cypher(question)

        print(f"Generated Query: {result['cypher_query']}")
        print(f"Approach: {result['approach_used']}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Category: {result['question_analysis']['category'].value}")
        print(f"Complexity: {result['question_analysis']['complexity'].value}")
        print(f"Patterns Matched: {len(result['matched_patterns'])}")
        print("-" * 60)

    # Show statistics
    stats = enhanced_generator.query_stats.get_summary()
    print(f"\nGeneration Statistics:")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['overall_success_rate']:.3f}")
    print(f"Avg Generation Time: {stats['avg_generation_time']:.3f}s")

    print("\n✅ Enhanced Cypher Generation System Ready for Integration!")