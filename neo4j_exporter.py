import logging
import re
from neo4j import GraphDatabase, Session, Transaction
from typing import List, Dict, Optional, Any, Set, Tuple
import sys
from pathlib import Path
import json
import configparser
import time
import os
from dotenv import load_dotenv

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
    logger.info("✅ Multi-provider LLM system available for Neo4j Exporter")
except ImportError as e:
    NEW_LLM_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Multi-provider LLM system not available: {e}. Using legacy system.")

try:
    import tomllib
except ImportError:
    try:
        import toml
    except ImportError:
        tomllib = None
        toml = None

# FIXED: Reduced logging verbosity - only log at INFO level by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class Neo4jExporter:
    """
    ENHANCED: Neo4j exporter with OCR metadata integration AND multi-provider LLM support.
    """

    def __init__(self, uri: str, user: str, password: str, config: Optional[Dict[str, Any]] = None,
                 enable_multi_provider_llm: bool = True):
        """
        ENHANCED: Initialize the exporter with multi-provider LLM support.

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            config: Configuration dictionary for multi-provider LLM
            enable_multi_provider_llm: Whether to enable multi-provider LLM features
        """
        self.driver: Optional[GraphDatabase.driver] = None

        # ENHANCED: Store configuration for multi-provider LLM
        self.config = config or {}
        self.enable_multi_provider_llm = enable_multi_provider_llm
        self.llm_managers = {}

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j at %s. Error: %s", uri, e)
            raise ConnectionError(f"Failed to connect to Neo4j: {e}") from e

        # ENHANCED: Initialize multi-provider LLM system
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            self._initialize_multi_provider_llm()

        logger.info(f"Neo4j Exporter initialized. Multi-provider LLM: {self.enable_multi_provider_llm}")

    def _initialize_multi_provider_llm(self):
        """ENHANCED: Initialize multi-provider LLM system for Neo4j export tasks"""
        try:
            # Import the main LLM configuration manager
            from GraphRAG_Document_AI_Platform import get_llm_config_manager

            main_llm_manager = get_llm_config_manager(self.config)

            # Create task-specific LLM managers for Neo4j export tasks
            export_tasks = ['entity_validation', 'relationship_validation', 'semantic_enrichment', 'data_quality_check']

            for task in export_tasks:
                try:
                    self.llm_managers[task] = main_llm_manager.get_llm_manager(task)
                    logger.info(f"✅ Initialized LLM manager for {task}")
                except Exception as e:
                    logger.warning(f"Could not initialize LLM manager for {task}: {e}")
                    self.llm_managers[task] = None

        except ImportError as e:
            logger.warning(f"Could not import main LLM configuration manager: {e}")
            self.llm_managers = {}

    def _enhanced_llm_call(self, task_name: str, prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
        """
        ENHANCED: Enhanced LLM call with multi-provider support for Neo4j export tasks.
        Falls back to basic functionality if enhanced LLM is not available.
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
                logger.warning(f"Enhanced LLM failed for {task_name}: {e}, falling back to basic functionality")

        # Fall back to basic functionality (no LLM call)
        logger.debug(f"🔄 Enhanced LLM not available for {task_name}, using basic functionality")
        return None

    def close(self):
        """Closes the Neo4j database driver connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Closed Neo4j connection")
                self.driver = None
            except Exception as e:
                logger.error("Error closing Neo4j connection: %s", e)

    def _sanitize_label(self, label: str) -> Optional[str]:
        """Sanitizes a string to be used as a Neo4j node label."""
        if not isinstance(label, str) or not label.strip():
            return None

        sanitized = label.strip()
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', sanitized)

        if sanitized and sanitized[0].isalpha():
            return sanitized
        else:
            return None

    def sanitize_predicate(self, predicate: str, to_upper: bool = True, replace_non_alnum: bool = True) -> str:
        """Sanitizes a predicate string to be used as a Neo4j relationship type."""
        if not isinstance(predicate, str):
            logger.warning("Predicate is not a string: %s", predicate)
            return ""

        result = predicate.strip()
        if not result:
            return ""

        if to_upper:
            result = result.upper()
        if replace_non_alnum:
            result = re.sub(r'[^a-zA-Z0-9_]+', '_', result)
            result = result.strip('_')
            if result and result[0].isdigit():
                result = '_' + result

        if not result:
            logger.warning("Predicate '%s' resulted in empty string after sanitization", predicate)
            return ""

        return result

    def validate_entity_with_llm(self, entity_name: str, entity_type: str, context: str = "") -> Dict[str, Any]:
        """
        ENHANCED: Validate entity using multi-provider LLM.

        Args:
            entity_name: Name of the entity to validate
            entity_type: Type of the entity
            context: Additional context for validation

        Returns:
            Dictionary with validation results
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {'valid': True, 'confidence': 0.5, 'suggestions': []}

        try:
            validation_prompt = f"""
            Please validate the following entity and provide feedback:

            Entity Name: {entity_name}
            Entity Type: {entity_type}
            Context: {context}

            Please evaluate:
            1. Whether the entity name is appropriate for the given type
            2. If the entity type is correct
            3. Any normalization suggestions
            4. Confidence in the entity classification

            Provide a confidence score (0-1) and specific suggestions if any.
            """

            system_prompt = """You are an expert at validating and normalizing entities in knowledge graphs. Provide accurate assessments of entity quality and suggestions for improvement."""

            response = self._enhanced_llm_call(
                task_name='entity_validation',
                prompt=validation_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.1
            )

            if response:
                # Simple parsing - in production, you'd want more sophisticated parsing
                confidence = 0.8  # Default confidence
                suggestions = []

                if 'invalid' in response.lower() or 'incorrect' in response.lower():
                    confidence = 0.3
                    suggestions.append("Entity may have classification issues based on LLM analysis")

                if 'normalize' in response.lower() or 'standardize' in response.lower():
                    suggestions.append("Consider normalizing entity name format")

                return {
                    'valid': confidence > 0.5,
                    'confidence': confidence,
                    'suggestions': suggestions,
                    'llm_feedback': response
                }

        except Exception as e:
            logger.error(f"Failed to validate entity with LLM: {e}")

        return {'valid': True, 'confidence': 0.5, 'suggestions': []}

    def validate_relationship_with_llm(self, subject: str, predicate: str, object_: str, context: str = "") -> Dict[
        str, Any]:
        """
        ENHANCED: Validate relationship using multi-provider LLM.

        Args:
            subject: Subject entity
            predicate: Relationship predicate
            object_: Object entity
            context: Additional context for validation

        Returns:
            Dictionary with validation results
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {'valid': True, 'confidence': 0.5, 'suggestions': []}

        try:
            validation_prompt = f"""
            Please validate the following relationship and provide feedback:

            Subject: {subject}
            Predicate: {predicate}
            Object: {object_}
            Context: {context}

            Please evaluate:
            1. Whether this relationship makes logical sense
            2. If the predicate is appropriate for connecting these entities
            3. Any suggestions for improving the relationship representation
            4. Confidence in the relationship validity

            Provide a confidence score (0-1) and specific suggestions if any.
            """

            system_prompt = """You are an expert at validating relationships in knowledge graphs. Assess whether relationships are logically valid and provide suggestions for improvement."""

            response = self._enhanced_llm_call(
                task_name='relationship_validation',
                prompt=validation_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.1
            )

            if response:
                # Simple parsing - in production, you'd want more sophisticated parsing
                confidence = 0.8  # Default confidence
                suggestions = []

                if 'invalid' in response.lower() or 'nonsensical' in response.lower():
                    confidence = 0.3
                    suggestions.append("Relationship may not be logically valid based on LLM analysis")

                if 'improve' in response.lower() or 'better' in response.lower():
                    suggestions.append("Consider improving relationship representation")

                return {
                    'valid': confidence > 0.5,
                    'confidence': confidence,
                    'suggestions': suggestions,
                    'llm_feedback': response
                }

        except Exception as e:
            logger.error(f"Failed to validate relationship with LLM: {e}")

        return {'valid': True, 'confidence': 0.5, 'suggestions': []}

    def enrich_entities_with_llm(self, entities: List[Dict[str, str]], context: str = "") -> List[Dict[str, Any]]:
        """
        ENHANCED: Enrich entities with additional semantic information using LLM.

        Args:
            entities: List of entities to enrich
            context: Additional context for enrichment

        Returns:
            List of enriched entities with additional metadata
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return entities  # Return original entities if enhancement not available

        enriched_entities = []

        for entity in entities:
            try:
                entity_name = entity.get('name', '')
                entity_type = entity.get('type', '')

                enrichment_prompt = f"""
                Provide semantic enrichment for the following entity:

                Entity: {entity_name}
                Type: {entity_type}
                Context: {context}

                Please provide:
                1. Alternative names or aliases
                2. Category refinement suggestions
                3. Key attributes or properties this entity might have
                4. Related entity types it commonly connects to

                Keep the response concise and structured.
                """

                system_prompt = """You are an expert at semantic enrichment of entities in knowledge graphs. Provide useful metadata and attributes for entities."""

                response = self._enhanced_llm_call(
                    task_name='semantic_enrichment',
                    prompt=enrichment_prompt,
                    system_prompt=system_prompt,
                    max_tokens=200,
                    temperature=0.3
                )

                # Create enriched entity
                enriched_entity = entity.copy()

                if response:
                    enriched_entity['llm_enrichment'] = response
                    enriched_entity['enriched'] = True
                else:
                    enriched_entity['enriched'] = False

                enriched_entities.append(enriched_entity)

            except Exception as e:
                logger.warning(f"Failed to enrich entity {entity.get('name', 'unknown')}: {e}")
                enriched_entities.append(entity)  # Add original entity if enrichment fails

        return enriched_entities

    def store_document_metadata(self, ocr_result: Dict, document_id: str) -> bool:
        """
        Store document-level metadata as nodes in Neo4j.

        Args:
            ocr_result: Enhanced OCR result with metadata
            document_id: Unique document identifier
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                # Extract metadata from enhanced OCR result
                file_metadata = ocr_result.get('file_metadata', {})
                content_metadata = ocr_result.get('content_metadata', {})
                quality_metrics = ocr_result.get('quality_metrics', {})
                doc_classification = ocr_result.get('document_classification', {})
                detected_entities = ocr_result.get('detected_entities', {})
                chunk_metadata = ocr_result.get('chunk_metadata', [])

                # Create Document node with comprehensive metadata
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.filename = $filename,
                        d.file_type = $file_type,
                        d.file_size = $file_size,
                        d.file_hash = $file_hash,
                        d.ocr_method = $ocr_method,
                        d.ocr_confidence = $ocr_confidence,
                        d.processing_time = $processing_time,
                        d.text_length = $text_length,
                        d.document_type = $document_type,
                        d.document_category = $document_category,
                        d.classification_confidence = $classification_confidence,
                        d.language = $language,
                        d.page_count = $page_count,
                        d.upload_timestamp = $upload_timestamp,
                        d.processing_timestamp = $processing_timestamp,
                        d.quality_score = $quality_score,
                        d.readability_score = $readability_score,
                        d.complexity = $complexity,
                        d.has_tables = $has_tables,
                        d.has_financial_data = $has_financial_data,
                        d.word_count = $word_count,
                        d.character_count = $character_count,
                        d.paragraph_count = $paragraph_count,
                        d.created_at = timestamp()
                """, {
                    'doc_id': document_id,
                    'filename': file_metadata.get('original_filename', 'unknown'),
                    'file_type': file_metadata.get('mime_type', 'unknown'),
                    'file_size': file_metadata.get('file_size_bytes', 0),
                    'file_hash': file_metadata.get('file_hash', ''),
                    'ocr_method': ocr_result.get('method_used', 'unknown'),
                    'ocr_confidence': ocr_result.get('confidence', 0.0),
                    'processing_time': ocr_result.get('processing_time', 0.0),
                    'text_length': len(ocr_result.get('text', '')),
                    'document_type': doc_classification.get('document_type', 'unknown'),
                    'document_category': doc_classification.get('category', 'general'),
                    'classification_confidence': doc_classification.get('classification_confidence', 0.0),
                    'language': content_metadata.get('language_detected', 'en'),
                    'page_count': 1,  # Could be enhanced for multi-page
                    'upload_timestamp': file_metadata.get('upload_timestamp', ''),
                    'processing_timestamp': ocr_result.get('processing_metadata', {}).get('extraction_timestamp', ''),
                    'quality_score': quality_metrics.get('quality_score', 0.0),
                    'readability_score': quality_metrics.get('readability_score', 0.0),
                    'complexity': quality_metrics.get('complexity', 'unknown'),
                    'has_tables': content_metadata.get('has_tables', False),
                    'has_financial_data': content_metadata.get('has_financial_data', False),
                    'word_count': content_metadata.get('word_count', 0),
                    'character_count': content_metadata.get('character_count', 0),
                    'paragraph_count': content_metadata.get('paragraph_count', 0)
                })

                # Create Chunk nodes and link to Document
                if chunk_metadata:
                    for chunk in chunk_metadata:
                        chunk_id = chunk.get('chunk_id')
                        chunk_text = chunk.get('chunk_text', '')

                        if chunk_id and chunk_text.strip():
                            # Create/update Chunk node
                            session.run("""
                                MERGE (c:Chunk {id: $chunk_id})
                                SET c.text = $chunk_text,
                                    c.word_count = $word_count,
                                    c.character_count = $character_count,
                                    c.start_word = $start_word,
                                    c.end_word = $end_word,
                                    c.created_at = timestamp()
                            """, {
                                'chunk_id': chunk_id,
                                'chunk_text': chunk_text,
                                'word_count': chunk.get('word_count', 0),
                                'character_count': chunk.get('character_count', 0),
                                'start_word': chunk.get('start_word', 0),
                                'end_word': chunk.get('end_word', 0)
                            })

                            # Link Document to Chunk
                            session.run("""
                                MATCH (d:Document {id: $doc_id})
                                MATCH (c:Chunk {id: $chunk_id})
                                MERGE (d)-[:CONTAINS_CHUNK]->(c)
                            """, {
                                'doc_id': document_id,
                                'chunk_id': chunk_id
                            })

                # ENHANCED: Create Entity nodes with LLM validation if enabled
                if detected_entities:
                    entities_to_process = []
                    for entity_type, entities in detected_entities.items():
                        for entity_name in entities:
                            entities_to_process.append({'name': entity_name, 'type': entity_type})

                    # Enrich entities with LLM if available
                    if self.enable_multi_provider_llm:
                        context = f"Document: {file_metadata.get('original_filename', 'unknown')}, Type: {doc_classification.get('document_type', 'unknown')}"
                        entities_to_process = self.enrich_entities_with_llm(entities_to_process, context)

                    for entity_info in entities_to_process:
                        entity_name = entity_info['name']
                        entity_type = entity_info['type']

                        # Validate entity with LLM if available
                        validation_result = self.validate_entity_with_llm(entity_name, entity_type)

                        # Create/update Entity node with type and validation results
                        sanitized_entity_type = self._sanitize_label(entity_type)

                        entity_params = {
                            'entity_name': entity_name,
                            'entity_type': entity_type,
                            'validation_confidence': validation_result.get('confidence', 1.0),
                            'llm_validated': self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE,
                            'enriched': entity_info.get('enriched', False)
                        }

                        # Add enrichment data if available
                        if entity_info.get('llm_enrichment'):
                            entity_params['llm_enrichment'] = entity_info['llm_enrichment']

                        if sanitized_entity_type:
                            session.run(f"""
                                MERGE (e:Entity:`{sanitized_entity_type}` {{name: $entity_name}})
                                SET e.entity_type = $entity_type,
                                    e.validation_confidence = $validation_confidence,
                                    e.llm_validated = $llm_validated,
                                    e.enriched = $enriched,
                                    e.llm_enrichment = $llm_enrichment,
                                    e.created_at = coalesce(e.created_at, timestamp())
                            """, entity_params)
                        else:
                            session.run("""
                                MERGE (e:Entity {name: $entity_name})
                                SET e.entity_type = $entity_type,
                                    e.validation_confidence = $validation_confidence,
                                    e.llm_validated = $llm_validated,
                                    e.enriched = $enriched,
                                    e.llm_enrichment = $llm_enrichment,
                                    e.created_at = coalesce(e.created_at, timestamp())
                            """, entity_params)

                        # Link Document to Entity
                        session.run("""
                            MATCH (d:Document {id: $doc_id})
                            MATCH (e:Entity {name: $entity_name})
                            MERGE (d)-[:MENTIONS {entity_type: $entity_type, confidence: $confidence}]->(e)
                        """, {
                            'doc_id': document_id,
                            'entity_name': entity_name,
                            'entity_type': entity_type,
                            'confidence': ocr_result.get('confidence', 1.0)
                        })

                logger.info(f"Stored document metadata for {document_id} with enhanced LLM validation")
                return True

        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            return False

    def store_triples_with_metadata(self, triples: List[Dict], ocr_metadata: Dict = None) -> Tuple[bool, int]:
        """
        ENHANCED: Triple storage with LLM validation and metadata.

        Args:
            triples: List of extracted triples
            ocr_metadata: Metadata from enhanced OCR processing
        """
        if not triples:
            return True, 0

        try:
            with self.driver.session(database="neo4j") as session:
                # Store document metadata first if provided
                if ocr_metadata:
                    doc_id = ocr_metadata.get('file_metadata', {}).get('file_id', 'unknown')
                    self.store_document_metadata(ocr_metadata, doc_id)

                # ENHANCED: Process triples with LLM validation
                processed_count = 0
                skipped_count = 0
                validated_count = 0

                for triple in triples:
                    subject = triple.get("subject", "").strip()
                    predicate = triple.get("predicate", "").strip()
                    object_ = triple.get("object", "").strip()

                    if not all([subject, predicate, object_]):
                        skipped_count += 1
                        continue

                    # ENHANCED: Validate relationship with LLM if enabled
                    validation_result = {'valid': True, 'confidence': 1.0, 'suggestions': []}
                    if self.enable_multi_provider_llm:
                        context = f"Source: {ocr_metadata.get('file_metadata', {}).get('original_filename', 'unknown')}" if ocr_metadata else ""
                        validation_result = self.validate_relationship_with_llm(subject, predicate, object_, context)
                        if validation_result['valid']:
                            validated_count += 1

                    # Enhanced triple storage with validation metadata
                    rel_type = self.sanitize_predicate(predicate)

                    if not rel_type:
                        skipped_count += 1
                        continue

                    # Extract enhanced metadata
                    subject_type = triple.get('subject_type', 'Entity')
                    object_type = triple.get('object_type', 'Entity')

                    try:
                        session.run(f"""
                            MERGE (s:Entity {{name: $subject}})
                            SET s.entity_type = $subject_type,
                                s.created_at = coalesce(s.created_at, timestamp())

                            MERGE (o:Entity {{name: $object}})
                            SET o.entity_type = $object_type,
                                o.created_at = coalesce(o.created_at, timestamp())

                            MERGE (s)-[r:`{rel_type}`]->(o)
                            SET r.original = $original_predicate,
                                r.confidence = $confidence,
                                r.extraction_method = $extraction_method,
                                r.source_document = $source_document,
                                r.chunk_id = $chunk_id,
                                r.inferred = $inferred,
                                r.llm_validated = $llm_validated,
                                r.validation_confidence = $validation_confidence,
                                r.llm_feedback = $llm_feedback,
                                r.created_at = coalesce(r.created_at, timestamp()),
                                r.updated_at = timestamp()
                        """, {
                            'subject': subject,
                            'object': object_,
                            'subject_type': subject_type,
                            'object_type': object_type,
                            'original_predicate': predicate,
                            'confidence': triple.get('confidence', 1.0),
                            'extraction_method': ocr_metadata.get('method_used',
                                                                  'unknown') if ocr_metadata else 'unknown',
                            'source_document': ocr_metadata.get('file_metadata', {}).get('file_id',
                                                                                         'unknown') if ocr_metadata else 'unknown',
                            'chunk_id': triple.get('chunk_id', 'unknown'),
                            'inferred': triple.get('inferred', False),
                            'llm_validated': self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE,
                            'validation_confidence': validation_result.get('confidence', 1.0),
                            'llm_feedback': validation_result.get('llm_feedback', '')
                        })

                        processed_count += 1

                    except Exception as e:
                        logger.error(f"Failed to store triple {subject}-{predicate}-{object_}: {e}")
                        skipped_count += 1
                        continue

                logger.info(
                    f"Stored {processed_count} triples with enhanced LLM validation ({validated_count} validated), skipped {skipped_count}")
                return True, processed_count

        except Exception as e:
            logger.error(f"Failed to store enhanced triples: {e}")
            return False, 0

    def store_triples(self, triples: List[Dict], store_chunks: bool = True, sanitize: bool = True) -> Tuple[bool, int]:
        """
        Original store_triples method maintained for backward compatibility.
        """
        num_attempted = len(triples)
        if not triples:
            logger.warning("No triples provided to store in Neo4j")
            return True, 0

        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False, 0

        processed_chunk_ids: Set[str] = set()

        logger.info(f"Starting batch transaction to store/update {num_attempted} triples...")
        try:
            with self.driver.session(database="neo4j") as session:
                session.execute_write(
                    self._store_triples_transaction,
                    triples,
                    processed_chunk_ids,
                    store_chunks,
                    sanitize
                )
            logger.info(f"Batch transaction completed successfully for {num_attempted} triples")
            return True, num_attempted

        except Exception as e:
            logger.error(f"Batch transaction failed: {e}")
            return False, 0

    def _store_triples_transaction(self, tx: Transaction, triples: List[Dict], processed_chunk_ids: Set[str],
                                   store_chunks: bool, sanitize: bool):
        """Internal function with reduced debug logging."""
        skipped_in_batch = 0
        processed_in_batch = 0

        for i, triple in enumerate(triples):
            subject = triple.get("subject")
            predicate = triple.get("predicate")
            object_ = triple.get("object")
            subject_type = triple.get("subject_type")
            object_type = triple.get("object_type")

            # Validation
            if not all(isinstance(item, str) and item.strip() for item in [subject, predicate, object_]):
                skipped_in_batch += 1
                continue

            subject = subject.strip()
            predicate = predicate.strip()
            object_ = object_.strip()

            # Sanitize labels and relationship type
            sanitized_subj_label = self._sanitize_label(subject_type) if subject_type else None
            sanitized_obj_label = self._sanitize_label(object_type) if object_type else None
            rel_type = self.sanitize_predicate(predicate) if sanitize else predicate

            if not rel_type:
                skipped_in_batch += 1
                continue

            # Optional properties
            inferred = triple.get("inferred", False)
            try:
                confidence = float(triple.get("confidence", 1.0))
            except (ValueError, TypeError):
                confidence = 1.0

            # Standard MERGE without APOC (more compatible)
            try:
                # Create subject entity
                if sanitized_subj_label:
                    tx.run(f"""
                        MERGE (s:Entity:`{sanitized_subj_label}` {{name: $subject}})
                        SET s.created_at = coalesce(s.created_at, timestamp())
                    """, subject=subject)
                else:
                    tx.run("""
                        MERGE (s:Entity {name: $subject})
                        SET s.created_at = coalesce(s.created_at, timestamp())
                    """, subject=subject)

                # Create object entity
                if sanitized_obj_label:
                    tx.run(f"""
                        MERGE (o:Entity:`{sanitized_obj_label}` {{name: $object}})
                        SET o.created_at = coalesce(o.created_at, timestamp())
                    """, object=object_)
                else:
                    tx.run("""
                        MERGE (o:Entity {name: $object})
                        SET o.created_at = coalesce(o.created_at, timestamp())
                    """, object=object_)

                # Create relationship
                tx.run(f"""
                    MATCH (s:Entity {{name: $subject}})
                    MATCH (o:Entity {{name: $object}})
                    MERGE (s)-[r:`{rel_type}`]->(o)
                    SET r.original = $original_predicate,
                        r.inferred = $inferred,
                        r.confidence = $confidence,
                        r.created_at = coalesce(r.created_at, timestamp()),
                        r.updated_at = timestamp()
                """, {
                    'subject': subject,
                    'object': object_,
                    'original_predicate': predicate,
                    'inferred': inferred,
                    'confidence': confidence
                })

            except Exception as e:
                logger.error("Failed to run relationship MERGE for triple #%d: %s", i, e)
                skipped_in_batch += 1
                continue

            # Chunk handling (existing logic)
            if store_chunks:
                chunk_id_val = triple.get("chunk_id", triple.get("chunk"))
                chunk_text = triple.get("chunk_text", "")

                if chunk_id_val is not None and isinstance(chunk_text, str) and chunk_text.strip():
                    chunk_id = str(chunk_id_val)
                    chunk_text_stripped = chunk_text.strip()

                    if chunk_id not in processed_chunk_ids:
                        try:
                            tx.run(
                                """
                                MERGE (c:Chunk {id: $chunk_id})
                                ON CREATE SET c.text = $chunk_text, c.created_at = timestamp()
                                ON MATCH SET c.updated_at = timestamp()
                                """,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text_stripped
                            )
                            processed_chunk_ids.add(chunk_id)
                        except Exception as e:
                            logger.error("Failed to MERGE Chunk node with id %s: %s", chunk_id, e)

                    if chunk_id in processed_chunk_ids:
                        try:
                            tx.run(
                                """
                                MATCH (s:Entity {name: $subject})
                                MATCH (o:Entity {name: $object})
                                MATCH (c:Chunk {id: $chunk_id})
                                MERGE (s)-[:FROM_CHUNK]->(c)
                                MERGE (o)-[:FROM_CHUNK]->(c)
                                """,
                                chunk_id=chunk_id,
                                subject=subject,
                                object=object_
                            )
                        except Exception as e:
                            logger.error("Failed to link entities to chunk '%s': %s", chunk_id, e)

            processed_in_batch += 1

        # Transaction completion logging
        if skipped_in_batch > 0:
            logger.warning(f"Transaction processed: {processed_in_batch}, skipped: {skipped_in_batch}")
        else:
            logger.info(f"Transaction processed: {processed_in_batch} triples successfully")

    def perform_data_quality_check_with_llm(self, document_id: str) -> Dict[str, Any]:
        """
        ENHANCED: Perform comprehensive data quality check using LLM.

        Args:
            document_id: Document to analyze

        Returns:
            Dictionary with quality check results
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {'quality_score': 0.8, 'issues': [], 'suggestions': []}

        try:
            with self.driver.session(database="neo4j") as session:
                # Get sample data for quality check
                sample_data = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                    MATCH (s:Entity)-[r]->(o:Entity)
                    WHERE (s)-[:FROM_CHUNK]->(c) AND (o)-[:FROM_CHUNK]->(c)
                    RETURN s.name as subject, type(r) as predicate, o.name as object, r.confidence as confidence
                    LIMIT 10
                """, doc_id=document_id)

                sample_triples = [record.data() for record in sample_data]

                if not sample_triples:
                    return {'quality_score': 0.5, 'issues': ['No data found for quality check'], 'suggestions': []}

                # Create quality check prompt
                triples_text = "\n".join(
                    [f"{t['subject']} -> {t['predicate']} -> {t['object']} (confidence: {t['confidence']})"
                     for t in sample_triples])

                quality_check_prompt = f"""
                Please analyze the quality of the following knowledge graph data:

                Document ID: {document_id}
                Sample Relationships:
                {triples_text}

                Please evaluate:
                1. Data consistency and logical coherence
                2. Relationship quality and appropriateness
                3. Entity naming conventions
                4. Potential data quality issues
                5. Suggestions for improvement

                Provide a quality score (0-1) and specific issues/suggestions.
                """

                system_prompt = """You are a data quality expert specializing in knowledge graphs. Analyze the provided data for quality issues and provide actionable recommendations."""

                response = self._enhanced_llm_call(
                    task_name='data_quality_check',
                    prompt=quality_check_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.1
                )

                if response:
                    # Parse quality check response
                    quality_score = 0.8  # Default
                    issues = []
                    suggestions = []

                    # Simple parsing - in production, you'd want more sophisticated parsing
                    if 'poor' in response.lower() or 'low quality' in response.lower():
                        quality_score = 0.3
                        issues.append("LLM identified potential quality issues")
                    elif 'good' in response.lower() or 'high quality' in response.lower():
                        quality_score = 0.9

                    if 'inconsistent' in response.lower():
                        issues.append("Data inconsistencies detected")

                    if 'improve' in response.lower():
                        suggestions.append("Consider implementing suggested improvements")

                    return {
                        'quality_score': quality_score,
                        'issues': issues,
                        'suggestions': suggestions,
                        'llm_analysis': response,
                        'analyzed_triples': len(sample_triples)
                    }

        except Exception as e:
            logger.error(f"Failed to perform data quality check: {e}")

        return {'quality_score': 0.5, 'issues': ['Quality check failed'], 'suggestions': []}

    def get_llm_provider_info(self) -> Dict[str, Any]:
        """ENHANCED: Get information about configured LLM providers for Neo4j export tasks."""
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

    def get_enhanced_system_health(self) -> Dict[str, Any]:
        """ENHANCED: Get system health including multi-provider LLM status."""
        health = {
            'neo4j_connected': False,
            'multi_provider_llm_active': False,
            'llm_providers_ready': [],
            'validation_enabled': False,
            'enrichment_enabled': False
        }

        try:
            # Check Neo4j connection
            if self.driver:
                test_result = self.driver.execute_query("RETURN 1 as test", database_="neo4j")
                health['neo4j_connected'] = bool(test_result)

            # Check multi-provider LLM
            if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
                health['multi_provider_llm_active'] = True
                health['validation_enabled'] = bool(self.llm_managers.get('entity_validation'))
                health['enrichment_enabled'] = bool(self.llm_managers.get('semantic_enrichment'))

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

    def get_document_analysis_summary(self, document_id: str) -> Dict:
        """
        ENHANCED: Get comprehensive analysis summary for a document with LLM insights.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary with document analysis summary including LLM quality check
        """
        if not self.driver:
            return {}

        try:
            with self.driver.session(database="neo4j") as session:
                # Get document metadata
                doc_result = session.run("""
                    MATCH (d:Document {id: $doc_id})
                    RETURN d
                """, doc_id=document_id)

                doc_record = doc_result.single()
                if not doc_record:
                    return {}

                doc_data = dict(doc_record['d'])

                # Get entity mentions
                entities_result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:MENTIONS]->(e:Entity)
                    RETURN e.name as entity, e.entity_type as type, e.validation_confidence as validation, 
                           e.llm_validated as llm_validated, count(*) as mentions
                    ORDER BY mentions DESC
                    LIMIT 20
                """, doc_id=document_id)

                entities = []
                for record in entities_result:
                    entity_data = {
                        'name': record['entity'],
                        'type': record['type'],
                        'mentions': record['mentions']
                    }
                    if record['validation']:
                        entity_data['validation_confidence'] = record['validation']
                    if record['llm_validated']:
                        entity_data['llm_validated'] = record['llm_validated']
                    entities.append(entity_data)

                # Get relationship summary with validation info
                rels_result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                    MATCH (s:Entity)-[r]->(o:Entity)
                    WHERE (s)-[:FROM_CHUNK]->(c) AND (o)-[:FROM_CHUNK]->(c)
                    RETURN type(r) as relationship_type, count(*) as count, 
                           avg(r.validation_confidence) as avg_validation,
                           sum(CASE WHEN r.llm_validated = true THEN 1 ELSE 0 END) as llm_validated_count
                    ORDER BY count DESC
                    LIMIT 10
                """, doc_id=document_id)

                relationships = []
                for record in rels_result:
                    rel_data = {
                        'type': record['relationship_type'],
                        'count': record['count']
                    }
                    if record['avg_validation']:
                        rel_data['avg_validation_confidence'] = record['avg_validation']
                    if record['llm_validated_count']:
                        rel_data['llm_validated_count'] = record['llm_validated_count']
                    relationships.append(rel_data)

                # Get chunk summary
                chunks_result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                    RETURN count(c) as chunk_count, avg(c.word_count) as avg_words_per_chunk
                """, doc_id=document_id)

                chunk_summary = chunks_result.single()

                # ENHANCED: Perform LLM quality check
                quality_check = self.perform_data_quality_check_with_llm(document_id)

                analysis_summary = {
                    'document_metadata': doc_data,
                    'top_entities': entities,
                    'relationship_summary': relationships,
                    'chunk_summary': {
                        'total_chunks': chunk_summary['chunk_count'] if chunk_summary else 0,
                        'avg_words_per_chunk': chunk_summary['avg_words_per_chunk'] if chunk_summary else 0
                    },
                    'quality_analysis': quality_check,
                    'llm_enhancement_summary': {
                        'validation_enabled': self.enable_multi_provider_llm and bool(
                            self.llm_managers.get('entity_validation')),
                        'enrichment_enabled': self.enable_multi_provider_llm and bool(
                            self.llm_managers.get('semantic_enrichment')),
                        'quality_check_enabled': self.enable_multi_provider_llm and bool(
                            self.llm_managers.get('data_quality_check'))
                    },
                    'analysis_timestamp': time.time()
                }

                return analysis_summary

        except Exception as e:
            logger.error(f"Failed to get enhanced document analysis: {e}")
            return {}

    def get_related_facts_with_context(self, entity_name: str, predicate_filter: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """Retrieve facts connected to a given entity (existing method)."""
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return []

        logger.info("Fetching related facts for entity: '%s'", entity_name)

        base_query = """
        // Outgoing relationships
        MATCH (e:Entity)-[r]->(o:Entity)
        WHERE toLower(e.name) = toLower($name)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c1:Chunk)
        OPTIONAL MATCH (o)-[:FROM_CHUNK]->(c2:Chunk)
        WITH e, r, o, coalesce(c1.text, c2.text, "") AS chunk_text
        {predicate_filter_clause}
        RETURN e.name AS subject, r.original AS predicate, o.name AS object, chunk_text, type(r) as type,
               r.validation_confidence as validation, r.llm_validated as llm_validated

        UNION

        // Incoming relationships
        MATCH (s:Entity)-[r]->(e:Entity)
        WHERE toLower(e.name) = toLower($name)
        OPTIONAL MATCH (s)-[:FROM_CHUNK]->(c1:Chunk)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c2:Chunk)
        WITH s, r, e, coalesce(c1.text, c2.text, "") AS chunk_text
        {predicate_filter_clause}
        RETURN s.name AS subject, r.original AS predicate, e.name AS object, chunk_text, type(r) as type,
               r.validation_confidence as validation, r.llm_validated as llm_validated
        """

        clause = ""
        parameters: Dict[str, Any] = {"name": entity_name}
        if predicate_filter:
            clause = "WHERE toLower(r.original) CONTAINS toLower($predicate)"
            parameters["predicate"] = predicate_filter

        query = base_query.format(predicate_filter_clause=clause)

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, **parameters)
                records = [record.data() for record in result]
                logger.info("Found %d related facts for entity '%s'", len(records), entity_name)
                return records
        except Exception as e:
            logger.error("Failed to execute get_related_facts query: %s", e)
            return []

    def get_graph_stats(self) -> Dict[str, int]:
        """ENHANCED: Calculate basic statistics about the graph including LLM enhancement metrics."""
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return {"entity_count": 0, "document_count": 0, "chunk_count": 0, "relationship_count": 0}

        logger.info("Calculating enhanced graph statistics...")
        query = """
        CALL () { MATCH (n:Entity) RETURN count(n) AS entity_count }
        CALL { MATCH (n:Document) RETURN count(n) AS document_count }
        CALL { MATCH (n:Chunk) RETURN count(n) AS chunk_count }
        CALL { MATCH ()-[r]->() RETURN count(r) AS relationship_count }
        CALL { MATCH (n:Entity) WHERE n.llm_validated = true RETURN count(n) AS validated_entities }
        CALL { MATCH ()-[r]->() WHERE r.llm_validated = true RETURN count(r) AS validated_relationships }
        CALL { MATCH (n:Entity) WHERE n.enriched = true RETURN count(n) AS enriched_entities }
        RETURN entity_count, document_count, chunk_count, relationship_count, 
               validated_entities, validated_relationships, enriched_entities
        """

        try:
            records, summary, keys = self.driver.execute_query(query, database_="neo4j")
            record = records[0] if records else None

            if record:
                stats = {
                    "entity_count": record.get("entity_count", 0),
                    "document_count": record.get("document_count", 0),
                    "chunk_count": record.get("chunk_count", 0),
                    "relationship_count": record.get("relationship_count", 0),
                    "validated_entities": record.get("validated_entities", 0),
                    "validated_relationships": record.get("validated_relationships", 0),
                    "enriched_entities": record.get("enriched_entities", 0),
                }

                # Calculate enhancement percentages
                if stats["entity_count"] > 0:
                    stats["validation_percentage"] = (stats["validated_entities"] / stats["entity_count"]) * 100
                    stats["enrichment_percentage"] = (stats["enriched_entities"] / stats["entity_count"]) * 100
                else:
                    stats["validation_percentage"] = 0
                    stats["enrichment_percentage"] = 0

                logger.info("Enhanced Graph Stats: %s", stats)
                return stats
            else:
                logger.warning("Graph stats query returned no record")
                return {"entity_count": 0, "document_count": 0, "chunk_count": 0, "relationship_count": 0}
        except Exception as e:
            logger.error("Failed to execute get_graph_stats query: %s", e)
            return {"entity_count": 0, "document_count": 0, "chunk_count": 0, "relationship_count": 0}

    def __enter__(self):
        """Enter the runtime context for using 'with Neo4jExporter(...)'."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and ensure connection closure."""
        self.close()
        return False


# ENHANCED: Factory functions for creating enhanced Neo4j exporters

def create_enhanced_neo4j_exporter(uri: str, user: str, password: str, config: Dict[str, Any],
                                   enable_multi_provider_llm: bool = True) -> Neo4jExporter:
    """
    Factory function for creating enhanced Neo4j exporter with multi-provider LLM support.

    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
        config: Configuration dictionary for multi-provider LLM
        enable_multi_provider_llm: Whether to enable multi-provider LLM features
    """
    return Neo4jExporter(
        uri=uri,
        user=user,
        password=password,
        config=config,
        enable_multi_provider_llm=enable_multi_provider_llm
    )


def create_standard_neo4j_exporter(uri: str, user: str, password: str) -> Neo4jExporter:
    """
    Factory function for creating standard Neo4j exporter (backward compatibility).
    """
    return Neo4jExporter(uri=uri, user=user, password=password, enable_multi_provider_llm=False)


# Example usage for enhanced OCR integration
if __name__ == "__main__":
    print("--- Enhanced Neo4j Exporter with Multi-Provider LLM Support ---")

    # ADDED: Load .env file first
    load_dotenv()

    config_path = Path("graph_config.ini")
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        # Read from config file
        uri = config.get("neo4j", "uri", fallback=None)
        user = config.get("neo4j", "user", fallback=None)
        password = config.get("neo4j", "password", fallback=None)

        # ADDED: Override with environment variables
        uri = os.getenv('NEO4J_URI') or uri
        user = os.getenv('NEO4J_USER') or user
        password = os.getenv('NEO4J_PASSWORD') or password

        if not all([uri, user, password]):
            print("ERROR: Missing Neo4j configuration")
            sys.exit(1)

        print(f"[INFO] Connecting to Neo4j at: {uri}")

        # Enhanced LLM configuration
        llm_config = {
            'entity_validation': {
                'provider': 'openai',
                'model': 'gpt-4',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'max_tokens': 300,
                'temperature': 0.1
            },
            'semantic_enrichment': {
                'provider': 'anthropic',
                'model': 'claude-3-sonnet',
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'max_tokens': 200,
                'temperature': 0.3
            },
            'data_quality_check': {
                'provider': 'gemini',
                'model': 'gemini-1.5-flash',
                'api_key': os.getenv('GOOGLE_API_KEY'),
                'max_tokens': 500,
                'temperature': 0.1
            }
        }

        with create_enhanced_neo4j_exporter(uri, user, password, llm_config) as exporter:
            # Check system health
            print("\n--- System Health Check ---")
            health = exporter.get_enhanced_system_health()
            print(f"Neo4j Connected: {health['neo4j_connected']}")
            print(f"Multi-provider LLM Active: {health['multi_provider_llm_active']}")
            print(f"Validation Enabled: {health['validation_enabled']}")
            print(f"Enrichment Enabled: {health['enrichment_enabled']}")

            # Get provider information
            print("\n--- LLM Provider Information ---")
            provider_info = exporter.get_llm_provider_info()
            for provider in provider_info.get('providers', []):
                print(f"Task: {provider['task']}, Provider: {provider['primary_provider']['name']}")

            # Example: Store OCR result with enhanced LLM validation
            sample_ocr_result = {
                'text': 'Sample extracted text...',
                'method_used': 'gemini_1.5_flash',
                'confidence': 0.95,
                'processing_time': 2.5,
                'file_metadata': {
                    'file_id': 'doc_12345',
                    'original_filename': 'sample_invoice.pdf',
                    'mime_type': 'application/pdf',
                    'file_size_bytes': 245760,
                    'file_hash': 'abc123def456',
                    'upload_timestamp': '2024-06-20T10:00:00Z'
                },
                'content_metadata': {
                    'word_count': 250,
                    'character_count': 1500,
                    'has_tables': True,
                    'has_financial_data': True,
                    'language_detected': 'en'
                },
                'quality_metrics': {
                    'quality_score': 0.92,
                    'readability_score': 75.0,
                    'complexity': 'medium'
                },
                'document_classification': {
                    'document_type': 'invoice',
                    'category': 'financial',
                    'classification_confidence': 0.88
                },
                'detected_entities': {
                    'companies': ['ABC Corp', 'XYZ Ltd'],
                    'wells': ['A-1', 'B-12'],
                    'formations': ['Daman Formation']
                },
                'chunk_metadata': [
                    {
                        'chunk_id': 'chunk_0',
                        'chunk_text': 'Invoice ABC Corp...',
                        'word_count': 125,
                        'character_count': 750,
                        'start_word': 0,
                        'end_word': 125
                    }
                ]
            }

            # Store document metadata with LLM enhancement
            print("\n--- Storing Enhanced Document Metadata ---")
            doc_id = sample_ocr_result['file_metadata']['file_id']
            exporter.store_document_metadata(sample_ocr_result, doc_id)

            # Example triples with enhanced validation
            sample_triples = [
                {
                    'subject': 'ABC Corp',
                    'predicate': 'issues_invoice_to',
                    'object': 'XYZ Ltd',
                    'subject_type': 'companies',
                    'object_type': 'companies',
                    'chunk_id': 'chunk_0',
                    'confidence': 0.95,
                    'inferred': False
                }
            ]

            # Store triples with LLM validation
            print("\n--- Storing Triples with LLM Validation ---")
            success, count = exporter.store_triples_with_metadata(sample_triples, sample_ocr_result)

            if success:
                print(f"Successfully stored {count} triples with LLM validation")

                # Get enhanced document analysis
                print("\n--- Enhanced Document Analysis ---")
                analysis = exporter.get_document_analysis_summary(doc_id)

                if analysis:
                    print(f"Document Type: {analysis['document_metadata'].get('document_type', 'unknown')}")
                    print(f"Quality Score: {analysis['quality_analysis'].get('quality_score', 'N/A')}")
                    print(f"LLM Validation Enabled: {analysis['llm_enhancement_summary']['validation_enabled']}")
                    print(f"LLM Enrichment Enabled: {analysis['llm_enhancement_summary']['enrichment_enabled']}")

                # Get enhanced graph stats
                print("\n--- Enhanced Graph Statistics ---")
                stats = exporter.get_graph_stats()
                for key, value in stats.items():
                    if 'percentage' in key:
                        print(f"  {key.replace('_', ' ').title()}: {value:.1f}%")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")

            else:
                print("Failed to store triples with LLM validation")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n--- Enhanced Neo4j Exporter with Multi-Provider LLM Complete ---")