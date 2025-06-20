import logging
import re
from neo4j import GraphDatabase, Session, Transaction
from typing import List, Dict, Optional, Any, Set, Tuple
import sys
from pathlib import Path
import json
import configparser
import time

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
    Enhanced Neo4j exporter with OCR metadata integration.
    """

    def __init__(self, uri: str, user: str, password: str):
        """Initialize the exporter and connect to Neo4j."""
        self.driver: Optional[GraphDatabase.driver] = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j at %s. Error: %s", uri, e)
            raise ConnectionError(f"Failed to connect to Neo4j: {e}") from e

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

                # Create Entity nodes and relationships with document context
                if detected_entities:
                    for entity_type, entities in detected_entities.items():
                        for entity_name in entities:
                            # Create/update Entity node with type
                            sanitized_entity_type = self._sanitize_label(entity_type)

                            if sanitized_entity_type:
                                session.run(f"""
                                    MERGE (e:Entity:`{sanitized_entity_type}` {{name: $entity_name}})
                                    SET e.entity_type = $entity_type,
                                        e.created_at = coalesce(e.created_at, timestamp())
                                """, {
                                    'entity_name': entity_name,
                                    'entity_type': entity_type
                                })
                            else:
                                session.run("""
                                    MERGE (e:Entity {name: $entity_name})
                                    SET e.entity_type = $entity_type,
                                        e.created_at = coalesce(e.created_at, timestamp())
                                """, {
                                    'entity_name': entity_name,
                                    'entity_type': entity_type
                                })

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

                logger.info(f"Stored document metadata for {document_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            return False

    def store_triples_with_metadata(self, triples: List[Dict], ocr_metadata: Dict = None) -> Tuple[bool, int]:
        """
        Enhanced triple storage that includes OCR metadata and document context.

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

                # Process triples with enhanced metadata
                processed_count = 0
                skipped_count = 0

                for triple in triples:
                    subject = triple.get("subject", "").strip()
                    predicate = triple.get("predicate", "").strip()
                    object_ = triple.get("object", "").strip()

                    if not all([subject, predicate, object_]):
                        skipped_count += 1
                        continue

                    # Enhanced triple storage with metadata
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
                            'inferred': triple.get('inferred', False)
                        })

                        processed_count += 1

                    except Exception as e:
                        logger.error(f"Failed to store triple {subject}-{predicate}-{object_}: {e}")
                        skipped_count += 1
                        continue

                logger.info(f"Stored {processed_count} triples with enhanced metadata, skipped {skipped_count}")
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

    def get_document_analysis_summary(self, document_id: str) -> Dict:
        """
        Get comprehensive analysis summary for a document.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary with document analysis summary
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
                    RETURN e.name as entity, e.entity_type as type, count(*) as mentions
                    ORDER BY mentions DESC
                    LIMIT 20
                """, doc_id=document_id)

                entities = [{'name': record['entity'], 'type': record['type'], 'mentions': record['mentions']}
                            for record in entities_result]

                # Get relationship summary
                rels_result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                    MATCH (s:Entity)-[r]->(o:Entity)
                    WHERE (s)-[:FROM_CHUNK]->(c) AND (o)-[:FROM_CHUNK]->(c)
                    RETURN type(r) as relationship_type, count(*) as count
                    ORDER BY count DESC
                    LIMIT 10
                """, doc_id=document_id)

                relationships = [{'type': record['relationship_type'], 'count': record['count']}
                                 for record in rels_result]

                # Get chunk summary
                chunks_result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                    RETURN count(c) as chunk_count, avg(c.word_count) as avg_words_per_chunk
                """, doc_id=document_id)

                chunk_summary = chunks_result.single()

                return {
                    'document_metadata': doc_data,
                    'top_entities': entities,
                    'relationship_summary': relationships,
                    'chunk_summary': {
                        'total_chunks': chunk_summary['chunk_count'] if chunk_summary else 0,
                        'avg_words_per_chunk': chunk_summary['avg_words_per_chunk'] if chunk_summary else 0
                    },
                    'analysis_timestamp': time.time()
                }

        except Exception as e:
            logger.error(f"Failed to get document analysis: {e}")
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
        RETURN e.name AS subject, r.original AS predicate, o.name AS object, chunk_text, type(r) as type

        UNION

        // Incoming relationships
        MATCH (s:Entity)-[r]->(e:Entity)
        WHERE toLower(e.name) = toLower($name)
        OPTIONAL MATCH (s)-[:FROM_CHUNK]->(c1:Chunk)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c2:Chunk)
        WITH s, r, e, coalesce(c1.text, c2.text, "") AS chunk_text
        {predicate_filter_clause}
        RETURN s.name AS subject, r.original AS predicate, e.name AS object, chunk_text, type(r) as type
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
        """Calculate basic statistics about the graph (existing method)."""
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return {"entity_count": 0, "document_count": 0, "chunk_count": 0, "relationship_count": 0}

        logger.info("Calculating graph statistics...")
        query = """
        CALL { MATCH (n:Entity) RETURN count(n) AS entity_count }
        CALL { MATCH (n:Document) RETURN count(n) AS document_count }
        CALL { MATCH (n:Chunk) RETURN count(n) AS chunk_count }
        CALL { MATCH ()-[r]->() RETURN count(r) AS relationship_count }
        RETURN entity_count, document_count, chunk_count, relationship_count
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
                }
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


# Example usage for enhanced OCR integration
if __name__ == "__main__":
    print("--- Enhanced Neo4j Exporter with OCR Integration ---")

    config_path = Path("graph_config.ini")
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        uri = config.get("neo4j", "uri", fallback=None)
        user = config.get("neo4j", "user", fallback=None)
        password = config.get("neo4j", "password", fallback=None)

        if not all([uri, user, password]):
            print("ERROR: Missing Neo4j configuration")
            sys.exit(1)

        print(f"[INFO] Connecting to Neo4j at: {uri}")

        with Neo4jExporter(uri, user, password) as exporter:
            # Example: Store OCR result with metadata
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
                    },
                    {
                        'chunk_id': 'chunk_1',
                        'chunk_text': 'Payment terms...',
                        'word_count': 125,
                        'character_count': 750,
                        'start_word': 125,
                        'end_word': 250
                    }
                ]
            }

            # Store document metadata
            print("[INFO] Storing document metadata...")
            doc_id = sample_ocr_result['file_metadata']['file_id']
            exporter.store_document_metadata(sample_ocr_result, doc_id)

            # Example triples with enhanced metadata
            sample_triples = [
                {
                    'subject': 'ABC Corp',
                    'predicate': 'issues_invoice_to',
                    'object': 'XYZ Ltd',
                    'subject_type': 'companies',
                    'object_type': 'companies',
                    'chunk_id': 'chunk_0',
                    'chunk_text': 'Invoice ABC Corp...',
                    'confidence': 0.95,
                    'inferred': False
                },
                {
                    'subject': 'A-1',
                    'predicate': 'located_in',
                    'object': 'Daman Formation',
                    'subject_type': 'wells',
                    'object_type': 'formations',
                    'chunk_id': 'chunk_1',
                    'chunk_text': 'Payment terms...',
                    'confidence': 0.87,
                    'inferred': False
                }
            ]

            # Store triples with metadata
            print("[INFO] Storing triples with enhanced metadata...")
            success, count = exporter.store_triples_with_metadata(sample_triples, sample_ocr_result)

            if success:
                print(f"[INFO] Successfully stored {count} triples with metadata")

                # Get document analysis
                print("[INFO] Retrieving document analysis...")
                analysis = exporter.get_document_analysis_summary(doc_id)

                if analysis:
                    print("\n--- Document Analysis Summary ---")
                    print(f"Document Type: {analysis['document_metadata'].get('document_type', 'unknown')}")
                    print(f"Quality Score: {analysis['document_metadata'].get('quality_score', 0)}")
                    print(f"Total Entities: {len(analysis['top_entities'])}")
                    print(f"Total Chunks: {analysis['chunk_summary']['total_chunks']}")

                    print("\nTop Entities:")
                    for entity in analysis['top_entities'][:5]:
                        print(f"  - {entity['name']} ({entity['type']}) - {entity['mentions']} mentions")

                # Get updated graph stats
                print("\n--- Updated Graph Statistics ---")
                stats = exporter.get_graph_stats()
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")

            else:
                print("[ERROR] Failed to store triples with metadata")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n--- Enhanced Neo4j Exporter Complete ---")