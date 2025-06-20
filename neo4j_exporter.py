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
logger.setLevel(logging.INFO)  # CHANGED from DEBUG to INFO

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    # SIMPLIFIED: Less verbose formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class Neo4jExporter:
    """
    FIXED: Exports triples to Neo4j with reduced logging verbosity.
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
            # REMOVED: Excessive warning logging
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

    def store_triples(self, triples: List[Dict], store_chunks: bool = True, sanitize: bool = True) -> Tuple[bool, int]:
        """
        FIXED: Store triples with reduced logging verbosity.
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
        """FIXED: Internal function with reduced debug logging."""
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
                # REDUCED: Only log warnings for invalid triples, not every single one
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

            # Construct node patterns
            subj_pattern = f"s:Entity"
            if sanitized_subj_label:
                subj_pattern += f":`{sanitized_subj_label}`"

            obj_pattern = f"o:Entity"
            if sanitized_obj_label:
                obj_pattern += f":`{sanitized_obj_label}`"

            # Optional properties
            inferred = triple.get("inferred", False)
            try:
                confidence = float(triple.get("confidence", 1.0))
            except (ValueError, TypeError):
                confidence = 1.0

            # REMOVED: Excessive debug logging of every triple processing

            # APOC Version - Requires APOC Plugin
            cypher_rel = f"""
            MERGE (s:Entity {{name: $subject}})
            ON CREATE SET s.created_at = timestamp(), s.name = $subject
            WITH s
            CALL apoc.create.addLabels(s, [$subject_label_param]) YIELD node AS subjNode

            MERGE (o:Entity {{name: $object}})
            ON CREATE SET o.created_at = timestamp(), o.name = $object
            WITH subjNode, o
            CALL apoc.create.addLabels(o, [$object_label_param]) YIELD node AS objNode

            MERGE (subjNode)-[r:`{rel_type}`]->(objNode)
            ON CREATE SET
                r.original = $original_predicate,
                r.inferred = $inferred,
                r.confidence = $confidence,
                r.created_at = timestamp()
            ON MATCH SET
                r.confidence = $confidence,
                r.inferred = $inferred,
                r.updated_at = timestamp()
            """

            params = {
                "subject": subject,
                "object": object_,
                "subject_label_param": sanitized_subj_label,
                "object_label_param": sanitized_obj_label,
                "original_predicate": predicate,
                "inferred": inferred,
                "confidence": confidence
            }

            try:
                tx.run(cypher_rel, **params)
                # REMOVED: Debug logging of successful relationships
            except Exception as e:
                logger.error("Failed to run relationship MERGE for triple #%d: %s", i, e)
                skipped_in_batch += 1
                continue

            # Chunk handling
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
                            # REMOVED: Debug logging of chunk merging
                        except Exception as e:
                            logger.error("Failed to MERGE Chunk node with id %s: %s", chunk_id, e)

                    if chunk_id in processed_chunk_ids:
                        try:
                            tx.run(
                                f"""
                                MATCH ({subj_pattern} {{name: $subject}})
                                MATCH ({obj_pattern} {{name: $object}})
                                MATCH (c:Chunk {{id: $chunk_id}})
                                MERGE (s)-[:FROM_CHUNK]->(c)
                                MERGE (o)-[:FROM_CHUNK]->(c)
                                """,
                                chunk_id=chunk_id,
                                subject=subject,
                                object=object_
                            )
                            # REMOVED: Debug logging of chunk linking
                        except Exception as e:
                            logger.error("Failed to link entities to chunk '%s': %s", chunk_id, e)

            processed_in_batch += 1

        # REDUCED: Less verbose transaction completion logging
        if skipped_in_batch > 0:
            logger.warning(f"Transaction processed: {processed_in_batch}, skipped: {skipped_in_batch}")
        else:
            logger.info(f"Transaction processed: {processed_in_batch} triples successfully")

    def get_related_facts_with_context(self, entity_name: str, predicate_filter: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """Retrieve facts connected to a given entity."""
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
        """Calculate basic statistics about the graph."""
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}

        logger.info("Calculating graph statistics...")
        query = """
        CALL { MATCH (n:Entity) RETURN count(n) AS entity_count }
        CALL { MATCH (n:Chunk) RETURN count(n) AS chunk_count }
        CALL { MATCH ()-[r]->() RETURN count(r) AS relationship_count }
        RETURN entity_count, chunk_count, relationship_count
        """

        try:
            records, summary, keys = self.driver.execute_query(query, database_="neo4j")
            record = records[0] if records else None

            if record:
                stats = {
                    "entity_count": record.get("entity_count", 0),
                    "chunk_count": record.get("chunk_count", 0),
                    "relationship_count": record.get("relationship_count", 0),
                }
                logger.info("Graph Stats: %s", stats)
                return stats
            else:
                logger.warning("Graph stats query returned no record")
                return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}
        except Exception as e:
            logger.error("Failed to execute get_graph_stats query: %s", e)
            return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}

    def __enter__(self):
        """Enter the runtime context for using 'with Neo4jExporter(...)'."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and ensure connection closure."""
        self.close()
        return False


def store_document_metadata_in_neo4j(self, ocr_result: Dict, document_id: str):
    """
    Store document-level metadata as nodes in Neo4j.

    Args:
        ocr_result: OCR result with metadata
        document_id: Unique document identifier
    """
    if not self.driver:
        logger.error("Neo4j driver not initialized")
        return False

    try:
        with self.driver.session(database="neo4j") as session:
            # Create Document node with metadata
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.filename = $filename,
                    d.file_type = $file_type,
                    d.file_size = $file_size,
                    d.ocr_method = $ocr_method,
                    d.ocr_confidence = $ocr_confidence,
                    d.processing_time = $processing_time,
                    d.text_length = $text_length,
                    d.document_type = $document_type,
                    d.language = $language,
                    d.page_count = $page_count,
                    d.upload_timestamp = $upload_timestamp,
                    d.processing_timestamp = $processing_timestamp,
                    d.quality_score = $quality_score,
                    d.has_tables = $has_tables,
                    d.word_count = $word_count,
                    d.created_at = timestamp()
            """, {
                'doc_id': document_id,
                'filename': ocr_result.get('original_filename', 'unknown'),
                'file_type': ocr_result.get('file_type', 'unknown'),
                'file_size': ocr_result.get('file_size_bytes', 0),
                'ocr_method': ocr_result.get('method_used', 'unknown'),
                'ocr_confidence': ocr_result.get('confidence', 0.0),
                'processing_time': ocr_result.get('processing_time', 0.0),
                'text_length': len(ocr_result.get('text', '')),
                'document_type': ocr_result.get('document_type', 'unknown'),
                'language': ocr_result.get('language_detected', 'en'),
                'page_count': ocr_result.get('page_count', 1),
                'upload_timestamp': ocr_result.get('upload_timestamp', ''),
                'processing_timestamp': ocr_result.get('extraction_timestamp', ''),
                'quality_score': ocr_result.get('text_quality_score', 0.0),
                'has_tables': ocr_result.get('has_tables', False),
                'word_count': len(ocr_result.get('text', '').split())
            })

            # Link Document to Chunks
            chunk_ids = ocr_result.get('chunk_ids', [])
            for chunk_id in chunk_ids:
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (d)-[:CONTAINS_CHUNK]->(c)
                """, {
                    'doc_id': document_id,
                    'chunk_id': chunk_id
                })

            # Create entity relationships with document context
            detected_entities = ocr_result.get('detected_entities', {})
            for entity_type, entities in detected_entities.items():
                for entity_name in entities:
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MERGE (e:Entity {name: $entity_name})
                        MERGE (d)-[:MENTIONS {entity_type: $entity_type}]->(e)
                    """, {
                        'doc_id': document_id,
                        'entity_name': entity_name,
                        'entity_type': entity_type
                    })

            logger.info(f"Stored document metadata for {document_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to store document metadata: {e}")
        return False


def enhanced_store_triples_with_metadata(self, triples: List[Dict], ocr_metadata: Dict = None):
    """
    Enhanced triple storage that includes OCR metadata and document context.

    Args:
        triples: List of extracted triples
        ocr_metadata: Metadata from OCR processing
    """
    if not triples:
        return True, 0

    try:
        with self.driver.session(database="neo4j") as session:
            with session.begin_transaction() as tx:
                # Store document metadata first if provided
                if ocr_metadata:
                    doc_id = ocr_metadata.get('file_id', 'unknown')
                    self.store_document_metadata_in_neo4j(ocr_metadata, doc_id)

                # Process triples with enhanced metadata
                for triple in triples:
                    subject = triple.get("subject", "").strip()
                    predicate = triple.get("predicate", "").strip()
                    object_ = triple.get("object", "").strip()

                    if not all([subject, predicate, object_]):
                        continue

                    # Enhanced triple storage with metadata
                    rel_type = self.sanitize_predicate(predicate)

                    tx.run("""
                        MERGE (s:Entity {name: $subject})
                        MERGE (o:Entity {name: $object})
                        MERGE (s)-[r:`{rel_type}`]->(o)
                        SET r.original = $original_predicate,
                            r.confidence = $confidence,
                            r.extraction_method = $extraction_method,
                            r.source_document = $source_document,
                            r.chunk_id = $chunk_id,
                            r.created_at = timestamp()
                    """.format(rel_type=rel_type), {
                        'subject': subject,
                        'object': object_,
                        'original_predicate': predicate,
                        'confidence': triple.get('confidence', 1.0),
                        'extraction_method': ocr_metadata.get('method_used', 'unknown') if ocr_metadata else 'unknown',
                        'source_document': ocr_metadata.get('file_id', 'unknown') if ocr_metadata else 'unknown',
                        'chunk_id': triple.get('chunk_id', 'unknown')
                    })

                logger.info(f"Stored {len(triples)} triples with enhanced metadata")
                return True, len(triples)

    except Exception as e:
        logger.error(f"Failed to store enhanced triples: {e}")
        return False, 0


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
                RETURN e.name as entity, count(*) as mentions
                ORDER BY mentions DESC
                LIMIT 20
            """, doc_id=document_id)

            entities = [{'name': record['entity'], 'mentions': record['mentions']}
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

            return {
                'document_metadata': doc_data,
                'top_entities': entities,
                'relationship_summary': relationships,
                'analysis_timestamp': time.time()
            }

    except Exception as e:
        logger.error(f"Failed to get document analysis: {e}")
        return {}


# Example usage in main block with reduced logging
if __name__ == "__main__":
    print("--- Neo4j Exporter Script Start ---")

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
        file_path = None

        if "data" in config and "json_file" in config["data"]:
            file_path_str = config.get("data", "json_file", fallback=None)
            if file_path_str:
                file_path = Path(file_path_str)

        if not all([uri, user, password, file_path]):
            missing = [
                "neo4j.uri" if not uri else None,
                "neo4j.user" if not user else None,
                "neo4j.password" if not password else None,
                "data.json_file" if not file_path else None,
            ]
            missing_str = ", ".join(filter(None, missing))
            print(f"ERROR: Missing required configuration values: {missing_str}")
            sys.exit(1)

        print(f"[INFO] Neo4j URI: {uri}")
        print(f"[INFO] Input JSON file path: {file_path}")

        if file_path.exists():
            print(f"[INFO] Loading triples from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    triples_data = json.load(f)

                if not isinstance(triples_data, list):
                    print(f"ERROR: Expected a JSON list in {file_path}")
                    sys.exit(1)

                print(f"[INFO] Loaded {len(triples_data)} potential triples from JSON")

                # Add sample entity types for testing
                for i, t in enumerate(triples_data):
                    if i % 4 == 0:
                        t['subject_type'] = 'Well'
                        t['object_type'] = 'Formation'
                    elif i % 4 == 1:
                        t['subject_type'] = 'Company'
                        t['object_type'] = 'Well'
                    elif i % 4 == 2:
                        t['subject_type'] = 'Formation'
                        t['object_type'] = 'Concept'
                    else:
                        t['subject_type'] = 'Concept'
                        t['object_type'] = 'Location'

                    t.setdefault('subject_type', None)
                    t.setdefault('object_type', None)

            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to decode JSON from {file_path}: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"ERROR: Failed to read file {file_path}: {e}")
                sys.exit(1)

            try:
                print("[INFO] Initializing Neo4jExporter...")
                with Neo4jExporter(uri, user, password) as exporter:
                    print("[INFO] Storing triples in Neo4j...")
                    exporter.store_triples(triples_data)

                    print("\n--- Post-Import Verification ---")
                    entity_to_check = "henry"
                    print(f"\nFetching related facts for entity: '{entity_to_check}'...")

                    related_facts_all = exporter.get_related_facts_with_context(entity_to_check)
                    related_facts_inspired = exporter.get_related_facts_with_context(entity_to_check,
                                                                                     predicate_filter="inspired by")

                    if related_facts_all:
                        print(
                            f"Found {len(related_facts_all)} total facts related to '{entity_to_check}' (showing max 10):")
                        for i, fact in enumerate(related_facts_all[:10]):
                            print(
                                f"  - {fact.get('subject')} -[{fact.get('predicate')}]-> {fact.get('object')} (Type: {fact.get('type')})")
                        if len(related_facts_all) > 10:
                            print("  ...")
                    else:
                        print(f"No related facts found for '{entity_to_check}'")

                    if related_facts_inspired:
                        print(
                            f"\nFound {len(related_facts_inspired)} facts for '{entity_to_check}' matching filter 'inspired by':")
                        for fact in related_facts_inspired:
                            print(
                                f"  - {fact.get('subject')} -[{fact.get('predicate')}]-> {fact.get('object')} (Type: {fact.get('type')})")
                    else:
                        print(f"\nNo related facts found for '{entity_to_check}' matching filter 'inspired by'")

                    print("\nFetching graph statistics...")
                    stats = exporter.get_graph_stats()
                    print("\nGraph Summary:")
                    print(f"  Entities (:Entity): {stats.get('entity_count', 'N/A')}")
                    print(f"  Chunks (:Chunk):   {stats.get('chunk_count', 'N/A')}")
                    print(f"  Relationships:     {stats.get('relationship_count', 'N/A')}")

            except ConnectionError as e:
                print(f"\nERROR: Could not connect to Neo4j. Please check URI, credentials, and database status")
                sys.exit(1)

        else:
            print(f"ERROR: Input JSON file not found at {file_path}")
            sys.exit(1)

    except configparser.Error as e:
        print(f"ERROR reading configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

    print("\n--- Neo4j Exporter Script End ---")