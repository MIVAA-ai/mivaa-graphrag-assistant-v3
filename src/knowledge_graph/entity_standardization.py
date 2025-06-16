"""Entity standardization and relationship inference for knowledge graphs."""
import logging # Import logging
import re
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple # Added Tuple

# Setup logger (ensure handlers aren't added multiple times if imported)
logger = logging.getLogger(__name__)
# Set default level - can be overridden by external config
logger.setLevel(logging.DEBUG) # Set to DEBUG to capture detailed logs

if not logger.handlers:
    import sys # Import sys only if needed
    handler = logging.StreamHandler(sys.stdout) # Log to standard output
    # More detailed formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent duplicate logs if root logger has handlers


# Assuming these imports are available in the project structure
try:
    # Assuming llm.py is in src.knowledge_graph
    from src.knowledge_graph.llm import call_llm, extract_json_from_text # Added extract_json_from_text
    # Assuming prompts.py is in src.knowledge_graph
    from src.knowledge_graph.prompts import (
        ENTITY_RESOLUTION_SYSTEM_PROMPT,
        get_entity_resolution_user_prompt,
        RELATIONSHIP_INFERENCE_SYSTEM_PROMPT,
        get_relationship_inference_user_prompt,
        WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT,
        get_within_community_inference_user_prompt
    )
    # Test LLM availability
    llm_features_available = True
except ImportError as e:
     logger.error(f"Failed to import LLM/prompt modules in entity_standardization: {e}. LLM-based features might fail.")
     llm_features_available = False
     # Define dummy functions if imports fail to prevent NameErrors,
     # but log a critical warning.
     def call_llm(*args, **kwargs):
         logger.error("call_llm function not available due to import error.")
         raise NotImplementedError("call_llm is not available.")
     def extract_json_from_text(*args, **kwargs):
         logger.error("extract_json_from_text function not available due to import error.")
         return None
     # Define dummy prompts if needed, or let it fail if prompts are essential
     ENTITY_RESOLUTION_SYSTEM_PROMPT = "ENTITY_RESOLUTION_SYSTEM_PROMPT_MISSING"
     RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = "RELATIONSHIP_INFERENCE_SYSTEM_PROMPT_MISSING"
     WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = "WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT_MISSING"
     def get_entity_resolution_user_prompt(*args): return "get_entity_resolution_user_prompt_MISSING"
     def get_relationship_inference_user_prompt(*args): return "get_relationship_inference_user_prompt_MISSING"
     def get_within_community_inference_user_prompt(*args): return "get_within_community_inference_user_prompt_MISSING"


# Define allowed cross-type relationships
ALLOWED_CROSS_TYPE_RELATIONSHIPS = {
    # Invoice relationships
    ('Invoice', 'Date'): True,
    ('Invoice', 'Company'): True,
    ('Invoice', 'Currency'): True,
    ('Invoice', 'Value'): True,

    # Activity relationships
    ('Activity', 'Date'): True,
    ('Activity', 'Equipment'): True,
    ('Activity', 'Asset'): True,
    ('Activity', 'Person'): True,
    ('Activity', 'Material'): True,

    # Equipment relationships
    ('Equipment', 'Activity'): True,
    ('Equipment', 'Asset'): True,
    ('Equipment', 'Location'): True,

    # Asset relationships
    ('Asset', 'Activity'): True,
    ('Asset', 'Location'): True,
    ('Asset', 'Material'): True,

    # Company relationships
    ('Company', 'Location'): True,
    ('Company', 'Invoice'): True,
    ('Company', 'Contract'): True,

    # Task relationships
    ('Task', 'Date'): True,
    ('Task', 'Equipment'): True,
    ('Task', 'Description'): True,
    ('Task', 'Value'): True,

    # Location relationships
    ('Location', 'Facility'): True,
    ('Location', 'Contract'): True,

    # Facility relationships
    ('Facility', 'Identifier'): True
}

def is_cross_type_relationship_allowed(type1: str, type2: str) -> bool:
    """Check if a cross-type relationship between two entity types is allowed"""
    if not type1 or not type2:
        return True  # Allow if types are unknown

    if type1 == type2:
        return True  # Same type is always allowed

    # Check both directions
    return (ALLOWED_CROSS_TYPE_RELATIONSHIPS.get((type1, type2), False) or
            ALLOWED_CROSS_TYPE_RELATIONSHIPS.get((type2, type1), False))

def check_entity_compatibility(entity1_orig: str, entity2_orig: str,
                               entity_details: dict) -> bool:
    """Check if two entities can be merged/related"""

    e1_lower = entity1_orig.lower()
    e2_lower = entity2_orig.lower()

    types1 = entity_details.get(e1_lower, {}).get("types", set())
    types2 = entity_details.get(e2_lower, {}).get("types", set())

    # If either entity has no type info, allow the relationship
    if not types1 or not types2:
        return True

    # Check if any combination of types is allowed
    for t1 in types1:
        for t2 in types2:
            if is_cross_type_relationship_allowed(t1, t2):
                return True

    # Log the rejection for debugging (but less verbose than before)
    logger.debug(f"Rejecting merge between '{entity1_orig}' ({types1}) and '{entity2_orig}' ({types2}) - no valid cross-type relationship")
    return False


def limit_predicate_length(predicate, max_words=3):
    """
    Enforce a maximum word limit on predicates.

    Args:
        predicate: The original predicate string
        max_words: Maximum number of words allowed (default: 3)

    Returns:
        Shortened predicate with no more than max_words
    """
    if not isinstance(predicate, str): predicate = str(predicate) # Handle non-string input
    words = predicate.split()
    if len(words) <= max_words:
        return predicate

    # If too long, use only the first max_words words
    shortened = ' '.join(words[:max_words])

    # Remove trailing prepositions or articles if they're the last word
    stop_words = {'a', 'an', 'the', 'of', 'with', 'by', 'to', 'from', 'in', 'on', 'for'}
    last_word = shortened.split()[-1].lower() if shortened else ""
    if last_word in stop_words and len(words) > 1:
        shortened = ' '.join(shortened.split()[:-1])

    return shortened

# <<< ADDED: Synonym Application Helper >>>
def apply_synonyms(name: Any, synonym_map: Dict[str, str]) -> Any:
    """
    Applies predefined synonyms (loaded from config) to an entity name.
    Looks up the lowercase version and returns the canonical name if found.

    Args:
        name: The entity name (string or other type).
        synonym_map: Dictionary mapping lowercase variants to canonical names.

    Returns:
        The canonical name (preserving its casing) if a synonym match is found,
        otherwise returns the original name.
    """
    if not isinstance(name, str) or not name.strip():
        return name # Return original if not a non-empty string

    lower_name = name.lower().strip()
    canonical_name = synonym_map.get(lower_name)

    if canonical_name:
        # Only log if a change actually occurred
        if canonical_name != name:
             logger.debug(f"Applied synonym: '{name}' -> '{canonical_name}'")
        return canonical_name
    else:
        return name # Return original name if no synonym found
# <<< END: Synonym Application Helper ---

# --- Main Standardization Function ---

def standardize_entities(triples: List[Dict], config: Dict) -> List[Dict]:
    """
    Standardize entity names across all triples, preserving chunk information.
    Applies synonyms from config, then uses rule-based grouping.
    Prioritizes longer, more complete names within groups of similar entities.

    Args:
        triples: List of dictionaries with 'subject', 'predicate', 'object',
                 'chunk_id', and 'chunk_text' keys. Entity types are optional
                 but will be preserved if present.
        config: Configuration dictionary, expected to contain standardization settings
                including the optional [standardization][synonyms] table.

    Returns:
        List of triples with standardized entity names and preserved chunk info.
    """
    if not triples:
        logger.warning("standardize_entities received empty list, returning empty.")
        return []

    logger.info(f"Starting entity standardization for {len(triples)} input triples...")

    # <<< Configuration Extraction for Synonyms >>>
    standardization_config = config.get("standardization", {})
    synonym_map = standardization_config.get("synonyms", {})
    if not isinstance(synonym_map, dict):
        logger.warning("Synonyms in config is not a dictionary/table. Disabling synonym mapping.")
        synonym_map = {}
    else:
        # Ensure keys are lowercase and values are strings for reliable lookup
        synonym_map = {str(k).lower(): str(v) for k, v in synonym_map.items() if k and v}
        if synonym_map:
             logger.info(f"Loaded {len(synonym_map)} entity synonyms from configuration.")
        else:
             logger.info("No valid entity synonyms found in configuration.")
    # <<< END: Configuration Extraction >>>

    # --- Initial Validation & Synonym Application ---
    valid_triples = [] # Keep track of originally valid triples if needed
    invalid_count = 0
    triples_with_synonyms = [] # Store triples after synonym step

    for i, triple in enumerate(triples):
        # Basic structure and content check
        if not (isinstance(triple, dict) and
                all(k in triple for k in ["subject", "predicate", "object"]) and
                isinstance(triple["subject"], str) and triple["subject"].strip() and
                isinstance(triple["predicate"], str) and triple["predicate"].strip() and
                isinstance(triple["object"], str) and triple["object"].strip()):
            logger.warning(f"Skipping invalid or incomplete triple #{i} (empty S/P/O): {triple}")
            invalid_count += 1
            continue

        # Ensure metadata keys exist
        triple.setdefault('chunk_id', None)
        triple.setdefault('chunk_text', "")
        triple.setdefault('subject_type', None)
        triple.setdefault('object_type', None)
        valid_triples.append(triple) # Add to original valid list

        # --- Apply Synonyms Here ---
        if synonym_map:
            mod_triple = triple.copy() # Work on a copy
            mod_triple["subject"] = apply_synonyms(triple["subject"], synonym_map)
            mod_triple["object"] = apply_synonyms(triple["object"], synonym_map)
            triples_with_synonyms.append(mod_triple)
        else:
            # If no synonyms, the list is the same as valid_triples
            triples_with_synonyms.append(triple)
        # --- End Synonym Application ---

    if invalid_count > 0: logger.warning(f"Filtered out {invalid_count} invalid/incomplete triples during initial validation.")
    if not triples_with_synonyms: logger.error("No valid triples remaining after initial validation and synonym step."); return []

    # --- The rest of the logic now operates on 'triples_with_synonyms' ---

    # 1. Extract all unique entities (lowercase for mapping) and store original casings + types
    #    Collect details based on the names *after* synonym application
    entity_details = defaultdict(lambda: {"casings": set(), "types": set(), "count": 0})
    for triple in triples_with_synonyms: # Use the list processed by synonyms
        subj_lower = triple["subject"].lower()
        obj_lower = triple["object"].lower()

        entity_details[subj_lower]["casings"].add(triple["subject"]) # Store casing after synonym step
        entity_details[subj_lower]["count"] += 1
        if triple.get("subject_type"):
            entity_details[subj_lower]["types"].add(triple["subject_type"])

        entity_details[obj_lower]["casings"].add(triple["object"]) # Store casing after synonym step
        entity_details[obj_lower]["count"] += 1
        if triple.get("object_type"):
            entity_details[obj_lower]["types"].add(triple["object_type"])

    all_entities_lower = set(entity_details.keys())
    logger.debug(f"Collected details for {len(all_entities_lower)} unique lowercase entities post-synonym step.")

    # 2. Group similar entities - first by exact match after lowercasing and removing stopwords
    standardized_entities_map = {} # Map lowercase variant -> standard form (original casing)
    entity_groups = defaultdict(list)

    # Helper function to normalize text for comparison
    def normalize_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "as", "ltd", "inc", "corp", "limited", "llc", "plc"}
        text = re.sub(r'[^\w\s#-]', '', text) # Keep alphanumeric, whitespace, hash, hyphen
        words = [word for word in text.split() if word not in stopwords]
        return " ".join(words)

    # Process entities (longer first might still be beneficial)
    sorted_entities = sorted(list(all_entities_lower), key=lambda x: (-len(x), x))

    # First pass: Standard normalization grouping
    for entity_lower in sorted_entities:
        normalized = normalize_text(entity_lower)
        if normalized:
            entity_groups[normalized].append(entity_lower) # Group lowercase variants

    # 3. For each group, choose the most representative name (longest, then most frequent)
    for group_key, variants_lower in entity_groups.items():
        if len(variants_lower) == 1:
            lower_variant = variants_lower[0]
            # Find the longest original casing for this single variant
            casings = entity_details[lower_variant]["casings"]
            original_casing = max(casings, key=len) if casings else lower_variant
            standardized_entities_map[lower_variant] = original_casing
        else:
            # Multiple variants, choose the LONGEST first, then most frequent as tie-breaker
            # Use length of representative casing for sorting
            def get_rep_casing_len(variant_low):
                 casings = entity_details[variant_low]["casings"]
                 return max(len(c) for c in casings) if casings else 0

            standard_form_lower = sorted(variants_lower, key=lambda x: (-get_rep_casing_len(x), -entity_details[x]["count"]))[0]

            # Get the longest original casing for the chosen standard form
            casings = entity_details[standard_form_lower]["casings"]
            standard_form_original = max(casings, key=len) if casings else standard_form_lower

            if len(variants_lower) > 1: # Log only if grouping occurred
                logger.debug(f"Rule-based Group '{group_key}': Variants {variants_lower} -> Standardized to '{standard_form_original}'")

            # Map all variants in this group to the chosen standard form
            for variant_lower in variants_lower:
                standardized_entities_map[variant_lower] = standard_form_original

    # 4. Second pass: check for root word/subset relationships (Optional, using existing logic)
    additional_standardizations = {}
    standard_forms_original = set(standardized_entities_map.values())
    sorted_standards_original = sorted(list(standard_forms_original), key=len)

    for i, entity1_orig in enumerate(sorted_standards_original):
        e1_lower = entity1_orig.lower()
        e1_words = set(e1_lower.split())

        for entity2_orig in sorted_standards_original[i + 1:]:
            e2_lower = entity2_orig.lower()
            if e1_lower == e2_lower: continue

            # FIXED: Use the new compatibility check instead of strict type matching
            if not check_entity_compatibility(entity1_orig, entity2_orig, entity_details):
                continue

            e2_words = set(e2_lower.split())
            # Check for subset relationship
            if e1_words.issubset(e2_words) and len(e1_words) > 0:
                additional_standardizations[e1_lower] = entity2_orig
            elif e2_words.issubset(e1_words) and len(e2_words) > 0:
                additional_standardizations[e2_lower] = entity1_orig
            else:
                # Check for stem similarity (existing logic)
                stems1 = {word[:4] for word in e1_words if len(word) > 4}
                stems2 = {word[:4] for word in e2_words if len(word) > 4}
                if stems1 and stems2:
                    shared_stems = stems1.intersection(stems2)
                    min_stems = min(len(stems1), len(stems2))
                    if min_stems > 0 and shared_stems and (len(shared_stems) / min_stems) > 0.5:
                        if len(entity1_orig) >= len(entity2_orig):
                            additional_standardizations[e2_lower] = entity1_orig
                        else:
                            additional_standardizations[e1_lower] = entity2_orig

    # Apply additional standardizations to the map (resolve chains)
    final_map = standardized_entities_map.copy()
    changed = True
    while changed:
        changed = False
        for entity_lower, standard_orig in list(final_map.items()):
            standard_lower = standard_orig.lower()
            if standard_lower in additional_standardizations:
                new_standard = additional_standardizations[standard_lower]
                if final_map[entity_lower] != new_standard:
                    final_map[entity_lower] = new_standard
                    changed = True

    # 5. Apply final standardization map to all triples, preserving original chunk info & types
    standardized_triples_final = []
    processed_keys_log = set() # For logging map application just once per variant
    for triple in triples_with_synonyms: # Use the list that already had synonyms applied
        subj_lower_syn = triple["subject"].lower() # Lowercase name *after* synonym step
        obj_lower_syn = triple["object"].lower()  # Lowercase name *after* synonym step

        # Get the final standard names using the map, default to the post-synonym name
        final_subj = final_map.get(subj_lower_syn, triple["subject"])
        final_obj = final_map.get(obj_lower_syn, triple["object"])

        # Log the mapping action only once per unique lowercase variant
        if subj_lower_syn not in processed_keys_log and subj_lower_syn in final_map and final_map[subj_lower_syn] != triple["subject"]:
            logger.debug(f"Final Map Applied: Subject '{triple['subject']}' (lower: {subj_lower_syn}) -> '{final_subj}'")
            processed_keys_log.add(subj_lower_syn)
        if obj_lower_syn not in processed_keys_log and obj_lower_syn in final_map and final_map[obj_lower_syn] != triple["object"]:
            logger.debug(f"Final Map Applied: Object '{triple['object']}' (lower: {obj_lower_syn}) -> '{final_obj}'")
            processed_keys_log.add(obj_lower_syn)

        # --- Get consolidated types for the final canonical entities ---
        final_subj_lower_canon = final_subj.lower()
        final_obj_lower_canon = final_obj.lower()

        # Gather all types associated with any variant mapping to the final subject/object
        all_subj_types = set()
        for variant_lower, standard_name in final_map.items():
            if standard_name == final_subj:
                all_subj_types.update(entity_details.get(variant_lower, {}).get("types", set()))
        all_subj_types.update(entity_details.get(final_subj_lower_canon, {}).get("types", set()))
        all_subj_types = {t for t in all_subj_types if t} # Filter out None

        all_obj_types = set()
        for variant_lower, standard_name in final_map.items():
             if standard_name == final_obj:
                 all_obj_types.update(entity_details.get(variant_lower, {}).get("types", set()))
        all_obj_types.update(entity_details.get(final_obj_lower_canon, {}).get("types", set()))
        all_obj_types = {t for t in all_obj_types if t} # Filter out None

        # Choose a primary type (e.g., first non-None type found, or original as fallback)
        subj_type = next(iter(all_subj_types), triple.get("subject_type"))
        obj_type = next(iter(all_obj_types), triple.get("object_type"))

        # Create the new triple dictionary
        new_triple = {
            "subject": final_subj,
            "subject_type": subj_type, # Use consolidated/fallback type
            "predicate": limit_predicate_length(triple["predicate"]), # Apply length limit
            "object": final_obj,
            "object_type": obj_type, # Use consolidated/fallback type
            "chunk_id": triple.get("chunk_id"), # Preserve original chunk info
            "chunk_text": triple.get("chunk_text", ""), # Preserve original chunk info
            # Preserve other potential fields like 'inferred', 'confidence'
            **{k: v for k, v in triple.items() if k not in {'subject', 'subject_type', 'predicate', 'object', 'object_type', 'chunk_id', 'chunk_text'}}
        }
        standardized_triples_final.append(new_triple)

    # 6. Optional: LLM-based resolution (ensure it preserves chunk info & types if used)
    # FIXED: Use local availability check instead of global variable
    if standardization_config.get("use_llm_for_entities", False):
        if llm_features_available:
            logger.info("Attempting LLM-based entity resolution after rule-based standardization...")
            # Ensure _resolve_entities_with_llm preserves metadata correctly
            standardized_triples_final = _resolve_entities_with_llm(standardized_triples_final, config)
        else:
             logger.warning("LLM entity resolution requested in config but LLM features are not available. Skipping.")

    # 7. Filter out self-referencing triples
    filtered_triples = [triple for triple in standardized_triples_final if triple["subject"] != triple["object"]]
    removed_count = len(standardized_triples_final) - len(filtered_triples)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} self-referencing triples after standardization.")

    logger.info(f"Entity standardization finished. Final triple count: {len(filtered_triples)}")
    if filtered_triples:
        logger.debug(f"Example final standardized triple: {filtered_triples[0]}")

    return filtered_triples

# --- Rest of the functions (_infer_relationships, _identify_communities, etc.) remain the same ---
# --- Ensure the temperature fallback fix is present in LLM calls within inference functions ---

def infer_relationships(triples: List[Dict], config: Dict) -> List[Dict]:
    """
    Infer additional relationships between entities, preserving chunk info for original triples.

    Args:
        triples: List of dictionaries with standardized entity names and chunk info.
        config: Configuration dictionary.

    Returns:
        List of triples (original + inferred) with chunk info preserved where applicable.
    """
    if not triples or len(triples) < 2:
        return triples

    logger.info("Inferring additional relationships between entities...")

    # Keep track of original triples to ensure their chunk info is preserved
    original_triples = triples # Keep reference to the input list

    # Create a graph representation for easier traversal
    graph = defaultdict(set)
    all_entities = set()
    for triple in original_triples:
        subj = triple["subject"]
        obj = triple["object"]
        graph[subj].add(obj)
        all_entities.add(subj)
        all_entities.add(obj)

    # Find disconnected communities
    communities = _identify_communities(graph)
    logger.info(f"Identified {len(communities)} disconnected communities in the graph")

    newly_inferred_triples = [] # Store only the newly created triples

    # --- Performance Optimization: Check config flag for LLM inference ---
    run_llm_inference = config.get("inference", {}).get("use_llm_for_inference", True) # Default to True if not specified
    if run_llm_inference:
        logger.info("LLM-based relationship inference is ENABLED.")
        # Infer relationships between different communities
        community_triples = _infer_relationships_with_llm(original_triples, communities, config)
        if community_triples:
            newly_inferred_triples.extend(community_triples)

        # Infer relationships within the same communities for semantically related entities
        within_community_triples = _infer_within_community_relationships(original_triples, communities, config)
        if within_community_triples:
            newly_inferred_triples.extend(within_community_triples)
    else:
        logger.info("LLM-based relationship inference is DISABLED via configuration.")


    # Apply transitive inference rules (Non-LLM based)
    transitive_triples = _apply_transitive_inference(original_triples, graph)
    if transitive_triples:
        newly_inferred_triples.extend(transitive_triples)

    # Infer relationships based on lexical similarity (Non-LLM based)
    lexical_triples = _infer_relationships_by_lexical_similarity(all_entities, original_triples)
    if lexical_triples:
        newly_inferred_triples.extend(lexical_triples)

    # Combine original and newly inferred triples
    combined_triples = original_triples + newly_inferred_triples
    logger.info(f"Total triples before deduplication: {len(combined_triples)} ({len(original_triples)} original, {len(newly_inferred_triples)} inferred)")


    # De-duplicate triples, prioritizing originals (which have chunk info)
    unique_triples = _deduplicate_triples(combined_triples)
    logger.info(f"Total triples after deduplication: {len(unique_triples)}")


    # Final pass: ensure all predicates follow the 3-word limit
    # and ensure inferred triples have default chunk info if missing
    final_processed_triples = []
    for triple in unique_triples:
        triple["predicate"] = limit_predicate_length(triple["predicate"])
        # Ensure chunk keys exist, especially for inferred triples
        triple.setdefault('chunk_id', None)
        triple.setdefault('chunk_text', "")
        # Ensure type keys exist (might be None if not extracted/inferred)
        triple.setdefault('subject_type', None)
        triple.setdefault('object_type', None)
        final_processed_triples.append(triple)


    # Filter out self-referencing triples
    filtered_triples = [triple for triple in final_processed_triples if triple["subject"] != triple["object"]]
    removed_count = len(final_processed_triples) - len(filtered_triples)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} self-referencing triples after inference.")

    inferred_count = sum(1 for t in filtered_triples if t.get("inferred"))
    logger.info(f"Inference complete. Final triple count: {len(filtered_triples)} ({inferred_count} inferred).")
    # Log an example to verify chunk info presence/absence
    if filtered_triples:
        logger.debug(f"Example final triple after inference: {filtered_triples[0]}")

    return filtered_triples

def _identify_communities(graph):
    """
    Identify disconnected communities in the graph.

    Args:
        graph: Dictionary representing the graph structure {node: {neighbor1, neighbor2}}

    Returns:
        List of sets, where each set contains nodes in a community
    """
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    visited = set()
    communities = []

    # Build an undirected representation for traversal
    undirected_graph = defaultdict(set)
    for u, neighbors in graph.items():
        for v in neighbors:
            undirected_graph[u].add(v)
            undirected_graph[v].add(u)
    # Add nodes that might only appear as objects
    for node in all_nodes:
        if node not in undirected_graph:
            undirected_graph[node] = set()


    def bfs(start_node, current_community):
        """Perform BFS to find connected component."""
        queue = [start_node]
        visited.add(start_node)
        current_community.add(start_node)
        while queue:
            node = queue.pop(0)
            for neighbor in undirected_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_community.add(neighbor)
                    queue.append(neighbor)

    for node in all_nodes:
        if node not in visited:
            community = set()
            bfs(node, community)
            if community: # Ensure we don't add empty sets if logic error occurs
                 communities.append(community)

    return communities

def _apply_transitive_inference(triples: List[Dict], graph: Dict[Any, Set]) -> List[Dict]:
    """
    Apply transitive inference to find new relationships. A->B, B->C => A->C.
    Newly inferred triples will have chunk_id=None, chunk_text="".

    Args:
        triples: List of triple dictionaries (used to get original predicates).
        graph: Dictionary representing the graph structure {node: {neighbor1, neighbor2}}.

    Returns:
        List of new inferred triples.
    """
    new_triples = []
    # Create a map for quick predicate lookup
    predicates = {}
    for triple in triples:
        key = (triple["subject"], triple["object"])
        # Store predicate and original chunk info if available
        predicates[key] = {
            "predicate": triple["predicate"],
            "chunk_id": triple.get("chunk_id"),
            "chunk_text": triple.get("chunk_text", "")
            # Store types if available
            # "subject_type": triple.get("subject_type"),
            # "object_type": triple.get("object_type")
        }

    logger.info("Applying transitive inference...")
    count = 0
    for subj in list(graph.keys()): # Iterate over a copy of keys
        if subj not in graph: continue # Skip if node was only an object
        for mid in list(graph[subj]): # Iterate over a copy of neighbors
            if mid not in graph: continue # Skip if intermediate node has no outgoing edges
            for obj in list(graph[mid]): # Iterate over a copy of neighbors
                # Check conditions: A != C and A -> C doesn't already exist
                if subj != obj and (subj, obj) not in predicates:
                    # Get original predicates
                    pred1_info = predicates.get((subj, mid))
                    pred2_info = predicates.get((mid, obj))

                    # Proceed only if both intermediate predicates are found
                    if pred1_info and pred2_info:
                        pred1 = pred1_info["predicate"]
                        pred2 = pred2_info["predicate"]
                        # Generate a new predicate
                        new_pred = f"{pred1} via {mid} leading to {pred2}" # More descriptive
                        new_triple = {
                            "subject": subj,
                            "predicate": limit_predicate_length(new_pred),
                            "object": obj,
                            "inferred": True,
                            "confidence": 0.7, # Assign lower confidence
                            "chunk_id": None, # No single source chunk
                            "chunk_text": "",   # No single source chunk text
                            # Types would need to be inferred or inherited if needed
                            "subject_type": None, # Or look up subj type
                            "object_type": None   # Or look up obj type
                        }
                        # Avoid adding duplicates within this inference step
                        if (subj, new_triple["predicate"], obj) not in {(t["subject"], t["predicate"], t["object"]) for t in new_triples}:
                            new_triples.append(new_triple)
                            count += 1

    logger.info(f"Inferred {count} relationships via transitive inference.")
    return new_triples

def _deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    """
    Remove duplicate triples based on (subject, predicate, object).
    Prioritizes keeping non-inferred triples over inferred ones.
    Preserves chunk_id and chunk_text from the kept triple.

    Args:
        triples: List of triple dictionaries.

    Returns:
        List of unique triples.
    """
    unique_triples_dict = {} # Key: (subject, predicate, object), Value: triple dict

    for triple in triples:
        # Ensure required keys exist before creating the tuple key
        subj = triple.get("subject")
        pred = triple.get("predicate")
        obj = triple.get("object")

        if subj is None or pred is None or obj is None:
            logger.warning(f"Skipping triple in deduplication due to missing S/P/O: {triple}")
            continue

        key = (subj, pred, obj)
        is_inferred = triple.get("inferred", False) # Default to False if key missing

        if key not in unique_triples_dict:
            # First time seeing this triple, add it
            unique_triples_dict[key] = triple
        else:
            # Duplicate found, decide which one to keep
            existing_triple = unique_triples_dict[key]
            existing_is_inferred = existing_triple.get("inferred", False)

            # If the current triple is NOT inferred and the existing one IS, replace it
            if not is_inferred and existing_is_inferred:
                unique_triples_dict[key] = triple
            # Otherwise, keep the existing one (either both inferred, both original, or existing is original)

    final_list = list(unique_triples_dict.values())
    logger.debug(f"Deduplicated {len(triples)} triples down to {len(final_list)}.")
    return final_list

def _resolve_entities_with_llm(triples, config):
    """
    Use LLM to help resolve entity references and standardize entity names.
    Preserves chunk_id and chunk_text. Needs update to preserve types.

    Args:
        triples: List of triples with potentially non-standardized entities
        config: Configuration dictionary

    Returns:
        List of triples with LLM-assisted entity standardization
    """
    logger.info("Attempting LLM-based entity resolution...") # Changed from warning
    # Extract all unique entities
    all_entities = set()
    for triple in triples:
        all_entities.add(triple["subject"])
        all_entities.add(triple["object"])

    # Limit entity count if necessary
    entity_limit = config.get("standardization", {}).get("llm_entity_limit", 100) # Configurable limit
    if len(all_entities) > entity_limit:
        entity_counts = defaultdict(int)
        for triple in triples:
            entity_counts[triple["subject"]] += 1
            entity_counts[triple["object"]] += 1
        all_entities = {entity for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:entity_limit]}
        logger.info(f"Limiting LLM entity resolution to top {entity_limit} entities.")

    entity_list = "\n".join(sorted(all_entities))
    system_prompt = ENTITY_RESOLUTION_SYSTEM_PROMPT
    user_prompt = get_entity_resolution_user_prompt(entity_list)

    try:
        # LLM configuration
        llm_config = config.get("llm", {}) # Get LLM sub-config
        model = llm_config.get("model")
        api_key = llm_config.get("api_key")
        max_tokens = llm_config.get("max_tokens_entity_resolution", 1024) # Specific max tokens
        temperature = llm_config.get("temperature_entity_resolution", 0.1) # Specific temp
        base_url = llm_config.get("base_url")

        if not model or not api_key:
             raise ValueError("LLM model or API key missing in config for entity resolution.")

        # Call LLM
        logger.debug(f"Calling LLM for entity resolution. Temp: {temperature}")
        response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                            system_prompt=system_prompt, max_tokens=max_tokens,
                            temperature=temperature, base_url=base_url)

        entity_mapping = extract_json_from_text(response)

        if entity_mapping and isinstance(entity_mapping, dict):
            logger.info(f"Applying LLM-based entity standardization for {len(entity_mapping)} entity groups.")
            entity_to_standard = {}
            for standard, variants in entity_mapping.items():
                for variant in variants:
                    entity_to_standard[variant] = standard
                entity_to_standard[standard] = standard # Map standard to itself

            # --- Apply mapping while preserving chunk info & types ---
            resolved_triples = []
            for triple in triples:
                new_triple = triple.copy() # Create a copy to modify
                new_triple["subject"] = entity_to_standard.get(triple["subject"], triple["subject"])
                new_triple["object"] = entity_to_standard.get(triple["object"], triple["object"])
                # chunk_id, chunk_text, subject_type, object_type are already in the copied dict
                resolved_triples.append(new_triple)
            return resolved_triples # Return the modified list
        else:
            logger.warning("Could not extract valid entity mapping from LLM response for entity resolution.")
            return triples # Return original triples if LLM fails

    except Exception as e:
        logger.error(f"Error in LLM-based entity resolution: {e}", exc_info=True)
        return triples # Return original triples on error

def _infer_relationships_with_llm(triples, communities, config):
    """
    Use LLM to infer relationships between disconnected communities.
    Inferred triples will have chunk_id=None, chunk_text="", types=None.

    Args:
        triples: List of existing triples
        communities: List of community sets
        config: Configuration dictionary

    Returns:
        List of new inferred triples
    """
    if len(communities) <= 1:
        logger.info("Only one community found, skipping LLM inference between communities.")
        return []

    # --- Performance Optimization: Limit number of community pairs processed ---
    max_community_pairs_to_process = config.get("inference", {}).get("llm_max_community_pairs", 5) # Default to 5 pairs
    num_communities_to_consider = config.get("inference", {}).get("llm_communities_to_consider", 3) # Default to top 3

    large_communities = sorted(communities, key=len, reverse=True)[:num_communities_to_consider]
    newly_inferred_triples = []
    processed_pairs = set()
    pairs_processed_count = 0

    logger.info(f"Attempting LLM inference between up to {max_community_pairs_to_process} pairs from the {len(large_communities)} largest communities.")

    for i, comm1 in enumerate(large_communities):
        for j, comm2 in enumerate(large_communities):
            if i >= j: continue

            # --- Stop if max pairs processed ---
            if pairs_processed_count >= max_community_pairs_to_process:
                logger.info(f"Reached limit ({max_community_pairs_to_process}) of community pairs for LLM inference.")
                break

            pair_key = tuple(sorted((i, j)))
            if pair_key in processed_pairs: continue
            processed_pairs.add(pair_key)
            pairs_processed_count += 1 # Increment count for this pair

            # Limit representatives and context per pair
            rep1 = list(comm1)[:min(5, len(comm1))]
            rep2 = list(comm2)[:min(5, len(comm2))]
            context_triples = [t for t in triples if t["subject"] in rep1 or t["subject"] in rep2 or t["object"] in rep1 or t["object"] in rep2][:15] # Reduced context
            triples_text = "\n".join([f"{t['subject']} {t['predicate']} {t['object']}" for t in context_triples])
            entities1 = ", ".join(rep1)
            entities2 = ", ".join(rep2)

            logger.debug(f"Running LLM inference for community pair {i+1} and {j+1}...")
            system_prompt = RELATIONSHIP_INFERENCE_SYSTEM_PROMPT
            user_prompt = get_relationship_inference_user_prompt(entities1, entities2, triples_text)

            try:
                llm_config = config.get("llm", {})
                model = llm_config.get("model")
                api_key = llm_config.get("api_key")
                max_tokens = llm_config.get("max_tokens_inference", 512)
                # --- FIX: Robust Temperature Fetching ---
                temperature = llm_config.get("temperature_inference", llm_config.get("temperature", 0.3))
                base_url = llm_config.get("base_url")

                if not model or not api_key:
                     raise ValueError("LLM model or API key missing in config for relationship inference.")

                logger.debug(f"Calling LLM for inter-community inference. Temp: {temperature}")
                response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                                    system_prompt=system_prompt, max_tokens=max_tokens,
                                    temperature=temperature, base_url=base_url)

                inferred_results = extract_json_from_text(response)

                if inferred_results and isinstance(inferred_results, list):
                    count_added = 0
                    for triple_data in inferred_results:
                        if isinstance(triple_data, dict) and all(k in triple_data for k in ["subject", "predicate", "object"]):
                            if triple_data["subject"] == triple_data["object"]: continue # Skip self-ref
                            new_triple = {
                                "subject": triple_data["subject"],
                                "predicate": limit_predicate_length(triple_data["predicate"]),
                                "object": triple_data["object"],
                                "inferred": True,
                                "confidence": triple_data.get("confidence", 0.6), # Use LLM confidence if provided
                                "chunk_id": None, # Explicitly None
                                "chunk_text": "",   # Explicitly Empty
                                "subject_type": None, # Type inference is harder
                                "object_type": None
                            }
                            newly_inferred_triples.append(new_triple)
                            count_added += 1
                    if count_added > 0:
                         logger.info(f"LLM inferred {count_added} new relationships between communities {i+1} and {j+1}.")
                else:
                    logger.warning(f"Could not extract valid inferred relationships from LLM response for communities {i+1}/{j+1}.")

            except Exception as e:
                logger.error(f"Error in LLM-based relationship inference between communities: {e}", exc_info=True)
        # Break outer loop if max pairs reached
        if pairs_processed_count >= max_community_pairs_to_process:
            break
    return newly_inferred_triples

def _infer_within_community_relationships(triples, communities, config):
    """
    Use LLM to infer relationships between entities within the same community.
    Inferred triples will have chunk_id=None, chunk_text="", types=None.

    Args:
        triples: List of existing triples
        communities: List of community sets
        config: Configuration dictionary

    Returns:
        List of new inferred triples
    """
    newly_inferred_triples = []
    logger.info("Attempting LLM inference within communities...")

    # --- Performance Optimization: Limit communities and pairs processed ---
    num_communities_to_process = config.get("inference", {}).get("llm_within_communities_to_process", 2) # Default to largest 2
    max_pairs_per_community = config.get("inference", {}).get("llm_max_pairs_per_community", 5) # Default to 5 pairs

    for idx, community in enumerate(sorted(communities, key=len, reverse=True)[:num_communities_to_process]):
        if len(community) < 5: continue # Skip small communities

        logger.debug(f"Processing within-community inference for community {idx+1} (size {len(community)})...")

        community_entities = list(community)
        connections = {(a, b): False for a in community_entities for b in community_entities if a != b}
        for triple in triples:
            if triple["subject"] in community_entities and triple["object"] in community_entities:
                connections[(triple["subject"], triple["object"])] = True
                connections[(triple["object"], triple["subject"])] = True # Consider undirected for finding pairs

        disconnected_pairs = []
        processed_undirected_pairs = set()
        # Find potentially related disconnected pairs
        for (a, b), connected in connections.items():
             pair_key = tuple(sorted((a,b)))
             if not connected and a != b and pair_key not in processed_undirected_pairs:
                a_words = set(a.lower().split())
                b_words = set(b.lower().split())
                if a_words.intersection(b_words) or a.lower() in b.lower() or b.lower() in a.lower():
                    disconnected_pairs.append((a, b))
                    processed_undirected_pairs.add(pair_key)

        # Limit pairs processed per community
        disconnected_pairs = disconnected_pairs[:max_pairs_per_community]
        if not disconnected_pairs:
             logger.debug(f"No promising disconnected pairs found in community {idx+1}.")
             continue

        logger.debug(f"Found {len(disconnected_pairs)} promising pairs for LLM inference in community {idx+1}.")

        entities_of_interest = {a for a, b in disconnected_pairs} | {b for a, b in disconnected_pairs}
        context_triples = [t for t in triples if t["subject"] in entities_of_interest or t["object"] in entities_of_interest][:15] # Reduced context
        triples_text = "\n".join([f"{t['subject']} {t['predicate']} {t['object']}" for t in context_triples])
        pairs_text = "\n".join([f"{a} and {b}" for a, b in disconnected_pairs])

        system_prompt = WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT
        user_prompt = get_within_community_inference_user_prompt(pairs_text, triples_text)

        try:
            llm_config = config.get("llm", {})
            model = llm_config.get("model")
            api_key = llm_config.get("api_key")
            max_tokens = llm_config.get("max_tokens_inference", 512)
            # --- FIX: Robust Temperature Fetching ---
            temperature = llm_config.get("temperature_inference", llm_config.get("temperature", 0.3))
            base_url = llm_config.get("base_url")

            if not model or not api_key:
                 raise ValueError("LLM model or API key missing in config for within-community inference.")

            logger.debug(f"Calling LLM for within-community inference. Temp: {temperature}")
            response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                                system_prompt=system_prompt, max_tokens=max_tokens,
                                temperature=temperature, base_url=base_url)

            inferred_results = extract_json_from_text(response)

            if inferred_results and isinstance(inferred_results, list):
                count_added = 0
                for triple_data in inferred_results:
                     if isinstance(triple_data, dict) and all(k in triple_data for k in ["subject", "predicate", "object"]):
                        if triple_data["subject"] == triple_data["object"]: continue
                        new_triple = {
                            "subject": triple_data["subject"],
                            "predicate": limit_predicate_length(triple_data["predicate"]),
                            "object": triple_data["object"],
                            "inferred": True,
                            "confidence": triple_data.get("confidence", 0.5),
                            "chunk_id": None,
                            "chunk_text": "",
                            "subject_type": None,
                            "object_type": None
                        }
                        newly_inferred_triples.append(new_triple)
                        count_added += 1
                if count_added > 0:
                     logger.info(f"LLM inferred {count_added} new relationships within community {idx+1}.")
            else:
                logger.warning(f"Could not extract valid inferred relationships from LLM response for community {idx+1}.")

        except Exception as e:
            logger.error(f"Error in LLM-based relationship inference within community {idx+1}: {e}", exc_info=True) # Log full traceback

    return newly_inferred_triples

def _infer_relationships_by_lexical_similarity(entities: Set, triples: List[Dict]) -> List[Dict]:
    """
    Infer relationships between entities based on lexical similarity.
    Inferred triples will have chunk_id=None, chunk_text="", types=None.

    Args:
        entities: Set of all entity names (standardized).
        triples: List of existing triples (used to check existing connections).

    Returns:
        List of new inferred triples.
    """
    new_triples = []
    processed_pairs = set() # Store tuples of sorted entity pairs

    # Create a set of existing relationship pairs (subject, object) for quick lookup
    existing_relationships = set()
    for triple in triples:
        # Store both directions for undirected check
        existing_relationships.add(tuple(sorted((triple["subject"], triple["object"]))))

    entities_list = sorted(list(entities)) # Sort for consistent processing order
    logger.info(f"Inferring lexical relationships among {len(entities_list)} entities...")
    count = 0

    for i, entity1 in enumerate(entities_list):
        for entity2 in entities_list[i + 1:]: # Avoid self-comparison and duplicates
            pair_key = tuple(sorted((entity1, entity2)))

            # Skip if already connected or processed
            if pair_key in existing_relationships or pair_key in processed_pairs:
                continue

            processed_pairs.add(pair_key)

            # Lexical similarity checks
            e1_lower = entity1.lower()
            e2_lower = entity2.lower()
            e1_words = set(e1_lower.split())
            e2_words = set(e2_lower.split())
            shared_words = e1_words.intersection(e2_words)

            inferred_pred = None
            subj, obj = entity1, entity2 # Default direction

            if shared_words:
                # Basic "related_to" for shared words
                inferred_pred = "related to"
            elif e1_lower in e2_lower:
                inferred_pred = "is type of"
                subj, obj = entity2, entity1 # Longer entity is type of shorter one
            elif e2_lower in e1_lower:
                inferred_pred = "is type of"
                subj, obj = entity1, entity2 # Longer entity is type of shorter one

            if inferred_pred:
                new_triple = {
                    "subject": subj,
                    "predicate": inferred_pred, # Already short
                    "object": obj,
                    "inferred": True,
                    "confidence": 0.4, # Low confidence for lexical
                    "chunk_id": None,
                    "chunk_text": "",
                    "subject_type": None,
                    "object_type": None
                }
                # Avoid adding duplicates within this inference step
                if (subj, inferred_pred, obj) not in {(t["subject"], t["predicate"], t["object"]) for t in new_triples}:
                     new_triples.append(new_triple)
                     count += 1

    logger.info(f"Inferred {count} relationships based on lexical similarity.")
    return new_triples