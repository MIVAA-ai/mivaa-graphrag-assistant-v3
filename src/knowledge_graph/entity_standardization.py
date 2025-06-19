"""Entity standardization and relationship inference for knowledge graphs."""
import logging
import re
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

# FIXED: Reduced logging verbosity - only log at WARNING level by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # CHANGED from DEBUG to WARNING

if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    # SIMPLIFIED: Less verbose formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

# Assuming these imports are available in the project structure
try:
    from src.knowledge_graph.llm import call_llm, extract_json_from_text
    from src.knowledge_graph.prompts import (
        ENTITY_RESOLUTION_SYSTEM_PROMPT,
        get_entity_resolution_user_prompt,
        RELATIONSHIP_INFERENCE_SYSTEM_PROMPT,
        get_relationship_inference_user_prompt,
        WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT,
        get_within_community_inference_user_prompt
    )
    llm_features_available = True
except ImportError as e:
    logger.error(f"Failed to import LLM/prompt modules: {e}")
    llm_features_available = False

    def call_llm(*args, **kwargs):
        logger.error("call_llm function not available due to import error.")
        raise NotImplementedError("call_llm is not available.")

    def extract_json_from_text(*args, **kwargs):
        logger.error("extract_json_from_text function not available due to import error.")
        return None

# Define allowed cross-type relationships
ALLOWED_CROSS_TYPE_RELATIONSHIPS = {
    ('Invoice', 'Date'): True,
    ('Invoice', 'Company'): True,
    ('Invoice', 'Currency'): True,
    ('Invoice', 'Value'): True,
    ('Activity', 'Date'): True,
    ('Activity', 'Equipment'): True,
    ('Activity', 'Asset'): True,
    ('Activity', 'Person'): True,
    ('Activity', 'Material'): True,
    ('Equipment', 'Activity'): True,
    ('Equipment', 'Asset'): True,
    ('Equipment', 'Location'): True,
    ('Asset', 'Activity'): True,
    ('Asset', 'Location'): True,
    ('Asset', 'Material'): True,
    ('Company', 'Location'): True,
    ('Company', 'Invoice'): True,
    ('Company', 'Contract'): True,
    ('Task', 'Date'): True,
    ('Task', 'Equipment'): True,
    ('Task', 'Description'): True,
    ('Task', 'Value'): True,
    ('Location', 'Facility'): True,
    ('Location', 'Contract'): True,
    ('Facility', 'Identifier'): True
}

def is_cross_type_relationship_allowed(type1: str, type2: str) -> bool:
    """Check if a cross-type relationship between two entity types is allowed"""
    if not type1 or not type2:
        return True

    if type1 == type2:
        return True

    return (ALLOWED_CROSS_TYPE_RELATIONSHIPS.get((type1, type2), False) or
            ALLOWED_CROSS_TYPE_RELATIONSHIPS.get((type2, type1), False))

def check_entity_compatibility(entity1_orig: str, entity2_orig: str,
                               entity_details: dict) -> bool:
    """Check if two entities can be merged/related"""
    e1_lower = entity1_orig.lower()
    e2_lower = entity2_orig.lower()

    types1 = entity_details.get(e1_lower, {}).get("types", set())
    types2 = entity_details.get(e2_lower, {}).get("types", set())

    if not types1 or not types2:
        return True

    for t1 in types1:
        for t2 in types2:
            if is_cross_type_relationship_allowed(t1, t2):
                return True

    # FIXED: Removed excessive debug logging of rejections
    return False

def limit_predicate_length(predicate, max_words=3):
    """Enforce a maximum word limit on predicates."""
    if not isinstance(predicate, str):
        predicate = str(predicate)
    words = predicate.split()
    if len(words) <= max_words:
        return predicate

    shortened = ' '.join(words[:max_words])
    stop_words = {'a', 'an', 'the', 'of', 'with', 'by', 'to', 'from', 'in', 'on', 'for'}
    last_word = shortened.split()[-1].lower() if shortened else ""
    if last_word in stop_words and len(words) > 1:
        shortened = ' '.join(shortened.split()[:-1])

    return shortened

def apply_synonyms(name: Any, synonym_map: Dict[str, str]) -> Any:
    """Apply predefined synonyms to an entity name."""
    if not isinstance(name, str) or not name.strip():
        return name

    lower_name = name.lower().strip()
    canonical_name = synonym_map.get(lower_name)

    if canonical_name and canonical_name != name:
        # REMOVED: Excessive debug logging of synonym applications
        return canonical_name
    else:
        return name

def standardize_entities(triples: List[Dict], config: Dict) -> List[Dict]:
    """Standardize entity names across all triples, preserving chunk information."""
    if not triples:
        logger.warning("standardize_entities received empty list")
        return []

    # REDUCED: Less verbose logging
    logger.info(f"Starting entity standardization for {len(triples)} triples")

    # Configuration extraction for synonyms
    standardization_config = config.get("standardization", {})
    synonym_map = standardization_config.get("synonyms", {})
    if not isinstance(synonym_map, dict):
        logger.warning("Synonyms in config is not a dictionary. Disabling synonym mapping.")
        synonym_map = {}
    else:
        synonym_map = {str(k).lower(): str(v) for k, v in synonym_map.items() if k and v}
        if synonym_map:
            logger.info(f"Loaded {len(synonym_map)} entity synonyms")

    # Initial validation & synonym application
    valid_triples = []
    invalid_count = 0
    triples_with_synonyms = []

    for i, triple in enumerate(triples):
        if not (isinstance(triple, dict) and
                all(k in triple for k in ["subject", "predicate", "object"]) and
                isinstance(triple["subject"], str) and triple["subject"].strip() and
                isinstance(triple["predicate"], str) and triple["predicate"].strip() and
                isinstance(triple["object"], str) and triple["object"].strip()):
            # REDUCED: Only log warnings for invalid triples, not every single one
            invalid_count += 1
            continue

        triple.setdefault('chunk_id', None)
        triple.setdefault('chunk_text', "")
        triple.setdefault('subject_type', None)
        triple.setdefault('object_type', None)
        valid_triples.append(triple)

        if synonym_map:
            mod_triple = triple.copy()
            mod_triple["subject"] = apply_synonyms(triple["subject"], synonym_map)
            mod_triple["object"] = apply_synonyms(triple["object"], synonym_map)
            triples_with_synonyms.append(mod_triple)
        else:
            triples_with_synonyms.append(triple)

    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} invalid/incomplete triples")

    if not triples_with_synonyms:
        logger.error("No valid triples remaining after validation")
        return []

    # Extract all unique entities and store original casings + types
    entity_details = defaultdict(lambda: {"casings": set(), "types": set(), "count": 0})
    for triple in triples_with_synonyms:
        subj_lower = triple["subject"].lower()
        obj_lower = triple["object"].lower()

        entity_details[subj_lower]["casings"].add(triple["subject"])
        entity_details[subj_lower]["count"] += 1
        if triple.get("subject_type"):
            entity_details[subj_lower]["types"].add(triple["subject_type"])

        entity_details[obj_lower]["casings"].add(triple["object"])
        entity_details[obj_lower]["count"] += 1
        if triple.get("object_type"):
            entity_details[obj_lower]["types"].add(triple["object_type"])

    all_entities_lower = set(entity_details.keys())

    # Group similar entities
    standardized_entities_map = {}
    entity_groups = defaultdict(list)

    def normalize_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "as", "ltd", "inc", "corp", "limited", "llc", "plc"}
        text = re.sub(r'[^\w\s#-]', '', text)
        words = [word for word in text.split() if word not in stopwords]
        return " ".join(words)

    sorted_entities = sorted(list(all_entities_lower), key=lambda x: (-len(x), x))

    # First pass: Standard normalization grouping
    for entity_lower in sorted_entities:
        normalized = normalize_text(entity_lower)
        if normalized:
            entity_groups[normalized].append(entity_lower)

    # For each group, choose the most representative name
    for group_key, variants_lower in entity_groups.items():
        if len(variants_lower) == 1:
            lower_variant = variants_lower[0]
            casings = entity_details[lower_variant]["casings"]
            original_casing = max(casings, key=len) if casings else lower_variant
            standardized_entities_map[lower_variant] = original_casing
        else:
            def get_rep_casing_len(variant_low):
                casings = entity_details[variant_low]["casings"]
                return max(len(c) for c in casings) if casings else 0

            standard_form_lower = sorted(variants_lower, key=lambda x: (-get_rep_casing_len(x), -entity_details[x]["count"]))[0]
            casings = entity_details[standard_form_lower]["casings"]
            standard_form_original = max(casings, key=len) if casings else standard_form_lower

            # REMOVED: Debug logging of grouping operations

            for variant_lower in variants_lower:
                standardized_entities_map[variant_lower] = standard_form_original

    # Second pass: check for root word/subset relationships
    additional_standardizations = {}
    standard_forms_original = set(standardized_entities_map.values())
    sorted_standards_original = sorted(list(standard_forms_original), key=len)

    for i, entity1_orig in enumerate(sorted_standards_original):
        e1_lower = entity1_orig.lower()
        e1_words = set(e1_lower.split())

        for entity2_orig in sorted_standards_original[i + 1:]:
            e2_lower = entity2_orig.lower()
            if e1_lower == e2_lower:
                continue

            if not check_entity_compatibility(entity1_orig, entity2_orig, entity_details):
                continue

            e2_words = set(e2_lower.split())
            if e1_words.issubset(e2_words) and len(e1_words) > 0:
                additional_standardizations[e1_lower] = entity2_orig
            elif e2_words.issubset(e1_words) and len(e2_words) > 0:
                additional_standardizations[e2_lower] = entity1_orig
            else:
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

    # Apply additional standardizations
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

    # Apply final standardization map to all triples
    standardized_triples_final = []
    for triple in triples_with_synonyms:
        subj_lower_syn = triple["subject"].lower()
        obj_lower_syn = triple["object"].lower()

        final_subj = final_map.get(subj_lower_syn, triple["subject"])
        final_obj = final_map.get(obj_lower_syn, triple["object"])

        # REMOVED: Excessive logging of map applications

        # Get consolidated types for the final canonical entities
        final_subj_lower_canon = final_subj.lower()
        final_obj_lower_canon = final_obj.lower()

        all_subj_types = set()
        for variant_lower, standard_name in final_map.items():
            if standard_name == final_subj:
                all_subj_types.update(entity_details.get(variant_lower, {}).get("types", set()))
        all_subj_types.update(entity_details.get(final_subj_lower_canon, {}).get("types", set()))
        all_subj_types = {t for t in all_subj_types if t}

        all_obj_types = set()
        for variant_lower, standard_name in final_map.items():
            if standard_name == final_obj:
                all_obj_types.update(entity_details.get(variant_lower, {}).get("types", set()))
        all_obj_types.update(entity_details.get(final_obj_lower_canon, {}).get("types", set()))
        all_obj_types = {t for t in all_obj_types if t}

        subj_type = next(iter(all_subj_types), triple.get("subject_type"))
        obj_type = next(iter(all_obj_types), triple.get("object_type"))

        new_triple = {
            "subject": final_subj,
            "subject_type": subj_type,
            "predicate": limit_predicate_length(triple["predicate"]),
            "object": final_obj,
            "object_type": obj_type,
            "chunk_id": triple.get("chunk_id"),
            "chunk_text": triple.get("chunk_text", ""),
            **{k: v for k, v in triple.items() if k not in {'subject', 'subject_type', 'predicate', 'object', 'object_type', 'chunk_id', 'chunk_text'}}
        }
        standardized_triples_final.append(new_triple)

    # Optional: LLM-based resolution
    if standardization_config.get("use_llm_for_entities", False):
        if llm_features_available:
            logger.info("Attempting LLM-based entity resolution...")
            standardized_triples_final = _resolve_entities_with_llm(standardized_triples_final, config)
        else:
            logger.warning("LLM entity resolution requested but not available")

    # Filter out self-referencing triples
    filtered_triples = [triple for triple in standardized_triples_final if triple["subject"] != triple["object"]]
    removed_count = len(standardized_triples_final) - len(filtered_triples)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} self-referencing triples")

    logger.info(f"Entity standardization finished. Final triple count: {len(filtered_triples)}")

    return filtered_triples

def infer_relationships(triples: List[Dict], config: Dict) -> List[Dict]:
    """Infer additional relationships between entities."""
    if not triples or len(triples) < 2:
        return triples

    logger.info("Inferring additional relationships between entities...")

    original_triples = triples
    graph = defaultdict(set)
    all_entities = set()
    for triple in original_triples:
        subj = triple["subject"]
        obj = triple["object"]
        graph[subj].add(obj)
        all_entities.add(subj)
        all_entities.add(obj)

    communities = _identify_communities(graph)
    logger.info(f"Identified {len(communities)} disconnected communities")

    newly_inferred_triples = []

    # FIXED: Get LLM config from the correct source
    run_llm_inference = config.get("inference", {}).get("enabled", True)
    if run_llm_inference:
        logger.info("LLM-based relationship inference is ENABLED")

        # Get LLM configuration for relationship inference
        llm_config = _get_inference_llm_config(config)
        if llm_config and llm_config.get("model") and llm_config.get("api_key"):
            community_triples = _infer_relationships_with_llm(original_triples, communities, config)
            if community_triples:
                newly_inferred_triples.extend(community_triples)

            within_community_triples = _infer_within_community_relationships(original_triples, communities, config)
            if within_community_triples:
                newly_inferred_triples.extend(within_community_triples)
        else:
            logger.warning("LLM inference enabled but configuration incomplete")
    else:
        logger.info("LLM-based relationship inference is DISABLED")

    # Apply transitive inference rules
    transitive_triples = _apply_transitive_inference(original_triples, graph)
    if transitive_triples:
        newly_inferred_triples.extend(transitive_triples)

    # Infer relationships based on lexical similarity
    lexical_triples = _infer_relationships_by_lexical_similarity(all_entities, original_triples)
    if lexical_triples:
        newly_inferred_triples.extend(lexical_triples)

    combined_triples = original_triples + newly_inferred_triples
    logger.info(f"Total triples before deduplication: {len(combined_triples)} ({len(original_triples)} original, {len(newly_inferred_triples)} inferred)")

    unique_triples = _deduplicate_triples(combined_triples)
    logger.info(f"Total triples after deduplication: {len(unique_triples)}")

    final_processed_triples = []
    for triple in unique_triples:
        triple["predicate"] = limit_predicate_length(triple["predicate"])
        triple.setdefault('chunk_id', None)
        triple.setdefault('chunk_text', "")
        triple.setdefault('subject_type', None)
        triple.setdefault('object_type', None)
        final_processed_triples.append(triple)

    filtered_triples = [triple for triple in final_processed_triples if triple["subject"] != triple["object"]]
    removed_count = len(final_processed_triples) - len(filtered_triples)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} self-referencing triples after inference")

    inferred_count = sum(1 for t in filtered_triples if t.get("inferred"))
    logger.info(f"Inference complete. Final triple count: {len(filtered_triples)} ({inferred_count} inferred)")

    return filtered_triples

# FIXED: New function to get LLM config for inference
def _get_inference_llm_config(config: Dict) -> Optional[Dict]:
    """Get LLM configuration for relationship inference from various config sources."""
    # Try relationship_inference specific config first (new TOML structure)
    rel_inference_config = config.get("relationship_inference", {})
    if rel_inference_config.get("model") and rel_inference_config.get("api_key"):
        return rel_inference_config

    # Try within_community_inference config
    within_config = config.get("within_community_inference", {})
    if within_config.get("model") and within_config.get("api_key"):
        return within_config

    # Fallback to main LLM config
    llm_config = config.get("llm_full_config", {}) or config.get("llm", {})
    if llm_config.get("model") and llm_config.get("api_key"):
        return llm_config

    # Final fallback to top-level config
    if config.get("LLM_MODEL") and config.get("LLM_API_KEY"):
        return {
            "model": config["LLM_MODEL"],
            "api_key": config["LLM_API_KEY"],
            "base_url": config.get("LLM_BASE_URL"),
            "max_tokens": config.get("LLM_MAX_TOKENS", 1500),
            "temperature": config.get("LLM_TEMPERATURE", 0.1)
        }

    return None

def _identify_communities(graph):
    """Identify disconnected communities in the graph."""
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    visited = set()
    communities = []

    undirected_graph = defaultdict(set)
    for u, neighbors in graph.items():
        for v in neighbors:
            undirected_graph[u].add(v)
            undirected_graph[v].add(u)

    for node in all_nodes:
        if node not in undirected_graph:
            undirected_graph[node] = set()

    def bfs(start_node, current_community):
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
            if community:
                communities.append(community)

    return communities

def _apply_transitive_inference(triples: List[Dict], graph: Dict[Any, Set]) -> List[Dict]:
    """Apply transitive inference to find new relationships."""
    new_triples = []
    predicates = {}
    for triple in triples:
        key = (triple["subject"], triple["object"])
        predicates[key] = {
            "predicate": triple["predicate"],
            "chunk_id": triple.get("chunk_id"),
            "chunk_text": triple.get("chunk_text", "")
        }

    logger.info("Applying transitive inference...")
    count = 0
    for subj in list(graph.keys()):
        if subj not in graph:
            continue
        for mid in list(graph[subj]):
            if mid not in graph:
                continue
            for obj in list(graph[mid]):
                if subj != obj and (subj, obj) not in predicates:
                    pred1_info = predicates.get((subj, mid))
                    pred2_info = predicates.get((mid, obj))

                    if pred1_info and pred2_info:
                        pred1 = pred1_info["predicate"]
                        pred2 = pred2_info["predicate"]
                        new_pred = f"{pred1} via {mid} leading to {pred2}"
                        new_triple = {
                            "subject": subj,
                            "predicate": limit_predicate_length(new_pred),
                            "object": obj,
                            "inferred": True,
                            "confidence": 0.7,
                            "chunk_id": None,
                            "chunk_text": "",
                            "subject_type": None,
                            "object_type": None
                        }
                        if (subj, new_triple["predicate"], obj) not in {(t["subject"], t["predicate"], t["object"]) for t in new_triples}:
                            new_triples.append(new_triple)
                            count += 1

    logger.info(f"Inferred {count} relationships via transitive inference")
    return new_triples

def _deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    """Remove duplicate triples based on (subject, predicate, object)."""
    unique_triples_dict = {}

    for triple in triples:
        subj = triple.get("subject")
        pred = triple.get("predicate")
        obj = triple.get("object")

        if subj is None or pred is None or obj is None:
            logger.warning(f"Skipping triple in deduplication due to missing S/P/O: {triple}")
            continue

        key = (subj, pred, obj)
        is_inferred = triple.get("inferred", False)

        if key not in unique_triples_dict:
            unique_triples_dict[key] = triple
        else:
            existing_triple = unique_triples_dict[key]
            existing_is_inferred = existing_triple.get("inferred", False)

            if not is_inferred and existing_is_inferred:
                unique_triples_dict[key] = triple

    final_list = list(unique_triples_dict.values())
    return final_list

def _resolve_entities_with_llm(triples, config):
    """Use LLM to help resolve entity references and standardize entity names."""
    logger.info("Attempting LLM-based entity resolution...")

    # Get LLM config
    llm_config = _get_inference_llm_config(config)
    if not llm_config:
        logger.warning("No LLM configuration available for entity resolution")
        return triples

    all_entities = set()
    for triple in triples:
        all_entities.add(triple["subject"])
        all_entities.add(triple["object"])

    entity_limit = config.get("standardization", {}).get("llm_entity_limit", 100)
    if len(all_entities) > entity_limit:
        entity_counts = defaultdict(int)
        for triple in triples:
            entity_counts[triple["subject"]] += 1
            entity_counts[triple["object"]] += 1
        all_entities = {entity for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:entity_limit]}
        logger.info(f"Limiting LLM entity resolution to top {entity_limit} entities")

    entity_list = "\n".join(sorted(all_entities))
    system_prompt = ENTITY_RESOLUTION_SYSTEM_PROMPT
    user_prompt = get_entity_resolution_user_prompt(entity_list)

    try:
        model = llm_config.get("model")
        api_key = llm_config.get("api_key")
        max_tokens = llm_config.get("max_tokens", 1024)
        temperature = llm_config.get("temperature", 0.1)
        base_url = llm_config.get("base_url")

        if not model or not api_key:
            raise ValueError("LLM model or API key missing in config for entity resolution")

        response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                            system_prompt=system_prompt, max_tokens=max_tokens,
                            temperature=temperature, base_url=base_url)

        entity_mapping = extract_json_from_text(response)

        if entity_mapping and isinstance(entity_mapping, dict):
            logger.info(f"Applying LLM-based entity standardization for {len(entity_mapping)} entity groups")
            entity_to_standard = {}
            for standard, variants in entity_mapping.items():
                for variant in variants:
                    entity_to_standard[variant] = standard
                entity_to_standard[standard] = standard

            resolved_triples = []
            for triple in triples:
                new_triple = triple.copy()
                new_triple["subject"] = entity_to_standard.get(triple["subject"], triple["subject"])
                new_triple["object"] = entity_to_standard.get(triple["object"], triple["object"])
                resolved_triples.append(new_triple)
            return resolved_triples
        else:
            logger.warning("Could not extract valid entity mapping from LLM response")
            return triples

    except Exception as e:
        logger.error(f"Error in LLM-based entity resolution: {e}")
        return triples

def _infer_relationships_with_llm(triples, communities, config):
    """Use LLM to infer relationships between disconnected communities."""
    if len(communities) <= 1:
        logger.info("Only one community found, skipping LLM inference between communities")
        return []

    # FIXED: Get LLM config properly
    llm_config = _get_inference_llm_config(config)
    if not llm_config:
        logger.warning("No LLM configuration available for relationship inference")
        return []

    max_community_pairs_to_process = config.get("inference", {}).get("llm_max_community_pairs", 5)
    num_communities_to_consider = config.get("inference", {}).get("llm_communities_to_consider", 3)

    large_communities = sorted(communities, key=len, reverse=True)[:num_communities_to_consider]
    newly_inferred_triples = []
    processed_pairs = set()
    pairs_processed_count = 0

    logger.info(f"Attempting LLM inference between up to {max_community_pairs_to_process} pairs from the {len(large_communities)} largest communities")

    for i, comm1 in enumerate(large_communities):
        for j, comm2 in enumerate(large_communities):
            if i >= j:
                continue

            if pairs_processed_count >= max_community_pairs_to_process:
                logger.info(f"Reached limit ({max_community_pairs_to_process}) of community pairs for LLM inference")
                break

            pair_key = tuple(sorted((i, j)))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            pairs_processed_count += 1

            rep1 = list(comm1)[:min(5, len(comm1))]
            rep2 = list(comm2)[:min(5, len(comm2))]
            context_triples = [t for t in triples if t["subject"] in rep1 or t["subject"] in rep2 or t["object"] in rep1 or t["object"] in rep2][:15]
            triples_text = "\n".join([f"{t['subject']} {t['predicate']} {t['object']}" for t in context_triples])
            entities1 = ", ".join(rep1)
            entities2 = ", ".join(rep2)

            system_prompt = RELATIONSHIP_INFERENCE_SYSTEM_PROMPT
            user_prompt = get_relationship_inference_user_prompt(entities1, entities2, triples_text)

            try:
                model = llm_config.get("model")
                api_key = llm_config.get("api_key")
                max_tokens = llm_config.get("max_tokens", 512)
                temperature = llm_config.get("temperature", 0.3)
                base_url = llm_config.get("base_url")

                if not model or not api_key:
                    raise ValueError("LLM model or API key missing in config for relationship inference")

                response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                                    system_prompt=system_prompt, max_tokens=max_tokens,
                                    temperature=temperature, base_url=base_url)

                inferred_results = extract_json_from_text(response)

                if inferred_results and isinstance(inferred_results, list):
                    count_added = 0
                    for triple_data in inferred_results:
                        if isinstance(triple_data, dict) and all(k in triple_data for k in ["subject", "predicate", "object"]):
                            if triple_data["subject"] == triple_data["object"]:
                                continue
                            new_triple = {
                                "subject": triple_data["subject"],
                                "predicate": limit_predicate_length(triple_data["predicate"]),
                                "object": triple_data["object"],
                                "inferred": True,
                                "confidence": triple_data.get("confidence", 0.6),
                                "chunk_id": None,
                                "chunk_text": "",
                                "subject_type": None,
                                "object_type": None
                            }
                            newly_inferred_triples.append(new_triple)
                            count_added += 1
                    if count_added > 0:
                        logger.info(f"LLM inferred {count_added} new relationships between communities {i+1} and {j+1}")

            except Exception as e:
                logger.error(f"Error in LLM-based relationship inference between communities: {e}")

        if pairs_processed_count >= max_community_pairs_to_process:
            break

    return newly_inferred_triples

def _infer_within_community_relationships(triples, communities, config):
    """Use LLM to infer relationships between entities within the same community."""
    newly_inferred_triples = []
    logger.info("Attempting LLM inference within communities...")

    # FIXED: Get LLM config properly
    llm_config = _get_inference_llm_config(config)
    if not llm_config:
        logger.warning("No LLM configuration available for within-community inference")
        return []

    num_communities_to_process = config.get("inference", {}).get("llm_within_communities_to_process", 2)
    max_pairs_per_community = config.get("inference", {}).get("llm_max_pairs_per_community", 5)

    for idx, community in enumerate(sorted(communities, key=len, reverse=True)[:num_communities_to_process]):
        if len(community) < 5:
            continue

        community_entities = list(community)
        connections = {(a, b): False for a in community_entities for b in community_entities if a != b}
        for triple in triples:
            if triple["subject"] in community_entities and triple["object"] in community_entities:
                connections[(triple["subject"], triple["object"])] = True
                connections[(triple["object"], triple["subject"])] = True

        disconnected_pairs = []
        processed_undirected_pairs = set()

        for (a, b), connected in connections.items():
            pair_key = tuple(sorted((a, b)))
            if not connected and a != b and pair_key not in processed_undirected_pairs:
                a_words = set(a.lower().split())
                b_words = set(b.lower().split())
                if a_words.intersection(b_words) or a.lower() in b.lower() or b.lower() in a.lower():
                    disconnected_pairs.append((a, b))
                    processed_undirected_pairs.add(pair_key)

        disconnected_pairs = disconnected_pairs[:max_pairs_per_community]
        if not disconnected_pairs:
            continue

        entities_of_interest = {a for a, b in disconnected_pairs} | {b for a, b in disconnected_pairs}
        context_triples = [t for t in triples if t["subject"] in entities_of_interest or t["object"] in entities_of_interest][:15]
        triples_text = "\n".join([f"{t['subject']} {t['predicate']} {t['object']}" for t in context_triples])
        pairs_text = "\n".join([f"{a} and {b}" for a, b in disconnected_pairs])

        system_prompt = WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT
        user_prompt = get_within_community_inference_user_prompt(pairs_text, triples_text)

        try:
            model = llm_config.get("model")
            api_key = llm_config.get("api_key")
            max_tokens = llm_config.get("max_tokens", 512)
            temperature = llm_config.get("temperature", 0.3)
            base_url = llm_config.get("base_url")

            if not model or not api_key:
                raise ValueError("LLM model or API key missing in config for within-community inference")

            response = call_llm(model=model, user_prompt=user_prompt, api_key=api_key,
                                system_prompt=system_prompt, max_tokens=max_tokens,
                                temperature=temperature, base_url=base_url)

            inferred_results = extract_json_from_text(response)

            if inferred_results and isinstance(inferred_results, list):
                count_added = 0
                for triple_data in inferred_results:
                    if isinstance(triple_data, dict) and all(k in triple_data for k in ["subject", "predicate", "object"]):
                        if triple_data["subject"] == triple_data["object"]:
                            continue
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
                    logger.info(f"LLM inferred {count_added} new relationships within community {idx+1}")

        except Exception as e:
            logger.error(f"Error in LLM-based relationship inference within community {idx+1}: {e}")

    return newly_inferred_triples

def _infer_relationships_by_lexical_similarity(entities: Set, triples: List[Dict]) -> List[Dict]:
    """Infer relationships between entities based on lexical similarity."""
    new_triples = []
    processed_pairs = set()

    existing_relationships = set()
    for triple in triples:
        existing_relationships.add(tuple(sorted((triple["subject"], triple["object"]))))

    entities_list = sorted(list(entities))
    logger.info(f"Inferring lexical relationships among {len(entities_list)} entities...")
    count = 0

    for i, entity1 in enumerate(entities_list):
        for entity2 in entities_list[i + 1:]:
            pair_key = tuple(sorted((entity1, entity2)))

            if pair_key in existing_relationships or pair_key in processed_pairs:
                continue

            processed_pairs.add(pair_key)

            e1_lower = entity1.lower()
            e2_lower = entity2.lower()
            e1_words = set(e1_lower.split())
            e2_words = set(e2_lower.split())
            shared_words = e1_words.intersection(e2_words)

            inferred_pred = None
            subj, obj = entity1, entity2

            if shared_words:
                inferred_pred = "related to"
            elif e1_lower in e2_lower:
                inferred_pred = "is type of"
                subj, obj = entity2, entity1
            elif e2_lower in e1_lower:
                inferred_pred = "is type of"
                subj, obj = entity1, entity2

            if inferred_pred:
                new_triple = {
                    "subject": subj,
                    "predicate": inferred_pred,
                    "object": obj,
                    "inferred": True,
                    "confidence": 0.4,
                    "chunk_id": None,
                    "chunk_text": "",
                    "subject_type": None,
                    "object_type": None
                }
                if (subj, inferred_pred, obj) not in {(t["subject"], t["predicate"], t["object"]) for t in new_triples}:
                    new_triples.append(new_triple)
                    count += 1

    logger.info(f"Inferred {count} relationships based on lexical similarity")
    return new_triples