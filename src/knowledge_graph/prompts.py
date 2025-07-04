"""Centralized repository for all LLM prompts used in the knowledge graph system."""

# Phase 1: Main extraction prompts
MAIN_SYSTEM_PROMPT = """
Role: You are an AI expert in Entity and Relationship Extraction for Physical Asset Management Knowledge Graph generation.

Responsibilities:
- Extract meaningful entities from asset management documents.
- Identify relationships (triplets) between assets, locations, personnel, and maintenance activities.
- Ensure predicates (relationship names) are extremely concise.

Critical Guidelines:
- Predicates must be maximum 6 words.
- Prefer 2-3 words for clarity and graph readability.
"""

# --- REFINED MAIN_USER_PROMPT (Focused on Core O&G Data) ---
MAIN_USER_PROMPT = """
Your critical task: Read the text below (delimited by triple backticks) and identify ALL Subject-Predicate-Object (S-P-O) relationships relevant to the **Physical Asset Management domain**. For EACH relationship, you MUST identify the TYPE for BOTH the subject and the object. Produce a single JSON array containing objects, where EACH object represents one S-P-O triple and MUST include ALL FIVE of the following keys: "subject", "subject_type", "predicate", "object", and "object_type".

Domain Context:
The text relates specifically to **Physical Asset Management**, focusing on **equipment tracking, maintenance operations, and facility management**. This includes concepts like:
- **Asset Data:** Equipment specifications, serial numbers, manufacturer information, installation dates, location assignments, condition status
- **Maintenance Data:** Work orders, inspection reports, preventive maintenance schedules, corrective actions, parts consumption, labor hours
- **Location Data:** Facility hierarchies (site → building → floor → room), geographic coordinates, asset placement, storage locations
- **Personnel Data:** Technicians, operators, inspectors, managers, contractors, certifications, assignments, responsibilities
- **Operational Data:** Equipment performance, utilization rates, downtime events, efficiency metrics, safety incidents
- **Supply Chain:** Parts inventory, suppliers, purchase orders, warranties, contracts, delivery schedules
- **Compliance:** Regulatory requirements, inspection schedules, certifications, safety protocols, environmental compliance
- **Entities:** Assets (Equipment, Machinery, Systems, Components), Locations (Sites, Buildings, Facilities), Personnel (Technicians, Operators, Managers), Organizations (Manufacturers, Suppliers, Contractors), Documents (Manuals, Reports, Procedures), Parts (Components, Materials, Tools), Measurements (Metrics, Readings, Specifications)

Follow these rules METICULOUSLY:

- **MANDATORY FIELDS:** Every JSON object in the output array MUST contain these exact five keys: `subject`, `subject_type`, `predicate`, `object`, `object_type`. NO EXCEPTIONS. If you cannot determine a specific type, use a reasonable default like "Asset", "Location", "Person", or "Document", but the key MUST be present.
- **Entity Consistency:** Use consistent, lowercase names for entities (e.g., "compressor unit 101" not "Compressor Unit #101", "preventive maintenance" not "PM"). Apply standard abbreviations where appropriate (e.g., "hvac" for Heating, Ventilation, and Air Conditioning).
- **Entity Types:** Identify the type for each subject and object using **Title Case**. Be specific to the Asset Management domain. Examples:
    - **Asset Types:** `Equipment`, `Machinery`, `System`, `Component`, `Infrastructure`, `Vehicle`, `Tool`, `Instrument`
    - **Location Types:** `Site`, `Building`, `Floor`, `Room`, `Area`, `Zone`, `Facility`, `Warehouse`, `Location`
    - **Personnel Types:** `Technician`, `Operator`, `Inspector`, `Manager`, `Contractor`, `Engineer`, `Supervisor`, `Person`
    - **Organization Types:** `Manufacturer`, `Supplier`, `ServiceProvider`, `Contractor`, `Operator`, `Company`, `Department`
    - **Document Types:** `Manual`, `Report`, `Procedure`, `Specification`, `Certificate`, `WorkOrder`, `Invoice`
    - **Maintenance Types:** `WorkOrder`, `Inspection`, `PreventiveMaintenance`, `CorrectiveMaintenance`, `Schedule`, `Task`
    - **Part Types:** `Component`, `SparePart`, `Material`, `Consumable`, `Tool`, `Inventory`
    - **Abstract Types:** `Measurement`, `Specification`, `Condition`, `Status`, `Priority`, `Cost`, `Date`, `Value`
- **Atomic Terms:** Identify distinct key terms. Break down complex descriptions if possible (e.g., "emergency shutdown system" might yield `(emergency_shutdown_system, is_type_of, safety_system)` and `(emergency_shutdown_system, located_in, control_room)`).
- **Handle Lists:** If the text mentions a list of items related to a subject (e.g., 'equipment including A, B, and C', 'technicians X, Y, Z assigned'), create **separate triples** for each item. Example: `(facility, contains_equipment, pump_a)`, `(facility, contains_equipment, compressor_b)`, etc.
- **Quantitative Data:** Extract specific metrics and link them to the relevant entity (e.g., `(pump_101, has_flow_rate, 500_gpm)`, `(motor_a, operating_temperature, 85_celsius)`, `(work_order_123, estimated_hours, 4.5_hours)`). Use predicates like `has_value`, `measures`, `operates_at`, `rated_for`.
- **CRITICAL PREDICATE LENGTH:** Predicates MUST be 4-6 words MAXIMUM, ideally 2-3 words. Be concise and use verbs. Examples: `located_in`, `manufactured_by`, `maintained_by`, `operates`, `contains`, `assigned_to`, `requires`, `scheduled_for`. Use lowercase with underscores (`snake_case`).
- **Completeness:** Extract ALL identifiable relationships relevant to the **physical asset management domain**.
- **Standardization:** Use consistent terminology (e.g., use "preventive maintenance" consistently, not "PM" or "planned maintenance").
- **Lowercase Values:** ALL text values for `subject`, `predicate`, and `object` MUST be lowercase.
- **No Special Characters:** Avoid symbols like %, @, ", ", °, etc., in values. Use plain text equivalents (e.g., "degrees c", "percent", "number").

Important Considerations:
- Precision in asset identification (equipment IDs, serial numbers, locations) is key.
- Maximize graph connectedness via consistent naming and relationship extraction.
- Consider the full context of maintenance operations and asset lifecycles.
- **ALL FIVE KEYS (`subject`, `subject_type`, `predicate`, `object`, `object_type`) ARE MANDATORY FOR EVERY TRIPLE.**

Output Requirements:
- Output ONLY the JSON array. No introductory text, commentary, or explanations.
- Ensure the entire output is a single, valid JSON array.
- Each object within the array MUST have the five required keys.

Example of the required output structure (Notice all five keys and domain relevance):

[
  {
    "subject": "compressor unit 101",
    "subject_type": "Equipment",
    "predicate": "located_in",
    "object": "building a",
    "object_type": "Building"
  },
  {
    "subject": "work order wo2024001",
    "subject_type": "WorkOrder",
    "predicate": "assigned_to",
    "object": "technician smith",
    "object_type": "Technician"
  },
  {
    "subject": "pump motor",
    "subject_type": "Equipment",
    "predicate": "manufactured_by",
    "object": "siemens",
    "object_type": "Manufacturer"
  },
  {
    "subject": "preventive maintenance pm101",
    "subject_type": "PreventiveMaintenance",
    "predicate": "scheduled_for",
    "object": "march 15 2025",
    "object_type": "Date"
  },
  {
    "subject": "temperature sensor",
    "subject_type": "Instrument",
    "predicate": "monitors",
    "object": "boiler temperature",
    "object_type": "Measurement"
  }
]

Crucial Reminder: Every single object in the JSON array must strictly adhere to having the `subject`, `subject_type`, `predicate`, `object`, and `object_type` keys. Ensure predicate is `snake_case`.

Text to analyze (between triple backticks):
"""

# Phase 2: Entity standardization prompts
ENTITY_RESOLUTION_SYSTEM_PROMPT = """
You are an expert in entity resolution and knowledge representation for Physical Asset Management systems.
Your task is to standardize entity names from an asset management knowledge graph to ensure consistency.
Focus on assets, equipment, locations, personnel, and maintenance-related entities.
"""

def get_entity_resolution_user_prompt(entity_list):
    return f"""
Below is a list of entity names extracted from a physical asset management knowledge graph. 
Some may refer to the same real-world entities but with different wording, abbreviations, or formatting.

Please identify groups of entities that refer to the same concept, and provide a standardized name for each group.
Focus on:
- Equipment and asset names (standardize model numbers, serial numbers, naming conventions)
- Location identifiers (building codes, room numbers, facility names)
- Personnel names and roles (technician titles, department names)
- Manufacturer and supplier names
- Part numbers and component identifiers

Return your answer as a JSON object where the keys are the standardized names and the values are arrays of all variant names that should map to that standard name.
Only include entities that have multiple variants or need standardization.

Entity list:
{entity_list}

Format your response as valid JSON like this:
{{
  "standardized name 1": ["variant 1", "variant 2"],
  "standardized name 2": ["variant 3", "variant 4", "variant 5"]
}}
"""

# Phase 3: Community relationship inference prompts
RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference for Physical Asset Management systems. 
Your task is to infer plausible relationships between disconnected entities in an asset management knowledge graph.
"""

def get_relationship_inference_user_prompt(entities1, entities2, triples_text):
    return f"""
I have an asset management knowledge graph with two disconnected communities of entities. 

Community 1 entities: {entities1}
Community 2 entities: {entities2}

Here are some existing relationships involving these entities:
{triples_text}

Please infer 2-3 plausible relationships between entities from Community 1 and entities from Community 2.
Focus on typical asset management relationships such as:
- Equipment-location relationships (located_in, installed_at)
- Maintenance relationships (maintained_by, requires_service)
- Operational relationships (operates, controls, monitors)
- Supply chain relationships (supplied_by, manufactured_by)
- Personnel assignments (assigned_to, supervised_by)

Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity from community 1",
    "predicate": "inferred relationship",
    "object": "entity from community 2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
For predicates, use short phrases that clearly describe the relationship.
IMPORTANT: Make sure the subject and object are different entities - avoid self-references.
"""

# Phase 4: Within-community relationship inference prompts (updated for asset management)
WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference for Physical Asset Management systems. 
Your task is to infer plausible relationships between semantically related entities that are not yet connected in an asset management knowledge graph.
"""

# Phase 4: Within-community relationship inference prompts
WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference for Physical Asset Management systems. 
Your task is to infer plausible relationships between semantically related entities that are not yet connected in an asset management knowledge graph.
"""

def get_within_community_inference_user_prompt(pairs_text, triples_text):
    return f"""
I have an asset management knowledge graph with several entities that appear to be semantically related but are not directly connected.

Here are some pairs of entities that might be related:
{pairs_text}

Here are some existing relationships involving these entities:
{triples_text}

Please infer plausible relationships between these disconnected pairs.
Focus on typical asset management relationships such as:
- Asset hierarchies (part_of, contains, composed_of)
- Maintenance dependencies (depends_on, affects, triggers)
- Operational relationships (powers, controls, feeds_into)
- Location relationships (adjacent_to, connects_to, serves)
- Temporal relationships (precedes, follows, scheduled_after)

Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity1",
    "predicate": "inferred relationship",
    "object": "entity2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
IMPORTANT: Make sure that the subject and object are different entities - avoid self-references.
"""


# --- NEW: Text-to-Cypher Prompt (with Fuzzy Matching for Entities & Relationships) ---
# In your prompts.py, REPLACE the TEXT_TO_CYPHER_SYSTEM_PROMPT with this fixed version:

TEXT_TO_CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query translator. Convert natural language questions into Cypher queries using the provided graph schema.

Graph Schema:
{dynamic_schema}

CRITICAL RULES:
1. **NO UNION QUERIES** - Avoid UNION completely, use single queries with OR conditions instead
2. Use the pre-linked entities when provided for exact matching
3. Generate a single MATCH pattern that captures all needed relationships

Entity Matching Rules:
- If pre-linked entity provided: Use exact match `WHERE toLower(node.name) = toLower('canonical_name')`
- If not pre-linked: Use fuzzy match `WHERE toLower(node.name) CONTAINS toLower('mention')`

Query Patterns (NO UNION):
- For multiple entity types: `MATCH (a)-[r]-(b) WHERE ... AND (b:Person OR b:Company OR b:Service) RETURN b.name, labels(b)`
- Simple exploration: `MATCH (a)-[r]-(related) WHERE ... RETURN a.name, type(r), related.name LIMIT 25`
- Specific relationships: `MATCH (a)-[r:SPECIFIC_TYPE]->(b) WHERE ...`

NEVER use UNION - it causes syntax errors. Instead use OR conditions in WHERE clauses.

Return Rules:
- Enclose query in ```cypher ... ```
- Return "NO_QUERY_GENERATED" if question cannot be answered
- Use LIMIT for broad queries
- Return properties (.name) not full nodes

Examples:

Question: Who are the personnel and contractors for project X?
```cypher
MATCH (e:Entity)-[r]-(related:Entity)
WHERE toLower(e.name) CONTAINS 'project x'
  AND (related:Person OR related:Company OR related:Service)
RETURN related.name AS name, 
       labels(related) AS entity_type,
       type(r) AS relationship
LIMIT 20
```

Question: Tell me about building 5 staff and vendors
```cypher
MATCH (e:Entity)-[r]-(related:Entity)
WHERE toLower(e.name) CONTAINS 'building 5'
  AND (related:Person OR related:Company)
OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c:Chunk)
RETURN e.name AS entity, 
       related.name AS related_entity, 
       labels(related) AS type,
       type(r) AS relationship,
       c.text AS context
LIMIT 25
```
"""

# User prompt template for Cypher generation, including few-shot examples
GENERATE_USER_TEMPLATE = """You are a Neo4j expert. Given an input question and potentially pre-linked entities, create a syntactically correct Cypher query to run.
Use the provided schema and few-shot examples to guide your query generation.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!

Schema:
{schema}

Few-shot Examples:
{fewshot_examples}

User Input:
{structured_input}

Cypher query:"""

# --- Prompts for Handling Empty Results ---

EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are a meticulous Cypher query analyst. Your objective is to diagnose why a given Cypher query returned zero results when executed against a graph database with the specified schema.

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Based *only* on the user's question, the executed Cypher query, and the schema, determine the *single most likely* reason for the empty result. Choose exclusively from these options:

1.  **QUERY_MISMATCH**: The query's structure (nodes, relationships, properties, filters) does not accurately reflect the semantic intent of the user's question given the schema. For example, the query looks for the wrong relationship type, filters on an inappropriate property, or misunderstands the core entities involved in the question.
2.  **DATA_ABSENT**: The query *does* accurately represent the user's question according to the schema, but the specific data requested simply does not exist in the graph. The query structure is appropriate for the question, but the graph lacks the necessary nodes or relationships satisfying the query's conditions.
3.  **AMBIGUOUS**: The user's question is too vague, unclear, or open to multiple interpretations, making it impossible to definitively determine if the query was mismatched or if the data is truly absent based *only* on the provided information.

**Output Format:**
Respond with ONLY ONE of the reason codes: `QUERY_MISMATCH`, `DATA_ABSENT`, or `AMBIGUOUS`. Do not provide any explanation or justification.

"""

EVALUATE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Executed Cypher Query (Returned 0 results):**
```cypher
{cypher}
```

**Diagnosis Code:**
"""

# --- Prompts for Revising Queries Evaluated as QUERY_MISMATCH ---

REVISE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are an expert Neo4j Cypher query generator specializing in query correction. You are given a user question, a graph schema, and an initial Cypher query that returned zero results. The initial query has been evaluated as a `QUERY_MISMATCH`, meaning it likely failed to accurately capture the user's intent based on the question and schema.

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Rewrite the original Cypher query to create a *revised* query that better aligns with the user's question and the provided schema.

**Revision Guidelines:**
1.  **Analyze Mismatch:** Identify the specific parts of the original query that likely caused the mismatch (e.g., wrong relationship type, incorrect node label, overly strict filtering on `name` property, wrong direction).
2.  **Reflect Intent:** Construct the revised query to target the entities, relationships, and properties implied by the user's question.
3.  **Consider Flexibility:**
    * If appropriate, make filters less strict (e.g., use `toLower(n.name) CONTAINS 'keyword'` instead of `toLower(n.name) = 'exact match'`).
    * If the relationship type was potentially wrong, try matching a more general pattern (`-[r]-`) and potentially filtering on `r.original` if keywords are available in the question.
    * Ensure the `RETURN` clause provides the specific information requested (e.g., return `entity.name` rather than the whole node).
4.  **Schema Adherence:** Ensure the revised query is valid according to the provided schema.
5.  **No Revision Case:** If, after careful analysis, you determine that the original query *was* the most plausible interpretation of the question despite returning no results, or if no meaningful revision seems possible to better capture the intent, then output the exact text `NO_REVISION`.

**Output Format:**
- If a revision is possible, output *only* the revised Cypher query enclosed in triple backticks: ```cypher\n[REVISED QUERY]\n```
- If no revision is appropriate, output *only* the exact text: `NO_REVISION`
- Do not include any explanations or comments outside the query block or the `NO_REVISION` text.

"""

REVISE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Original Cypher Query (Returned 0 results, evaluated as QUERY_MISMATCH):**
```cypher
{cypher}
```

**Revised Cypher Query or NO_REVISION:**
"""