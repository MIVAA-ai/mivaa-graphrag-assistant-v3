"""Centralized repository for all LLM prompts used in the knowledge graph system.
ENHANCED with Universal Pattern Support and Work-Order Awareness."""

# Phase 1: Main extraction prompts (PRESERVED - these are excellent)
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

# --- REFINED MAIN_USER_PROMPT (Focused on Core O&G Data) --- (PRESERVED)
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

# Phase 2: Entity standardization prompts (PRESERVED)
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

# Phase 3: Community relationship inference prompts (PRESERVED)
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

# Phase 4: Within-community relationship inference prompts (PRESERVED)
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

# ========================================================================
# 🎯 ENHANCED TEXT-TO-CYPHER PROMPT WITH UNIVERSAL PATTERN AWARENESS
# ========================================================================

TEXT_TO_CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query translator with advanced pattern recognition capabilities. Convert natural language questions into Cypher queries using the provided graph schema and universal pattern library.

Graph Schema:
{dynamic_schema}

CRITICAL RULES:
1. **NO UNION QUERIES** - Avoid UNION completely, use single queries with OR conditions instead
2. Use the pre-linked entities when provided for exact matching
3. Generate a single MATCH pattern that captures all needed relationships
4. **PRIORITIZE WORK-ORDER PATTERNS** - Your system uses a work-order-centric model

**UNIVERSAL PATTERN AWARENESS:**
Your system has intelligent pattern detection that automatically identifies the best query approach. However, you should understand the pattern priorities:

**🎯 WORK-ORDER PATTERNS (HIGHEST PRIORITY):**
The system prioritizes these patterns for maintenance-related questions:

**Pattern Priority Order:**
1. **Work-Order-Centric Maintenance** (confidence: 0.9) - Who maintains what equipment
2. **Personnel Work Assignment** (confidence: 0.9) - What work orders are assigned to whom  
3. **Work Order Detailed Analysis** (confidence: 0.9) - Complete work order details
4. **Asset Maintenance History** (confidence: 0.9) - Chronological work performed on equipment
5. **Work Order Status/Approval** (confidence: 0.9) - Work order workflow analysis
6. **Work Order by Approver** (confidence: 0.9) - Work orders approved by specific personnel

**🔄 UNIVERSAL FALLBACK PATTERNS (confidence: 0.7):**
- Asset maintenance workflow analysis
- Cost analysis and financial breakdown  
- Personnel assignment analysis
- Inventory and stock level analysis
- Location and facility analysis
- Temporal analysis
- Hierarchical relationship analysis
- Service & vendor analysis

**WORK ORDER RELATIONSHIP DIRECTIONS (CRITICAL):**
Your data follows these EXACT relationship patterns:
- Work Order → ASSIGNED_TO → Person (work orders are assigned to people)
- Work Order → PERFORMED_ON → Asset (work orders are performed on assets) 
- Work Order → APPROVED_BY → Manager (work orders are approved by managers)
- Work Order → COMPLETED_ON → Date (work completion dates)
- Work Order → USED_PART → Part (parts used in work orders)

**MAINTENANCE QUESTION DETECTION:**
The system automatically detects these as work-order questions:
- Questions with: 'maintenance', 'repair', 'service', 'work order', 'assigned to', 'performed on', 'who maintains'
- Questions combining equipment + personnel words
- Questions about work assignments, approvals, maintenance history

**ENHANCED PATTERN TEMPLATES:**

**Template 1: Who maintains [ASSET]? (Auto-detected, boosted confidence)**
```cypher
MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity),
      (wo)-[:PERFORMED_ON]->(asset:Entity)
WHERE toLower(asset.name) CONTAINS toLower('asset_name')
OPTIONAL MATCH (wo)-[:USED_PART|REQUIRES|INCLUDES]->(part:Entity)
WHERE 'Part' IN labels(part) OR 'Component' IN labels(part)
OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
RETURN asset.name as asset_name,
       wo.name as work_order,
       person.name as assigned_personnel,
       collect(DISTINCT part.name) as parts_used,
       collect(DISTINCT date.name) as work_dates
ORDER BY wo.name LIMIT 15
```

**Template 2: What work is assigned to [PERSON]? (Auto-detected, boosted confidence)**
```cypher
MATCH (person:Entity)<-[:ASSIGNED_TO]-(wo:Entity)-[:PERFORMED_ON]->(asset:Entity)
WHERE toLower(person.name) CONTAINS toLower('person_name')
AND ('Person' IN labels(person) OR 'Technician' IN labels(person))
OPTIONAL MATCH (wo)-[:USED_PART|REQUIRES]->(part:Entity)
WHERE 'Part' IN labels(part) OR 'Component' IN labels(part)
OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
RETURN person.name as personnel_name,
       wo.name as work_order,
       asset.name as asset_worked_on,
       collect(DISTINCT part.name) as parts_used,
       count(DISTINCT wo) as total_work_orders
ORDER BY total_work_orders DESC LIMIT 15
```

**Template 3: Maintenance history for [ASSET]? (Auto-detected, boosted confidence)**
```cypher
MATCH (asset:Entity)<-[:PERFORMED_ON]-(wo:Entity)
WHERE toLower(asset.name) CONTAINS toLower('asset_name')
OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
WHERE 'Person' IN labels(person) OR 'Technician' IN labels(person)
OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
OPTIONAL MATCH (wo)-[:HAS_TOTAL_COST|ESTIMATED_COST]->(cost:Entity)
WHERE 'Cost' IN labels(cost) OR 'Value' IN labels(cost)
RETURN asset.name as asset_name,
       wo.name as work_order,
       person.name as technician,
       date.name as work_date,
       cost.name as work_cost
ORDER BY date.name DESC LIMIT 20
```

**Template 4: Work order details? (Auto-detected for work order queries)**
```cypher
MATCH (wo:Entity)
WHERE toLower(wo.name) CONTAINS toLower('work_order_name')
AND ('WorkOrder' IN labels(wo) OR 'Task' IN labels(wo) OR 'Activity' IN labels(wo))
OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
WHERE 'Person' IN labels(person) OR 'Technician' IN labels(person)
OPTIONAL MATCH (wo)-[:PERFORMED_ON]->(asset:Entity)
WHERE 'Equipment' IN labels(asset) OR 'Machine' IN labels(asset)
OPTIONAL MATCH (wo)-[:USED_PART|REQUIRES|INCLUDES]->(part:Entity)
WHERE 'Part' IN labels(part) OR 'Component' IN labels(part)
OPTIONAL MATCH (wo)-[:HAS_TOTAL_COST|ESTIMATED_COST]->(cost:Entity)
WHERE 'Cost' IN labels(cost) OR 'Value' IN labels(cost)
OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR|ISSUED_ON]->(date:Entity)
WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
RETURN wo.name as work_order_name,
       collect(DISTINCT person.name) as assigned_personnel,
       collect(DISTINCT asset.name) as assets_affected,
       collect(DISTINCT part.name) as parts_materials,
       collect(DISTINCT cost.name) as costs,
       collect(DISTINCT date.name) as important_dates
LIMIT 10
```

**Template 5: Who approved work orders? (Auto-detected for approval queries)**
```cypher
MATCH (wo:Entity)
WHERE toLower(wo.name) CONTAINS toLower('work_order')
OR toLower(wo.name) CONTAINS 'work'
OR toLower(wo.name) CONTAINS 'wo'
OPTIONAL MATCH (wo)-[:APPROVED_BY]->(approver:Entity)
WHERE 'Person' IN labels(approver) OR 'Manager' IN labels(approver)
OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
WHERE 'Person' IN labels(person) OR 'Technician' IN labels(person)
OPTIONAL MATCH (wo)-[:PERFORMED_ON]->(asset:Entity)
OPTIONAL MATCH (wo)-[:HAS_STATUS|STATUS_OF]->(status:Entity)
WHERE 'Status' IN labels(status) OR 'State' IN labels(status)
RETURN wo.name as work_order,
       approver.name as approved_by,
       person.name as assigned_to,
       asset.name as target_asset,
       status.name as current_status
ORDER BY wo.name LIMIT 15
```

**Universal Fallback Patterns (Lower Priority, confidence 0.7):**
For non-maintenance queries, use enhanced universal patterns:

**Cost Analysis Pattern:**
```cypher
MATCH (entity:Entity)
WHERE toLower(entity.name) CONTAINS toLower('entity_name')
AND ('Invoice' IN labels(entity) OR 'Project' IN labels(entity) OR 'WorkOrder' IN labels(entity) OR 'Service' IN labels(entity))
OPTIONAL MATCH (entity)-[:HAS_TOTAL_COST|ESTIMATED_COST|RENTAL_COST|COST_OF_SERVICE]->(cost:Entity)
WHERE 'Cost' IN labels(cost) OR 'Value' IN labels(cost) OR 'Amount' IN labels(cost)
OPTIONAL MATCH (entity)-[:BILLED_TO|INVOICED_FOR|ASSIGNED_TO|RELATED_TO]->(party:Entity)
WHERE 'Company' IN labels(party) OR 'Person' IN labels(party) OR 'Vendor' IN labels(party)
RETURN entity.name as item_name,
       collect(DISTINCT cost.name) as cost_breakdown,
       collect(DISTINCT party.name) as associated_parties
LIMIT 15
```

**Location/Facility Pattern:**
```cypher
MATCH (location:Entity)
WHERE toLower(location.name) CONTAINS toLower('location_name')
AND ('Location' IN labels(location) OR 'Building' IN labels(location) OR 'Site' IN labels(location) OR 'Facility' IN labels(location))
OPTIONAL MATCH (location)-[:LOCATED_AT|CONTAINS|HOUSES]->(asset:Entity)
WHERE 'Equipment' IN labels(asset) OR 'Vehicle' IN labels(asset) OR 'Instrument' IN labels(asset)
OPTIONAL MATCH (location)-[:PERFORMED_AT|SCHEDULED_AT]->(activity:Entity)
WHERE 'Activity' IN labels(activity) OR 'Service' IN labels(activity) OR 'WorkOrder' IN labels(activity)
RETURN location.name as location_name,
       collect(DISTINCT asset.name) as assets_at_location,
       collect(DISTINCT activity.name) as activities,
       count(DISTINCT asset) as asset_count
LIMIT 15
```

**PATTERN SELECTION INTELLIGENCE:**
The system automatically:
1. **Detects work-order questions** using keyword analysis
2. **Boosts work-order pattern confidence** by +0.3 for maintenance questions
3. **Selects highest scoring pattern** based on question analysis
4. **Falls back to universal patterns** for non-maintenance queries

**Entity Matching Rules:**
- If pre-linked entity provided: Use exact match `WHERE toLower(node.name) = toLower('canonical_name')`
- If not pre-linked: Use fuzzy match `WHERE toLower(node.name) CONTAINS toLower('mention')`

**EXAMPLES WITH PATTERN DETECTION:**

Question: "Who maintains compressor unit 101?" 
→ Detected as: Work-order question (contains "maintains" + "compressor")
→ Selected Pattern: Work-Order-Centric Maintenance (confidence 0.9 + 0.3 boost = 1.2)
→ Generated Query: Uses Template 1

Question: "What is the cost breakdown for project alpha?"
→ Detected as: Cost analysis question (contains "cost breakdown" + "project")  
→ Selected Pattern: Universal Cost Analysis (confidence 0.7)
→ Generated Query: Uses Cost Analysis Pattern

Question: "Tell me about building 5 assets"
→ Detected as: Location question (contains "building" + "assets")
→ Selected Pattern: Universal Location Analysis (confidence 0.7)  
→ Generated Query: Uses Location/Facility Pattern

**Return Rules:**
- Enclose query in ```cypher ... ```
- Return "NO_QUERY_GENERATED" if question cannot be answered
- Use LIMIT for broad queries
- Return properties (.name) not full nodes
- **ALWAYS prioritize work-order patterns for maintenance questions**

**CONFIDENCE SCORING AWARENESS:**
The system scores patterns based on:
- Base pattern confidence (0.9 for work-order, 0.7 for universal)
- Question category match (+0.4 for work-order questions)
- Parameter relevance (+0.3 for matching parameters)
- Keyword overlap (up to +0.2)

NEVER use UNION - it causes syntax errors. Instead use OR conditions in WHERE clauses.
Always leverage the pattern intelligence - the system knows which patterns work best for each question type.
"""

# User prompt template for Cypher generation (ENHANCED)
GENERATE_USER_TEMPLATE = """You are a Neo4j expert with universal pattern awareness. Given an input question and potentially pre-linked entities, create a syntactically correct Cypher query to run.

The system has intelligent pattern detection that automatically identifies work-order vs. universal patterns based on question analysis. 

PATTERN DETECTION AWARENESS:
- Work-order patterns (confidence 0.9) for maintenance questions
- Universal patterns (confidence 0.7) for general queries  
- Automatic confidence boosting for relevant question types
- Smart parameter extraction and template filling

Use the provided schema and few-shot examples to guide your query generation.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!

Schema:
{schema}

Few-shot Examples:
{fewshot_examples}

User Input:
{structured_input}

Cypher query:"""

# --- ENHANCED Prompts for Handling Empty Results with Pattern Awareness ---

EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT = """
You are a meticulous Cypher query analyst with knowledge of universal pattern systems. Your objective is to diagnose why a given Cypher query returned zero results when executed against a graph database with the specified schema.

**PATTERN SYSTEM AWARENESS:**
The system uses intelligent pattern detection with:
- Work-order-centric patterns (high confidence 0.9) for maintenance queries
- Universal fallback patterns (confidence 0.7) for general queries
- Automatic pattern selection and confidence boosting
- Smart parameter extraction and template filling

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Based *only* on the user's question, the executed Cypher query, and the schema, determine the *single most likely* reason for the empty result. Choose exclusively from these options:

1.  **QUERY_MISMATCH**: The query's structure (nodes, relationships, properties, filters) does not accurately reflect the semantic intent of the user's question given the schema. For example, the query looks for the wrong relationship type, filters on an inappropriate property, or misunderstands the core entities involved in the question.
2.  **DATA_ABSENT**: The query *does* accurately represent the user's question according to the schema, but the specific data requested simply does not exist in the graph. The query structure is appropriate for the question, but the graph lacks the necessary nodes or relationships satisfying the query's conditions.
3.  **PATTERN_MISMATCH**: The query doesn't follow the expected work-order-centric patterns for maintenance questions, or uses the wrong universal pattern for the question type.
4.  **AMBIGUOUS**: The user's question is too vague, unclear, or open to multiple interpretations, making it impossible to definitively determine if the query was mismatched or if the data is truly absent based *only* on the provided information.

**Output Format:**
Respond with ONLY ONE of the reason codes: `QUERY_MISMATCH`, `DATA_ABSENT`, `PATTERN_MISMATCH`, or `AMBIGUOUS`. Do not provide any explanation or justification.
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

# --- ENHANCED Prompts for Revising Queries with Pattern Awareness ---

REVISE_EMPTY_RESULT_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query generator with deep knowledge of universal pattern systems and work-order-centric data models. You are given a user question, a graph schema, and an initial Cypher query that returned zero results.

**PATTERN SYSTEM KNOWLEDGE:**
- Work-order patterns use: (wo)-[:ASSIGNED_TO]->(person), (wo)-[:PERFORMED_ON]->(asset)
- Universal patterns cover: cost analysis, location queries, temporal analysis, etc.
- Maintenance questions should use work-order patterns with confidence boosting
- Pattern selection is based on question analysis and keyword detection

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Rewrite the original Cypher query to create a *revised* query that better aligns with the user's question, the provided schema, and the expected pattern system.

**Revision Guidelines:**
1.  **Pattern Alignment:** 
    - For maintenance questions ("who maintains", "what work", "assigned to"), use work-order patterns
    - For cost questions, use cost analysis patterns
    - For location questions, use facility/location patterns
    - For inventory questions, use stock/inventory patterns
2.  **Work-Order Pattern Structure:**
    - Use (wo:Entity)-[:ASSIGNED_TO]->(person:Entity) for personnel assignments
    - Use (wo:Entity)-[:PERFORMED_ON]->(asset:Entity) for asset maintenance
    - Use (wo:Entity)-[:APPROVED_BY]->(approver:Entity) for approvals
3.  **Universal Pattern Structure:**
    - Use appropriate entity types and relationship patterns
    - Include proper OPTIONAL MATCH clauses for related data
    - Apply correct label filtering
4.  **Consider Flexibility:**
    * Make filters less strict (use `CONTAINS` instead of exact matches)
    * Use proper label filtering (`'Person' IN labels(person)`)
    * Include relevant OPTIONAL MATCH patterns
5.  **Schema Adherence:** Ensure the revised query is valid according to the provided schema
6.  **No Revision Case:** If the original query was already optimal, output `NO_REVISION`

**Output Format:**
- If a revision is possible, output *only* the revised Cypher query enclosed in triple backticks: ```cypher\n[REVISED QUERY]\n```
- If no revision is appropriate, output *only* the exact text: `NO_REVISION`
- Do not include any explanations or comments outside the query block or the `NO_REVISION` text.

**Pattern-Aware Revision Examples:**

Original Query (maintenance question but wrong pattern):
```cypher
MATCH (asset:Entity)-[:MAINTAINED_BY]->(person:Entity)
WHERE asset.name CONTAINS 'compressor'
RETURN person.name
```

Revised Query (correct work-order pattern):
```cypher
MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity),
      (wo)-[:PERFORMED_ON]->(asset:Entity)
WHERE toLower(asset.name) CONTAINS 'compressor'
RETURN person.name as technician, wo.name as work_order
LIMIT 15
```

Original Query (cost question but wrong approach):
```cypher
MATCH (n:Entity) WHERE n.name CONTAINS 'project' RETURN n
```

Revised Query (correct cost analysis pattern):
```cypher
MATCH (entity:Entity)
WHERE toLower(entity.name) CONTAINS 'project'
AND ('Project' IN labels(entity) OR 'WorkOrder' IN labels(entity))
OPTIONAL MATCH (entity)-[:HAS_TOTAL_COST|ESTIMATED_COST]->(cost:Entity)
WHERE 'Cost' IN labels(cost) OR 'Value' IN labels(cost)
RETURN entity.name as project_name,
       collect(DISTINCT cost.name) as cost_breakdown
LIMIT 15
```
"""

REVISE_EMPTY_RESULT_USER_PROMPT = """
**Schema:**
{schema}

**User Question:**
{question}

**Original Cypher Query (Returned 0 results, needs pattern-aware revision):**
```cypher
{cypher}
```

**Revised Cypher Query or NO_REVISION:**
"""

# ========================================================================
# 🎯 NEW: PATTERN-AWARE FEW-SHOT EXAMPLES
# ========================================================================

def get_pattern_aware_few_shot_examples():
    """
    Get few-shot examples that demonstrate pattern awareness and proper query structure.
    These examples show the system how to use work-order vs universal patterns correctly.
    """
    return [
        {
            "question": "Who maintains compressor unit 101?",
            "approach": "work_order_pattern",
            "pattern_confidence": 0.9,
            "cypher": """MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity),
                               (wo)-[:PERFORMED_ON]->(asset:Entity)
                        WHERE toLower(asset.name) CONTAINS 'compressor unit 101'
                        RETURN person.name as technician, 
                               wo.name as work_order
                        LIMIT 15""",
            "explanation": "Work-order-centric maintenance pattern (highest priority for maintenance questions)"
        },
        {
            "question": "What work is assigned to john smith?",
            "approach": "work_order_pattern",
            "pattern_confidence": 0.9,
            "cypher": """MATCH (person:Entity)<-[:ASSIGNED_TO]-(wo:Entity)-[:PERFORMED_ON]->(asset:Entity)
                        WHERE toLower(person.name) CONTAINS 'john smith'
                        AND ('Person' IN labels(person) OR 'Technician' IN labels(person))
                        RETURN wo.name as work_order,
                               asset.name as asset_worked_on,
                               count(DISTINCT wo) as total_work_orders
                        ORDER BY total_work_orders DESC LIMIT 15""",
            "explanation": "Personnel work assignment pattern (work-order centric)"
        },
        {
            "question": "Tell me about work order wo2024001",
            "approach": "work_order_pattern",
            "pattern_confidence": 0.9,
            "cypher": """MATCH (wo:Entity)
                        WHERE toLower(wo.name) CONTAINS 'wo2024001'
                        AND ('WorkOrder' IN labels(wo) OR 'Task' IN labels(wo))
                        OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
                        OPTIONAL MATCH (wo)-[:PERFORMED_ON]->(asset:Entity)
                        OPTIONAL MATCH (wo)-[:APPROVED_BY]->(approver:Entity)
                        RETURN wo.name as work_order,
                               person.name as assigned_to,
                               asset.name as target_asset,
                               approver.name as approved_by
                        LIMIT 10""",
            "explanation": "Work order detailed analysis pattern"
        },
        {
            "question": "What is the maintenance history for pump 205?",
            "approach": "work_order_pattern",
            "pattern_confidence": 0.9,
            "cypher": """MATCH (asset:Entity)<-[:PERFORMED_ON]-(wo:Entity)
                        WHERE toLower(asset.name) CONTAINS 'pump 205'
                        OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
                        OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                        WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
                        RETURN asset.name as asset_name,
                               wo.name as work_order,
                               person.name as technician,
                               date.name as work_date
                        ORDER BY date.name DESC LIMIT 20""",
            "explanation": "Asset maintenance history pattern (temporal work-order analysis)"
        },
        {
            "question": "What is the cost breakdown for project alpha?",
            "approach": "universal_pattern",
            "pattern_confidence": 0.7,
            "cypher": """MATCH (entity:Entity)
                        WHERE toLower(entity.name) CONTAINS 'project alpha'
                        AND ('Project' IN labels(entity) OR 'WorkOrder' IN labels(entity))
                        OPTIONAL MATCH (entity)-[:HAS_TOTAL_COST|ESTIMATED_COST]->(cost:Entity)
                        WHERE 'Cost' IN labels(cost) OR 'Value' IN labels(cost)
                        OPTIONAL MATCH (entity)-[:BILLED_TO|INVOICED_FOR]->(party:Entity)
                        WHERE 'Company' IN labels(party) OR 'Person' IN labels(party)
                        RETURN entity.name as item_name,
                               collect(DISTINCT cost.name) as cost_breakdown,
                               collect(DISTINCT party.name) as associated_parties
                        LIMIT 15""",
            "explanation": "Universal cost analysis pattern (non-maintenance query)"
        },
        {
            "question": "What assets are in building 5?",
            "approach": "universal_pattern",
            "pattern_confidence": 0.7,
            "cypher": """MATCH (location:Entity)
                        WHERE toLower(location.name) CONTAINS 'building 5'
                        AND ('Location' IN labels(location) OR 'Building' IN labels(location))
                        OPTIONAL MATCH (location)-[:LOCATED_AT|CONTAINS|HOUSES]->(asset:Entity)
                        WHERE 'Equipment' IN labels(asset) OR 'Vehicle' IN labels(asset)
                        RETURN location.name as location_name,
                               collect(DISTINCT asset.name) as assets_at_location,
                               count(DISTINCT asset) as asset_count
                        LIMIT 15""",
            "explanation": "Universal location and facility analysis pattern"
        },
        {
            "question": "What spare parts do we have in stock?",
            "approach": "universal_pattern",
            "pattern_confidence": 0.7,
            "cypher": """MATCH (item:Entity)
                        WHERE ('Part' IN labels(item) OR 'Component' IN labels(item) OR 'SparePart' IN labels(item))
                        OPTIONAL MATCH (item)-[:HAS_QUANTITY|STOCK_LEVEL]->(qty:Entity)
                        WHERE 'Value' IN labels(qty) OR 'Integer' IN labels(qty)
                        OPTIONAL MATCH (item)-[:LOCATED_AT|STORED_IN]->(location:Entity)
                        WHERE 'Location' IN labels(location) OR 'Warehouse' IN labels(location)
                        RETURN item.name as part_name,
                               collect(DISTINCT qty.name) as quantities,
                               collect(DISTINCT location.name) as storage_locations
                        ORDER BY item.name LIMIT 20""",
            "explanation": "Universal inventory and stock level analysis pattern"
        }
    ]

# ========================================================================
# 🎯 PATTERN CONFIDENCE EXPLANATION PROMPT
# ========================================================================

PATTERN_EXPLANATION_PROMPT = """
The system uses intelligent pattern detection with the following confidence scoring:

**Work-Order Patterns (High Confidence: 0.9)**
- Work-Order-Centric Maintenance: Who maintains what equipment
- Personnel Work Assignment: What work orders are assigned to whom
- Work Order Detailed Analysis: Complete work order details  
- Asset Maintenance History: Chronological work performed on equipment
- Work Order Status/Approval: Work order workflow analysis

**Universal Patterns (Standard Confidence: 0.7)**
- Cost Analysis: Financial breakdown and cost relationships
- Location/Facility: Assets and activities by location
- Inventory/Stock: Parts and materials analysis
- Personnel Assignment: General work assignments
- Temporal Analysis: Time-based queries
- Hierarchical Relationships: Containment and structure
- Vendor/Service: Supplier and service provider analysis

**Confidence Boosting:**
- Work-order patterns receive +0.3 confidence boost for maintenance questions
- Questions with maintenance keywords trigger work-order pattern preference
- Parameter matching adds +0.3 confidence per matched parameter
- Category relevance adds up to +0.4 confidence

**Pattern Selection Logic:**
1. Detect question type (maintenance vs. general)
2. Calculate pattern confidence scores
3. Apply boosting for relevant patterns
4. Select highest scoring pattern
5. Generate query using selected pattern template

This ensures maintenance questions use work-order patterns while general questions use appropriate universal patterns.
"""

# ========================================================================
# 🎯 EXPORT ALL ENHANCED PROMPTS
# ========================================================================

__all__ = [
    'MAIN_SYSTEM_PROMPT',
    'MAIN_USER_PROMPT',
    'ENTITY_RESOLUTION_SYSTEM_PROMPT',
    'get_entity_resolution_user_prompt',
    'RELATIONSHIP_INFERENCE_SYSTEM_PROMPT',
    'get_relationship_inference_user_prompt',
    'WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT',
    'get_within_community_inference_user_prompt',
    'TEXT_TO_CYPHER_SYSTEM_PROMPT',
    'GENERATE_USER_TEMPLATE',
    'EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT',
    'EVALUATE_EMPTY_RESULT_USER_PROMPT',
    'REVISE_EMPTY_RESULT_SYSTEM_PROMPT',
    'REVISE_EMPTY_RESULT_USER_PROMPT',
    'get_pattern_aware_few_shot_examples',
    'PATTERN_EXPLANATION_PROMPT'
]