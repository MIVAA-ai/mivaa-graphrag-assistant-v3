# universal_asset_patterns.py - INDUSTRY-AGNOSTIC ASSET MANAGEMENT PATTERNS WITH ENHANCED INTELLIGENCE

import logging
import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from cypher_generation_improvement_strategy import (
    QueryPattern, QueryCategory, QueryComplexity
)

logger = logging.getLogger(__name__)


class IndustryType(Enum):
    """Supported industry types for pattern optimization"""
    OIL_GAS = "oil_gas"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    LOGISTICS = "logistics"
    UTILITIES = "utilities"
    CONSTRUCTION = "construction"
    MINING = "mining"
    AEROSPACE = "aerospace"
    AUTOMOTIVE = "automotive"
    GENERIC = "generic"


@dataclass
class DomainContext:
    """Context information about the current domain/industry"""
    industry: IndustryType
    common_entities: List[str]
    common_relationships: List[str]
    domain_keywords: List[str]
    cost_patterns: List[str]
    time_patterns: List[str]


class UniversalPatternLibrary:
    """
    Industry-agnostic pattern library that adapts to any domain.
    ENHANCED with work-order-centric patterns for asset management.
    Uses actual schema structure with flexible templates.
    """

    @staticmethod
    def get_universal_patterns() -> List[QueryPattern]:
        """
        Returns universal patterns with WORK-ORDER AWARENESS as TOP PRIORITY.
        Work-order patterns come first to ensure they get selected for maintenance questions.
        """
        return [

            # ========================================================================
            # 🎯 WORK-ORDER-CENTRIC PATTERNS (HIGHEST PRIORITY - TOP OF LIST)
            # ========================================================================

            # 1. WORK-ORDER-CENTRIC MAINTENANCE PATTERN (TOP PRIORITY)
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity),
                                  (wo)-[:PERFORMED_ON]->(asset:Entity)
                           WHERE toLower(asset.name) CONTAINS toLower('{asset}')
                           OPTIONAL MATCH (wo)-[r3:USED_PART|REQUIRES|INCLUDES]->(part:Entity)
                           WHERE 'Part' IN labels(part) OR 'Component' IN labels(part)
                           OPTIONAL MATCH (wo)-[r4:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                           WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
                           RETURN asset.name as asset_name,
                                  wo.name as work_order,
                                  person.name as assigned_personnel,
                                  collect(DISTINCT part.name) as parts_used,
                                  collect(DISTINCT date.name) as work_dates,
                                  type(r3) as part_relationship,
                                  type(r4) as date_relationship
                           ORDER BY wo.name LIMIT 15""",
                description="Work-order-centric asset maintenance analysis - who maintains what equipment",
                example_nl="Who maintains [ASSET] and what work was performed?",
                example_cypher="MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity), (wo)-[:PERFORMED_ON]->(asset:Entity) WHERE asset.name CONTAINS 'compressor' RETURN person.name, wo.name",
                parameters=["asset"],
                confidence_score=0.9
            ),

            # 2. PERSONNEL WORK ASSIGNMENT PATTERN (HIGH PRIORITY)
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (person:Entity)<-[:ASSIGNED_TO]-(wo:Entity)-[:PERFORMED_ON]->(asset:Entity)
                           WHERE toLower(person.name) CONTAINS toLower('{person}')
                           AND ('Person' IN labels(person) OR 'Technician' IN labels(person))
                           OPTIONAL MATCH (wo)-[r2:USED_PART|REQUIRES]->(part:Entity)
                           WHERE 'Part' IN labels(part) OR 'Component' IN labels(part)
                           OPTIONAL MATCH (wo)-[r3:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                           WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
                           RETURN person.name as personnel_name,
                                  wo.name as work_order,
                                  asset.name as asset_worked_on,
                                  collect(DISTINCT part.name) as parts_used,
                                  collect(DISTINCT date.name) as work_dates,
                                  count(DISTINCT wo) as total_work_orders
                           ORDER BY total_work_orders DESC, wo.name LIMIT 15""",
                description="Personnel work assignment analysis - what work orders are assigned to whom",
                example_nl="What work is assigned to [PERSON] and what assets do they maintain?",
                example_cypher="MATCH (person:Entity)<-[:ASSIGNED_TO]-(wo:Entity)-[:PERFORMED_ON]->(asset:Entity) WHERE person.name CONTAINS 'john' RETURN wo.name, asset.name",
                parameters=["person"],
                confidence_score=0.9
            ),

            # 3. WORK ORDER DETAILED ANALYSIS PATTERN
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.COMPLEX,
                template="""MATCH (wo:Entity)
                           WHERE toLower(wo.name) CONTAINS toLower('{work_order}')
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
                           ORDER BY wo.name LIMIT 10""",
                description="Comprehensive work order analysis - complete work order details",
                example_nl="Tell me everything about work order [WORK_ORDER]",
                example_cypher="MATCH (wo:Entity)-[:ASSIGNED_TO]->(person:Entity), (wo)-[:PERFORMED_ON]->(asset:Entity) WHERE wo.name CONTAINS 'wo2024001' RETURN wo.name, person.name, asset.name",
                parameters=["work_order"],
                confidence_score=0.9
            ),

            # 4. ASSET MAINTENANCE HISTORY PATTERN
            QueryPattern(
                category=QueryCategory.TEMPORAL,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (asset:Entity)<-[:PERFORMED_ON]-(wo:Entity)
                           WHERE toLower(asset.name) CONTAINS toLower('{asset}')
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
                                  cost.name as work_cost,
                                  count(DISTINCT wo) as total_work_orders
                           ORDER BY date.name DESC, wo.name LIMIT 20""",
                description="Asset maintenance history - chronological work performed on equipment",
                example_nl="What is the maintenance history for [ASSET]?",
                example_cypher="MATCH (asset:Entity)<-[:PERFORMED_ON]-(wo:Entity)-[:COMPLETED_ON]->(date:Entity) WHERE asset.name CONTAINS 'compressor' RETURN wo.name, date.name ORDER BY date.name DESC",
                parameters=["asset"],
                confidence_score=0.9
            ),

            # 5. WORK ORDER STATUS AND APPROVAL PATTERN
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (wo:Entity)
                           WHERE toLower(wo.name) CONTAINS toLower('{work_order}')
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
                           ORDER BY wo.name LIMIT 15""",
                description="Work order status and approval workflow analysis",
                example_nl="What is the status of work orders and who approved them?",
                example_cypher="MATCH (wo:Entity)-[:APPROVED_BY]->(approver:Entity), (wo)-[:ASSIGNED_TO]->(person:Entity) RETURN wo.name, approver.name, person.name",
                parameters=["work_order"],
                confidence_score=0.9
            ),

            # 6. WORK ORDER BY APPROVAL STATUS PATTERN
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (approver:Entity)<-[:APPROVED_BY]-(wo:Entity)
                           WHERE toLower(approver.name) CONTAINS toLower('{approver}')
                           AND ('Person' IN labels(approver) OR 'Manager' IN labels(approver))
                           OPTIONAL MATCH (wo)-[:ASSIGNED_TO]->(person:Entity)
                           WHERE 'Person' IN labels(person) OR 'Technician' IN labels(person)
                           OPTIONAL MATCH (wo)-[:PERFORMED_ON]->(asset:Entity)
                           OPTIONAL MATCH (wo)-[:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                           WHERE 'Date' IN labels(date) OR 'Time' IN labels(date)
                           RETURN approver.name as approved_by,
                                  wo.name as work_order,
                                  person.name as assigned_to,
                                  asset.name as target_asset,
                                  date.name as work_date,
                                  count(DISTINCT wo) as total_approved
                           ORDER BY date.name DESC, wo.name LIMIT 15""",
                description="Work orders approved by specific personnel",
                example_nl="What work orders has [APPROVER] approved?",
                example_cypher="MATCH (approver:Entity)<-[:APPROVED_BY]-(wo:Entity)-[:ASSIGNED_TO]->(person:Entity) WHERE approver.name CONTAINS 'mike' RETURN wo.name, person.name",
                parameters=["approver"],
                confidence_score=0.9
            ),

            # ========================================================================
            # 🔄 UNIVERSAL PATTERNS (ENHANCED FALLBACK PATTERNS)
            # ========================================================================

            # 7. UNIVERSAL ASSET MAINTENANCE PATTERN - ENHANCED FALLBACK
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (asset:Entity)
                           WHERE toLower(asset.name) CONTAINS toLower('{asset}')
                           OPTIONAL MATCH (asset)-[r1:ASSIGNED_TO|PERFORMED_BY|USED_IN|RELATED_TO]->(work:Entity)
                           WHERE 'WorkOrder' IN labels(work) 
                              OR 'Task' IN labels(work)
                              OR 'Activity' IN labels(work)
                              OR 'Service' IN labels(work)
                           OPTIONAL MATCH (work)-[r2:ASSIGNED_TO|PERFORMED_BY|OPERATED_BY]->(person:Entity)
                           WHERE 'Person' IN labels(person)
                              OR 'Technician' IN labels(person)
                              OR 'Operator' IN labels(person)
                           OPTIONAL MATCH (work)-[r3:USED_PART|INCLUDES|REQUIRES]->(part:Entity)
                           WHERE 'Part' IN labels(part)
                              OR 'Component' IN labels(part)
                              OR 'Material' IN labels(part)
                           RETURN asset.name as asset_name,
                                  collect(DISTINCT work.name) as maintenance_work,
                                  collect(DISTINCT person.name) as personnel,
                                  collect(DISTINCT part.name) as parts_materials,
                                  collect(DISTINCT type(r1)) as work_relationships
                           ORDER BY asset.name LIMIT 20""",
                description="Universal asset maintenance workflow analysis (fallback pattern)",
                example_nl="What maintenance work was performed on [ASSET] and what materials were used?",
                example_cypher="MATCH (asset:Entity)-[:RELATED_TO]->(work:Entity)-[:USED_PART]->(part:Entity) WHERE asset.name CONTAINS $asset RETURN asset.name, work.name, part.name",
                parameters=["asset"],
                confidence_score=0.7
            ),

            # 8. UNIVERSAL COST ANALYSIS PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.AGGREGATION,
                complexity=QueryComplexity.COMPLEX,
                template="""MATCH (entity:Entity)
                           WHERE toLower(entity.name) CONTAINS toLower('{entity}')
                           AND ('Invoice' IN labels(entity)
                                OR 'Project' IN labels(entity)
                                OR 'WorkOrder' IN labels(entity)
                                OR 'Service' IN labels(entity))
                           OPTIONAL MATCH (entity)-[r1:HAS_TOTAL_COST|ESTIMATED_COST|RENTAL_COST|COST_OF_SERVICE]->(cost:Entity)
                           WHERE 'Cost' IN labels(cost)
                              OR 'Value' IN labels(cost)
                              OR 'Amount' IN labels(cost)
                           OPTIONAL MATCH (entity)-[r2:BILLED_TO|INVOICED_FOR|ASSIGNED_TO|RELATED_TO]->(party:Entity)
                           WHERE 'Company' IN labels(party)
                              OR 'Person' IN labels(party)
                              OR 'Vendor' IN labels(party)
                           OPTIONAL MATCH (entity)-[r3:ISSUED_ON|COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                           WHERE 'Date' IN labels(date)
                              OR 'Time' IN labels(date)
                           RETURN entity.name as item_name,
                                  collect(DISTINCT cost.name) as cost_breakdown,
                                  collect(DISTINCT party.name) as associated_parties,
                                  collect(DISTINCT date.name) as relevant_dates
                           ORDER BY entity.name LIMIT 15""",
                description="Universal cost analysis and financial breakdown",
                example_nl="What is the total cost breakdown for [PROJECT/INVOICE] and who are the associated parties?",
                example_cypher="MATCH (proj:Entity)-[:HAS_TOTAL_COST]->(cost:Entity), (proj)-[:BILLED_TO]->(party:Entity) WHERE proj.name CONTAINS $entity RETURN proj.name, cost.name, party.name",
                parameters=["entity"],
                confidence_score=0.7
            ),

            # 9. UNIVERSAL PERSONNEL & ASSIGNMENT PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.MAINTENANCE_WORKFLOW,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (person:Entity)
                           WHERE toLower(person.name) CONTAINS toLower('{person}')
                           AND ('Person' IN labels(person)
                                OR 'Technician' IN labels(person)
                                OR 'Operator' IN labels(person)
                                OR 'Employee' IN labels(person))
                           OPTIONAL MATCH (person)-[r1:ASSIGNED_TO|PERFORMED_BY|OPERATED_BY|MANAGED_BY]->(work:Entity)
                           WHERE 'WorkOrder' IN labels(work)
                              OR 'Task' IN labels(work)
                              OR 'Activity' IN labels(work)
                              OR 'Project' IN labels(work)
                           OPTIONAL MATCH (work)-[r2:PERFORMED_ON|USED_IN|RELATED_TO]->(asset:Entity)
                           WHERE 'Equipment' IN labels(asset)
                              OR 'Vehicle' IN labels(asset)
                              OR 'Instrument' IN labels(asset)
                              OR 'Machine' IN labels(asset)
                           OPTIONAL MATCH (work)-[r3:COMPLETED_ON|SCHEDULED_FOR]->(date:Entity)
                           WHERE 'Date' IN labels(date)
                              OR 'Time' IN labels(date)
                           RETURN person.name as personnel_name,
                                  collect(DISTINCT work.name) as assigned_work,
                                  collect(DISTINCT asset.name) as assets_worked_on,
                                  collect(DISTINCT date.name) as work_dates,
                                  count(DISTINCT work) as total_assignments
                           ORDER BY total_assignments DESC LIMIT 15""",
                description="Universal personnel assignment and work history analysis",
                example_nl="What work has been assigned to [PERSON] and what assets have they worked on?",
                example_cypher="MATCH (person:Entity)-[:ASSIGNED_TO]->(work:Entity)-[:PERFORMED_ON]->(asset:Entity) WHERE person.name CONTAINS $person RETURN person.name, work.name, asset.name",
                parameters=["person"],
                confidence_score=0.7
            ),

            # 10. UNIVERSAL INVENTORY & STOCK PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.ENTITY_LOOKUP,
                complexity=QueryComplexity.SIMPLE,
                template="""MATCH (item:Entity)
                           WHERE ('Part' IN labels(item)
                                  OR 'Component' IN labels(item)
                                  OR 'Material' IN labels(item)
                                  OR 'SparePart' IN labels(item)
                                  OR 'Inventory' IN labels(item))
                           AND toLower(item.name) CONTAINS toLower('{item_type}')
                           OPTIONAL MATCH (item)-[r1:HAS_QUANTITY|STOCK_LEVEL|QUANTITY_USED]->(qty:Entity)
                           WHERE 'Value' IN labels(qty)
                              OR 'Measurement' IN labels(qty)
                              OR 'Integer' IN labels(qty)
                           OPTIONAL MATCH (item)-[r2:LOCATED_AT|STORED_IN|LOCATED_IN]->(location:Entity)
                           WHERE 'Location' IN labels(location)
                              OR 'Building' IN labels(location)
                              OR 'Site' IN labels(location)
                           OPTIONAL MATCH (item)-[r3:USED_IN|REQUIRED_FOR|PART_OF]->(equipment:Entity)
                           WHERE 'Equipment' IN labels(equipment)
                              OR 'Vehicle' IN labels(equipment)
                              OR 'Machine' IN labels(equipment)
                           RETURN item.name as item_name,
                                  collect(DISTINCT qty.name) as quantities,
                                  collect(DISTINCT location.name) as storage_locations,
                                  collect(DISTINCT equipment.name) as used_in_equipment
                           ORDER BY item.name LIMIT 20""",
                description="Universal inventory and stock level analysis",
                example_nl="What is our current stock level for [ITEM_TYPE] and where are they stored?",
                example_cypher="MATCH (item:Entity)-[:HAS_QUANTITY]->(qty:Entity), (item)-[:LOCATED_AT]->(loc:Entity) WHERE item.name CONTAINS $item_type RETURN item.name, qty.name, loc.name",
                parameters=["item_type"],
                confidence_score=0.7
            ),

            # 11. UNIVERSAL LOCATION & FACILITY PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.ASSET_HIERARCHY,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (location:Entity)
                           WHERE toLower(location.name) CONTAINS toLower('{location}')
                           AND ('Location' IN labels(location)
                                OR 'Building' IN labels(location)
                                OR 'Site' IN labels(location)
                                OR 'Facility' IN labels(location))
                           OPTIONAL MATCH (location)-[r1:LOCATED_AT|CONTAINS|HOUSES]->(asset:Entity)
                           WHERE 'Equipment' IN labels(asset)
                              OR 'Vehicle' IN labels(asset)
                              OR 'Instrument' IN labels(asset)
                           OPTIONAL MATCH (location)-[r2:PERFORMED_AT|SCHEDULED_AT]->(activity:Entity)
                           WHERE 'Activity' IN labels(activity)
                              OR 'Service' IN labels(activity)
                              OR 'WorkOrder' IN labels(activity)
                           OPTIONAL MATCH (location)-[r3:ASSIGNED_TO|MANAGED_BY]->(personnel:Entity)
                           WHERE 'Person' IN labels(personnel)
                              OR 'Technician' IN labels(personnel)
                           RETURN location.name as location_name,
                                  collect(DISTINCT asset.name) as assets_at_location,
                                  collect(DISTINCT activity.name) as activities,
                                  collect(DISTINCT personnel.name) as assigned_personnel,
                                  count(DISTINCT asset) as asset_count,
                                  count(DISTINCT activity) as activity_count
                           ORDER BY asset_count DESC, activity_count DESC LIMIT 15""",
                description="Universal location and facility analysis",
                example_nl="What assets and activities are associated with [LOCATION]?",
                example_cypher="MATCH (loc:Entity)-[:CONTAINS]->(asset:Entity), (loc)-[:PERFORMED_AT]->(activity:Entity) WHERE loc.name CONTAINS $location RETURN loc.name, asset.name, activity.name",
                parameters=["location"],
                confidence_score=0.7
            ),

            # 12. UNIVERSAL TEMPORAL ANALYSIS PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.TEMPORAL,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (entity:Entity)-[r]->(date:Entity)
                           WHERE ('Date' IN labels(date)
                                  OR 'Time' IN labels(date))
                           AND toLower(date.name) CONTAINS toLower('{time_period}')
                           OPTIONAL MATCH (entity)-[r2:HAS_TOTAL_COST|ESTIMATED_COST|RENTAL_COST]->(cost:Entity)
                           WHERE 'Cost' IN labels(cost)
                              OR 'Value' IN labels(cost)
                              OR 'Amount' IN labels(cost)
                           OPTIONAL MATCH (entity)-[r3:RELATED_TO|ASSIGNED_TO|PERFORMED_BY]->(associated:Entity)
                           WHERE 'Service' IN labels(associated)
                              OR 'WorkOrder' IN labels(associated)
                              OR 'Activity' IN labels(associated)
                           RETURN entity.name as item_name,
                                  date.name as time_period,
                                  collect(DISTINCT cost.name) as costs,
                                  collect(DISTINCT associated.name) as associated_items,
                                  type(r) as time_relationship
                           ORDER BY entity.name LIMIT 20""",
                description="Universal temporal and time-based analysis",
                example_nl="What costs and activities occurred during [TIME_PERIOD]?",
                example_cypher="MATCH (entity:Entity)-[:COMPLETED_ON]->(date:Entity)-[:COST_OF_SERVICE]->(cost:Entity) WHERE date.name CONTAINS $time_period RETURN entity.name, cost.name",
                parameters=["time_period"],
                confidence_score=0.7
            ),

            # 13. UNIVERSAL HIERARCHICAL RELATIONSHIP PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.ASSET_HIERARCHY,
                complexity=QueryComplexity.COMPLEX,
                template="""MATCH (parent:Entity)
                           WHERE toLower(parent.name) CONTAINS toLower('{parent}')
                           OPTIONAL MATCH (parent)-[r:CONTAINS|PART_OF|INCLUDES|LOCATED_IN*1..2]->(child:Entity)
                           OPTIONAL MATCH (child)-[r2:HAS_TOTAL_COST|ESTIMATED_COST]->(cost:Entity)
                           WHERE 'Cost' IN labels(cost)
                              OR 'Value' IN labels(cost)
                              OR 'Amount' IN labels(cost)
                           RETURN parent.name as parent_entity,
                                  child.name as child_entity,
                                  collect(DISTINCT type(r)) as relationship_path,
                                  collect(DISTINCT cost.name) as associated_costs,
                                  labels(parent) as parent_types,
                                  labels(child) as child_types
                           ORDER BY child.name LIMIT 20""",
                description="Universal hierarchical relationship and containment analysis",
                example_nl="What items are contained within [PARENT] and what are their associated costs?",
                example_cypher="MATCH (parent:Entity)-[:CONTAINS*1..3]->(child:Entity) WHERE parent.name CONTAINS $parent RETURN parent.name, child.name",
                parameters=["parent"],
                confidence_score=0.7
            ),

            # 14. UNIVERSAL SERVICE & VENDOR PATTERN - ENHANCED
            QueryPattern(
                category=QueryCategory.AGGREGATION,
                complexity=QueryComplexity.MODERATE,
                template="""MATCH (vendor:Entity)
                           WHERE toLower(vendor.name) CONTAINS toLower('{vendor}')
                           AND ('Company' IN labels(vendor)
                                OR 'Vendor' IN labels(vendor)
                                OR 'Supplier' IN labels(vendor)
                                OR 'Contractor' IN labels(vendor))
                           OPTIONAL MATCH (vendor)-[r1:PROVIDES|SUPPLIES|CONTRACTS]->(service:Entity)
                           WHERE 'Service' IN labels(service)
                              OR 'WorkOrder' IN labels(service)
                           OPTIONAL MATCH (service)-[r2:COST_OF_SERVICE|HAS_TOTAL_COST|BILLED_TO]->(cost:Entity)
                           WHERE 'Value' IN labels(cost)
                              OR 'Invoice' IN labels(cost)
                              OR 'Cost' IN labels(cost)
                           OPTIONAL MATCH (service)-[r3:PERFORMED_AT|LOCATED_AT]->(location:Entity)
                           WHERE 'Location' IN labels(location)
                              OR 'Site' IN labels(location)
                           RETURN vendor.name as vendor_name,
                                  collect(DISTINCT service.name) as services_provided,
                                  collect(DISTINCT cost.name) as service_costs,
                                  collect(DISTINCT location.name) as service_locations,
                                  count(DISTINCT service) as service_count
                           ORDER BY service_count DESC LIMIT 15""",
                description="Universal vendor and service provider analysis",
                example_nl="What services has [VENDOR] provided and what are the associated costs?",
                example_cypher="MATCH (vendor:Entity)-[:PROVIDES]->(service:Entity)-[:COST_OF_SERVICE]->(cost:Entity) WHERE vendor.name CONTAINS $vendor RETURN vendor.name, service.name, cost.name",
                parameters=["vendor"],
                confidence_score=0.7
            )
        ]


class DomainAdaptivePatternMatcher:
    """
    Adaptive pattern matcher that learns from your actual schema and adjusts patterns accordingly.
    ENHANCED with intelligent question classification and work-order awareness.
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.detected_domain = None
        self.schema_analysis = {}
        self.entity_type_mapping = {}
        self.relationship_type_mapping = {}
        self.adapted_patterns = None

    # ========================================================================
    # 🎯 ENHANCED QUESTION CLASSIFICATION SYSTEM
    # ========================================================================

    def _detect_work_order_question(self, question: str) -> bool:
        """
        ENHANCED: Better work order detection with exclusions for other pattern types
        """
        question_lower = question.lower()

        # EXCLUSIONS - These should NOT use work-order patterns
        exclusion_keywords = [
            # Cost/Financial queries
            'invoice', 'cost', 'amount', 'total', 'price', 'payment', 'bill', 'charge',
            'due', 'owed', 'balance', 'expense', 'budget', 'financial',

            # Inventory/Stock queries
            'stock', 'inventory', 'quantity', 'parts', 'spare', 'material',
            'warehouse', 'storage', 'supply',

            # Location/Facility queries (without maintenance context)
            'building', 'location', 'facility', 'site', 'floor', 'room',

            # Specification/Technical queries
            'flow rate', 'pressure', 'capacity', 'specification', 'model',
            'serial number', 'manufacturer'
        ]

        # First check exclusions
        if any(keyword in question_lower for keyword in exclusion_keywords):
            # But allow if there's strong maintenance context
            maintenance_context = ['maintenance', 'repair', 'work order', 'technician', 'assigned']
            if not any(context in question_lower for context in maintenance_context):
                return False

        # INCLUSIONS - Strong work-order indicators
        work_order_keywords = [
            'work order', 'wo2024', 'wo2025', 'afe-', 'task', 'activity',
            'maintenance', 'repair', 'service', 'fix', 'inspect', 'check',
            'assigned to', 'performed on', 'maintained by', 'technician',
            'who maintains', 'what work', 'maintenance history', 'approved by',
            'authorization', 'approve', 'signed'
        ]

        # Check for work order keywords
        if any(keyword in question_lower for keyword in work_order_keywords):
            return True

        # Personnel + equipment questions (usually maintenance)
        personnel_equipment_patterns = [
            'who maintains', 'who works on', 'who is assigned',
            'what work', 'maintenance for', 'repairs on', 'service on'
        ]

        if any(pattern in question_lower for pattern in personnel_equipment_patterns):
            return True

        # Equipment + personnel in same question (but not specification queries)
        equipment_words = ['compressor', 'pump', 'motor', 'equipment', 'machine', 'unit', 'device']
        personnel_words = ['technician', 'person', 'employee', 'worker', 'operator']

        has_equipment = any(word in question_lower for word in equipment_words)
        has_personnel = any(word in question_lower for word in personnel_words)

        if has_equipment and has_personnel:
            # But not if it's asking about specifications
            spec_words = ['flow rate', 'pressure', 'capacity', 'gpm', 'psi', 'hp']
            if not any(spec in question_lower for spec in spec_words):
                return True

        return False

    def _detect_cost_question(self, question: str) -> bool:
        """Detect cost/financial analysis questions"""
        question_lower = question.lower()

        cost_indicators = [
            'cost', 'amount', 'total', 'price', 'payment', 'invoice',
            'bill', 'charge', 'due', 'owed', 'balance', 'expense',
            'budget', 'financial', 'breakdown', 'how much'
        ]

        return any(indicator in question_lower for indicator in cost_indicators)

    def _detect_inventory_question(self, question: str) -> bool:
        """Detect inventory/stock questions"""
        question_lower = question.lower()

        inventory_indicators = [
            'stock', 'inventory', 'quantity', 'parts', 'spare', 'material',
            'warehouse', 'storage', 'supply', 'available', 'in stock',
            'stock level', 'quantity on hand'
        ]

        return any(indicator in question_lower for indicator in inventory_indicators)

    def _detect_location_question(self, question: str) -> bool:
        """Detect location/facility questions"""
        question_lower = question.lower()

        # Strong location indicators
        location_indicators = [
            'building', 'location', 'facility', 'site', 'floor', 'room',
            'where', 'located', 'housed', 'stored', 'placed'
        ]

        # But exclude if it's about maintenance at a location
        maintenance_context = ['maintenance', 'repair', 'work', 'assigned', 'performed']
        has_location = any(indicator in question_lower for indicator in location_indicators)
        has_maintenance = any(context in question_lower for context in maintenance_context)

        return has_location and not has_maintenance

    def _detect_specification_question(self, question: str) -> bool:
        """Detect equipment specification questions"""
        question_lower = question.lower()

        spec_indicators = [
            'flow rate', 'pressure', 'capacity', 'gpm', 'psi', 'hp',
            'horsepower', 'specification', 'model', 'serial number',
            'manufacturer', 'rating', 'performance'
        ]

        return any(indicator in question_lower for indicator in spec_indicators)

    def _classify_question_type(self, question: str) -> str:
        """Classify question into primary type for pattern routing"""

        # Priority order matters - check most specific first
        if self._detect_cost_question(question):
            return 'cost'
        elif self._detect_work_order_question(question):
            return 'work_order'
        elif self._detect_inventory_question(question):
            return 'inventory'
        elif self._detect_location_question(question):
            return 'location'
        elif self._detect_specification_question(question):
            return 'specification'
        else:
            return 'general'

    # ========================================================================
    # 🎯 ENHANCED PARAMETER EXTRACTION SYSTEM
    # ========================================================================

    def _extract_asset_from_question(self, question: str) -> Optional[str]:
        """Enhanced asset/equipment extraction with better pattern matching"""
        question_lower = question.lower()

        # Priority 1: Specific equipment with identifiers
        equipment_with_id_patterns = [
            r'\b((?:compressor|pump|motor|engine|turbine|generator|boiler|tank)\s+(?:unit\s+)?(?:\w+\d+|\d+\w*|\w+-\d+))\b',
            r'\b((?:equipment|machine|device|unit)\s+(?:\w+\d+|\d+\w*|\w+-\d+))\b'
        ]

        for pattern in equipment_with_id_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).strip()

        # Priority 2: Equipment types with specifications
        spec_patterns = [
            r'\b(pump)\s+(?:has|with)\s+(?:a\s+)?flow\s+rate',
            r'\b(motor)\s+(?:has|with)\s+(?:a\s+)?(?:power|hp|horsepower)',
            r'\b(compressor)\s+(?:has|with)\s+(?:a\s+)?(?:pressure|psi)',
            r'\b(tank)\s+(?:has|with)\s+(?:a\s+)?(?:capacity|volume)'
        ]

        for pattern in spec_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)

        # Priority 3: General equipment mentions
        equipment_patterns = [
            r'\b(compressor\s+unit\s+\w+)\b',
            r'\b(pump\s+\w+)\b',
            r'\b(motor\s+\w+)\b',
            r'\b((?:compressor|pump|motor|engine|turbine|generator|boiler|tank|equipment|machine|device))\b'
        ]

        for pattern in equipment_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).strip()

        # Priority 4: Flow rate or specification mentions (for equipment queries)
        if any(word in question_lower for word in ['flow rate', 'gpm', 'pressure', 'psi', 'horsepower', 'hp']):
            # Look for equipment type before the specification
            before_spec = re.search(
                r'\b(compressor|pump|motor|engine|turbine|generator|boiler|tank)\b.*(?:flow rate|gpm|pressure|psi|horsepower|hp)',
                question_lower)
            if before_spec:
                return before_spec.group(1)

        return None

    def _extract_work_order_from_question(self, question: str) -> Optional[str]:
        """Enhanced work order extraction with better AFE and WO pattern matching"""
        question_lower = question.lower()

        # Priority 1: AFE patterns (Authorization for Expenditure)
        afe_patterns = [
            r'\b(afe-\d{4}-\d{4})\b',  # AFE-2024-0078
            r'\b(afe\s*\d{4}\s*\d{4})\b',  # AFE 2024 0078
            r'\b(afe-\d{4}-\w+)\b',  # AFE-2024-ABC
            r'\b(afe\s+\w+-\d+)\b'  # AFE ABC-123
        ]

        for pattern in afe_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).replace(' ', '-')  # Normalize spacing

        # Priority 2: Work Order patterns
        wo_patterns = [
            r'\b(wo-?\d{4}-?\d{3,4})\b',  # WO2024001, WO-2024-001
            r'\b(work\s+order\s+\w+\d+)\b',  # work order WO2024001
            r'\b(wo\d{4,})\b',  # WO2024001
            r'\b(task\s+\w+\d+)\b'  # task T2024001
        ]

        for pattern in wo_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).upper()  # Normalize to uppercase

        # Priority 3: Generic work identifiers
        work_id_patterns = [
            r'\b((?:work|task|job|activity)\s+(?:\w+\d+|\d+\w*))\b',
            r'\b(\w+\d{4,})\b'  # Any identifier with 4+ digits
        ]

        for pattern in work_id_patterns:
            match = re.search(pattern, question_lower)
            if match:
                candidate = match.group(1)
                # Validate it looks like a work identifier
                if any(keyword in candidate for keyword in ['wo', 'afe', 'work', 'task', 'job']) or re.search(r'\d{4}',
                                                                                                              candidate):
                    return candidate.upper()

        return None

    def _extract_entity_from_question(self, question: str) -> Optional[str]:
        """Enhanced general entity extraction for invoices, projects, etc."""
        question_lower = question.lower()

        # Priority 1: Invoice patterns
        invoice_patterns = [
            r'\b(inv-\d{4}-\d{4})\b',  # INV-2024-7825
            r'\b(invoice\s+\w+\d+)\b',  # invoice INV123
            r'\b(inv\s*\d{4}\s*\d+)\b'  # INV 2024 7825
        ]

        for pattern in invoice_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).upper()

        # Priority 2: Project patterns
        project_patterns = [
            r'\b(project\s+\w+)\b',  # project alpha
            r'\b(proj-\w+)\b',  # proj-alpha
            r'\b(\w+\s+project)\b'  # alpha project
        ]

        for pattern in project_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)

        # Priority 3: Document/Reference patterns
        doc_patterns = [
            r'\b([A-Z]{2,4}-\d{4}-\d{3,4})\b',  # ABC-2024-123
            r'\b(\w+\d{4}\w+)\b',  # ABC2024DEF
            r'\b(ref\s+\w+)\b',  # ref ABC123
            r'\b(document\s+\w+)\b'  # document ABC123
        ]

        for pattern in doc_patterns:
            match = re.search(pattern, question, re.IGNORECASE)  # Case sensitive for IDs
            if match:
                return match.group(1)

        return None

    def _extract_approver_from_question(self, question: str) -> Optional[str]:
        """Enhanced approver extraction for approval workflow questions"""
        question_lower = question.lower()

        # Priority 1: Direct approver mentions with title
        approver_with_title_patterns = [
            r'\b((?:manager|supervisor|director|lead)\s+\w+\s+\w+)\b',  # Manager John Smith
            r'\b(\w+\s+\w+\s+(?:manager|supervisor|director|lead))\b',  # John Smith Manager
            r'\b((?:mr|ms|dr)\.?\s+\w+\s+\w+)\b'  # Mr. John Smith
        ]

        for pattern in approver_with_title_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).title()  # Proper case

        # Priority 2: Names in approval context
        approval_context_patterns = [
            r'approved\s+by\s+(\w+\s+\w+)',  # approved by John Smith
            r'(\w+\s+\w+)\s+approved',  # John Smith approved
            r'authorization\s+from\s+(\w+\s+\w+)',  # authorization from John Smith
            r'signed\s+by\s+(\w+\s+\w+)'  # signed by John Smith
        ]

        for pattern in approval_context_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).title()

        # Priority 3: General name patterns (only if approval context exists)
        if any(word in question_lower for word in ['approved', 'authorization', 'authorize', 'signed', 'approve']):
            name_patterns = [
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # John Smith
                r'\b([A-Z][a-z]+\s+[A-Z]\.)\b'  # John S.
            ]

            for pattern in name_patterns:
                match = re.search(pattern, question)
                if match:
                    return match.group(1)

        return None

    def _extract_cost_entity_from_question(self, question: str) -> Optional[str]:
        """Extract entity for cost/financial queries"""
        question_lower = question.lower()

        # Priority 1: Invoice references
        invoice_result = self._extract_entity_from_question(question)
        if invoice_result and any(word in invoice_result.lower() for word in ['inv', 'invoice']):
            return invoice_result

        # Priority 2: Project cost references
        project_patterns = [
            r'\b(project\s+\w+)\b',
            r'\b(\w+\s+project)\b',
            r'\b(phase\s+\w+)\b'
        ]

        for pattern in project_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)

        # Priority 3: General cost entity
        cost_entity_patterns = [
            r'cost\s+(?:of\s+|for\s+)?(\w+(?:\s+\w+)?)',
            r'amount\s+(?:for\s+|due\s+on\s+)?(\w+(?:\s+\w+)?)',
            r'total\s+(?:for\s+)?(\w+(?:\s+\w+)?)'
        ]

        for pattern in cost_entity_patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)

        return None

    def _extract_person_from_question(self, question: str) -> Optional[str]:
        """Extract person mention from question"""
        name_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # Full names
            r'\b([A-Z][a-z]+\s+[A-Z]\.)\b'  # Name with initial
        ]

        for pattern in name_patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1)
        return None

    def _extract_location_from_question(self, question: str) -> Optional[str]:
        """Extract location mention from question"""
        location_patterns = [
            r'\b(building\s+\w+)\b',
            r'\b(\w+\s+(?:site|facility|location|building))\b',
            r'\b(\w+\s+\w+\s+(?:pump|battery|flare))\b'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_vendor_from_question(self, question: str) -> Optional[str]:
        """Extract vendor/company mention from question"""
        company_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+(?:Inc|Corp|LLC|Ltd))?)\b',
            r'\b(National\s+\w+\s+\w+)\b'
        ]

        for pattern in company_patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1)
        return None

    def _extract_item_type_from_question(self, question: str) -> Optional[str]:
        """Extract item/part type from inventory-related questions"""
        inventory_patterns = [
            r'\b(spare\s+parts?)\b',
            r'\b(components?)\b',
            r'\b(materials?)\b',
            r'\b(parts?)\b',
            r'\b(inventory)\b',
            r'\b(stock)\b',
            r'\b(\w+\s+parts?)\b'  # "pump parts", "motor parts"
        ]

        for pattern in inventory_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_time_period_from_question(self, question: str) -> Optional[str]:
        """Extract time period from temporal questions"""
        time_patterns = [
            r'\b(this\s+month)\b',
            r'\b(next\s+month)\b',
            r'\b(last\s+month)\b',
            r'\b(this\s+year)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(\d{4})\b',  # Year
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # Date
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(recent|current|upcoming)\b'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_parent_from_question(self, question: str) -> Optional[str]:
        """Extract parent entity for hierarchical questions"""
        hierarchy_patterns = [
            r'\b(building\s+\w+)\b',
            r'\b(facility\s+\w+)\b',
            r'\b(site\s+\w+)\b',
            r'\b(plant\s+\w+)\b',
            r'\b(\w+\s+building)\b',
            r'\b(\w+\s+facility)\b'
        ]

        for pattern in hierarchy_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_parameters_from_question(self, question: str, parameters: List[str]) -> Dict[str, str]:
        """Enhanced parameter extraction with improved accuracy"""
        extracted = {}
        question_lower = question.lower()

        for param in parameters:
            if param == 'asset':
                value = self._extract_asset_from_question(question)
                extracted[param] = value or 'equipment'

            elif param == 'person':
                value = self._extract_person_from_question(question)
                extracted[param] = value or 'person'

            elif param == 'work_order':
                value = self._extract_work_order_from_question(question)
                extracted[param] = value or 'work'

            elif param == 'approver':
                value = self._extract_approver_from_question(question)
                extracted[param] = value or 'manager'

            elif param == 'entity':
                # Check if it's a cost-related query first
                if any(word in question_lower for word in ['cost', 'amount', 'total', 'price', 'invoice']):
                    value = self._extract_cost_entity_from_question(question)
                else:
                    value = self._extract_entity_from_question(question)
                extracted[param] = value or 'entity'

            elif param == 'location':
                value = self._extract_location_from_question(question)
                extracted[param] = value or 'location'

            elif param == 'item_type':
                value = self._extract_item_type_from_question(question)
                extracted[param] = value or 'part'

            elif param == 'time_period':
                value = self._extract_time_period_from_question(question)
                extracted[param] = value or 'date'

            elif param == 'parent':
                value = self._extract_parent_from_question(question)
                extracted[param] = value or 'parent'

            elif param == 'vendor':
                value = self._extract_vendor_from_question(question)
                extracted[param] = value or 'vendor'

        return extracted

    # ========================================================================
    # 🎯 ENHANCED PATTERN SELECTION SYSTEM
    # ========================================================================

    def _select_best_pattern_for_question(self, question: str) -> Optional[QueryPattern]:
        """
        ENHANCED: Smart pattern selection with question type detection
        """
        # Get all patterns
        all_patterns = self.adapted_patterns or UniversalPatternLibrary.get_universal_patterns()

        # STEP 1: Detect question type and route to appropriate patterns
        question_type = self._classify_question_type(question)

        # STEP 2: Filter patterns based on question type
        if question_type == 'cost':
            # Use cost analysis patterns
            relevant_patterns = [p for p in all_patterns if p.category == QueryCategory.AGGREGATION]
        elif question_type == 'inventory':
            # Use inventory patterns
            relevant_patterns = [p for p in all_patterns if p.category == QueryCategory.ENTITY_LOOKUP]
        elif question_type == 'location':
            # Use location patterns
            relevant_patterns = [p for p in all_patterns if p.category == QueryCategory.ASSET_HIERARCHY]
        elif question_type == 'work_order':
            # Use work-order patterns with boosting
            relevant_patterns = self._boost_work_order_patterns(all_patterns, question)
        else:
            # Use all patterns with standard scoring
            relevant_patterns = all_patterns

        # STEP 3: Score patterns
        pattern_scores = []

        for pattern in relevant_patterns:
            score = self._calculate_pattern_score(pattern, question, question_type)
            pattern_scores.append((pattern, score))

        # STEP 4: Return highest scoring pattern
        if pattern_scores:
            pattern_scores.sort(key=lambda x: x[1], reverse=True)
            best_pattern, best_score = pattern_scores[0]

            logger.debug(
                f"Pattern selection: {best_pattern.description} (score: {best_score:.3f}, type: {question_type})")

            if best_score > 0.3:
                return best_pattern

        return None

    def _calculate_pattern_score(self, pattern: QueryPattern, question: str, question_type: str) -> float:
        """Enhanced pattern scoring with question type awareness"""
        score = 0.0
        question_lower = question.lower()

        # Base confidence from pattern
        score += pattern.confidence_score * 0.3

        # MAJOR BOOST: Question type alignment
        if question_type == 'work_order' and self._is_work_order_pattern(pattern):
            score += 0.5  # Strong boost for work-order alignment
        elif question_type == 'cost' and pattern.category == QueryCategory.AGGREGATION:
            score += 0.5  # Strong boost for cost alignment
        elif question_type == 'inventory' and pattern.category == QueryCategory.ENTITY_LOOKUP:
            score += 0.5  # Strong boost for inventory alignment
        elif question_type == 'location' and pattern.category == QueryCategory.ASSET_HIERARCHY:
            score += 0.5  # Strong boost for location alignment

        # Enhanced category matching with question type
        category_match_bonus = {
            ('work_order', QueryCategory.MAINTENANCE_WORKFLOW): 0.4,
            ('cost', QueryCategory.AGGREGATION): 0.4,
            ('inventory', QueryCategory.ENTITY_LOOKUP): 0.4,
            ('location', QueryCategory.ASSET_HIERARCHY): 0.4,
            ('specification', QueryCategory.ENTITY_LOOKUP): 0.3,
            ('general', QueryCategory.MAINTENANCE_WORKFLOW): 0.2,
        }

        bonus = category_match_bonus.get((question_type, pattern.category), 0.0)
        score += bonus

        # Parameter relevance
        param_score = 0.0
        for param in pattern.parameters:
            param_variations = {
                'asset': ['asset', 'equipment', 'machine', 'device', 'pump', 'compressor', 'motor'],
                'person': ['person', 'technician', 'operator', 'worker', 'employee', 'staff'],
                'entity': ['invoice', 'project', 'item', 'entity', 'object'],
                'work_order': ['work order', 'wo', 'afe', 'task', 'activity'],
                'approver': ['manager', 'supervisor', 'approver', 'approved', 'authorization'],
                'location': ['location', 'building', 'site', 'facility', 'area', 'zone'],
                'vendor': ['vendor', 'supplier', 'company', 'contractor'],
                'item_type': ['part', 'component', 'material', 'spare', 'inventory'],
                'time_period': ['month', 'year', 'date', 'time', 'period'],
                'parent': ['building', 'facility', 'site', 'parent']
            }

            variations = param_variations.get(param, [param])
            param_matches = sum(1 for variation in variations if variation in question_lower)
            if param_matches > 0:
                param_score += 0.2

        score += min(param_score, 0.4)  # Cap parameter bonus

        # Question complexity alignment
        complexity_bonus = {
            QueryComplexity.SIMPLE: 0.1 if len(question.split()) <= 8 else 0.0,
            QueryComplexity.MODERATE: 0.1 if 8 < len(question.split()) <= 15 else 0.0,
            QueryComplexity.COMPLEX: 0.1 if len(question.split()) > 15 else 0.0
        }

        score += complexity_bonus.get(pattern.complexity, 0.0)

        return score

    def _boost_work_order_patterns(self, patterns: List[QueryPattern], question: str) -> List[QueryPattern]:
        """
        ENHANCED: Boost work-order patterns only for true maintenance questions
        """
        if not self._detect_work_order_question(question):
            return patterns

        boosted_patterns = []
        for pattern in patterns:
            if self._is_work_order_pattern(pattern):
                # Create a copy with boosted confidence
                boosted_pattern = QueryPattern(
                    category=pattern.category,
                    complexity=pattern.complexity,
                    template=pattern.template,
                    description=pattern.description,
                    example_nl=pattern.example_nl,
                    example_cypher=pattern.example_cypher,
                    parameters=pattern.parameters,
                    confidence_score=min(pattern.confidence_score + 0.4, 1.0)  # Increased boost
                )
                boosted_patterns.append(boosted_pattern)
            else:
                boosted_patterns.append(pattern)

        return boosted_patterns

    def _is_work_order_pattern(self, pattern: QueryPattern) -> bool:
        """
        AUGMENTED: Check if a pattern is work-order-centric
        """
        template = pattern.template.upper()
        description = pattern.description.lower()

        # Look for work-order relationship patterns
        work_order_indicators = [
            '[:ASSIGNED_TO]->',
            '[:PERFORMED_ON]->',
            '<-[:ASSIGNED_TO]-',
            '<-[:PERFORMED_ON]-',
            '[:APPROVED_BY]->',
            '<-[:APPROVED_BY]-',
            'work order', 'work-order', 'maintenance', 'personnel'
        ]

        return any(indicator in template or indicator in description for indicator in work_order_indicators)

    # ========================================================================
    # 🔍 SCHEMA ANALYSIS METHODS
    # ========================================================================

    def analyze_domain_and_schema(self) -> DomainContext:
        """
        Analyze the current Neo4j database to detect domain and create adaptive context.
        """
        logger.info("Analyzing database schema and domain context...")

        try:
            # Get schema information
            schema_info = self._get_comprehensive_schema()

            # Detect industry/domain
            detected_industry = self._detect_industry_type(schema_info)

            # Create adaptive mappings
            entity_mappings = self._create_entity_type_mappings(schema_info)
            relationship_mappings = self._create_relationship_mappings(schema_info)

            # Extract domain-specific keywords
            domain_keywords = self._extract_domain_keywords(schema_info)

            # Create domain context
            domain_context = DomainContext(
                industry=detected_industry,
                common_entities=schema_info.get('frequent_entities', []),
                common_relationships=schema_info.get('frequent_relationships', []),
                domain_keywords=domain_keywords,
                cost_patterns=self._identify_cost_patterns(schema_info),
                time_patterns=self._identify_time_patterns(schema_info)
            )

            self.detected_domain = domain_context
            logger.info(f"Detected domain: {detected_industry.value}")

            return domain_context

        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            # Return generic context as fallback
            return DomainContext(
                industry=IndustryType.GENERIC,
                common_entities=["Entity"],
                common_relationships=["RELATED_TO"],
                domain_keywords=[],
                cost_patterns=["cost", "price", "amount"],
                time_patterns=["date", "time", "when"]
            )

    def _get_comprehensive_schema(self) -> Dict[str, Any]:
        """Get comprehensive schema analysis from Neo4j"""
        schema_info = {
            'node_labels': [],
            'relationship_types': [],
            'frequent_entities': [],
            'frequent_relationships': [],
            'sample_data': []
        }

        try:
            with self.driver.session() as session:
                # Get node labels
                result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                record = result.single()
                if record:
                    schema_info['node_labels'] = record['labels']

                # Get relationship types
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                record = result.single()
                if record:
                    schema_info['relationship_types'] = record['types']

                # Get relationship frequency analysis
                result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN type(r) as rel_type, count(*) as frequency
                    ORDER BY frequency DESC LIMIT 20
                """)
                schema_info['frequent_relationships'] = [record['rel_type'] for record in result]

                # Sample entity names for domain detection
                result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.name as entity_name, labels(n) as entity_labels
                    LIMIT 50
                """)
                schema_info['sample_data'] = [
                    {'name': record['entity_name'], 'labels': record['entity_labels']}
                    for record in result if record['entity_name']
                ]

        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")

        return schema_info

    def _detect_industry_type(self, schema_info: Dict[str, Any]) -> IndustryType:
        """Detect industry type based on schema and data patterns"""
        # Industry-specific keyword patterns
        industry_patterns = {
            IndustryType.OIL_GAS: [
                'well', 'drilling', 'production', 'barrel', 'oil', 'gas', 'refinery',
                'pipeline', 'rig', 'field', 'reservoir', 'crude', 'petroleum'
            ],
            IndustryType.MANUFACTURING: [
                'production', 'assembly', 'factory', 'machine', 'manufacturing',
                'process', 'quality', 'batch', 'line', 'plant', 'facility'
            ],
            IndustryType.HEALTHCARE: [
                'patient', 'medical', 'hospital', 'doctor', 'nurse', 'treatment',
                'diagnosis', 'medication', 'clinic', 'healthcare', 'therapy'
            ],
            IndustryType.LOGISTICS: [
                'shipment', 'delivery', 'transport', 'warehouse', 'cargo',
                'logistics', 'freight', 'distribution', 'supply', 'route'
            ],
            IndustryType.UTILITIES: [
                'power', 'electricity', 'grid', 'utility', 'energy', 'transmission',
                'substation', 'transformer', 'meter', 'consumption'
            ]
        }

        # Analyze sample data for industry keywords
        sample_text = ""
        for item in schema_info.get('sample_data', []):
            if item.get('name'):
                sample_text += item['name'].lower() + " "

        # Analyze labels for industry hints
        labels_text = " ".join(schema_info.get('node_labels', [])).lower()
        relationships_text = " ".join(schema_info.get('relationship_types', [])).lower()
        all_text = sample_text + " " + labels_text + " " + relationships_text

        # Score each industry
        industry_scores = {}
        for industry, keywords in industry_patterns.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            industry_scores[industry] = score

        # Return the highest scoring industry or GENERIC if no clear match
        if industry_scores:
            best_industry = max(industry_scores, key=industry_scores.get)
            if industry_scores[best_industry] > 0:
                return best_industry

        return IndustryType.GENERIC

    def _create_entity_type_mappings(self, schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create mappings for different entity types based on actual schema"""
        labels = schema_info.get('node_labels', [])

        mappings = {
            'equipment_types': [],
            'person_types': [],
            'location_types': [],
            'cost_types': [],
            'work_types': [],
            'part_types': [],
            'date_types': [],
            'service_types': [],
            'asset_types': [],
            'vendor_types': []
        }

        # Map based on common patterns
        for label in labels:
            label_lower = label.lower()

            if any(word in label_lower for word in ['equipment', 'machine', 'device', 'instrument', 'tool']):
                mappings['equipment_types'].append(label_lower)
                mappings['asset_types'].append(label_lower)

            if any(word in label_lower for word in ['person', 'employee', 'technician', 'operator', 'worker', 'staff']):
                mappings['person_types'].append(label_lower)

            if any(word in label_lower for word in ['location', 'site', 'building', 'facility', 'room', 'area']):
                mappings['location_types'].append(label_lower)

            if any(word in label_lower for word in ['cost', 'price', 'amount', 'value', 'money', 'dollar']):
                mappings['cost_types'].append(label_lower)

            if any(word in label_lower for word in ['work', 'order', 'task', 'activity', 'job', 'assignment']):
                mappings['work_types'].append(label_lower)

            if any(word in label_lower for word in ['part', 'component', 'material', 'spare', 'supply']):
                mappings['part_types'].append(label_lower)

            if any(word in label_lower for word in ['date', 'time', 'when', 'schedule']):
                mappings['date_types'].append(label_lower)

            if any(word in label_lower for word in ['service', 'vendor', 'supplier', 'contractor', 'company']):
                mappings['service_types'].append(label_lower)
                mappings['vendor_types'].append(label_lower)

        # Add fallbacks
        for key in mappings:
            if not mappings[key]:
                mappings[key] = ['entity']  # Fallback to generic entity

        self.entity_type_mapping = mappings
        return mappings

    def _create_relationship_mappings(self, schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create relationship type mappings"""
        relationships = schema_info.get('relationship_types', [])

        mappings = {
            'assignment_relationships': [],
            'cost_relationships': [],
            'location_relationships': [],
            'temporal_relationships': [],
            'hierarchy_relationships': [],
            'service_relationships': []
        }

        for rel in relationships:
            rel_lower = rel.lower()

            if any(word in rel_lower for word in ['assigned', 'performed', 'operated', 'managed']):
                mappings['assignment_relationships'].append(rel)

            if any(word in rel_lower for word in ['cost', 'price', 'amount', 'billed', 'invoiced']):
                mappings['cost_relationships'].append(rel)

            if any(word in rel_lower for word in ['located', 'at', 'in', 'on', 'housed']):
                mappings['location_relationships'].append(rel)

            if any(word in rel_lower for word in ['completed', 'scheduled', 'issued', 'due']):
                mappings['temporal_relationships'].append(rel)

            if any(word in rel_lower for word in ['contains', 'part_of', 'includes', 'parent']):
                mappings['hierarchy_relationships'].append(rel)

            if any(word in rel_lower for word in ['provides', 'supplies', 'service', 'contracts']):
                mappings['service_relationships'].append(rel)

        self.relationship_type_mapping = mappings
        return mappings

    def _extract_domain_keywords(self, schema_info: Dict[str, Any]) -> List[str]:
        """Extract domain-specific keywords from the data"""
        keywords = set()

        # Extract from sample entity names
        for item in schema_info.get('sample_data', []):
            if item.get('name'):
                name_parts = re.findall(r'\b\w+\b', item['name'].lower())
                keywords.update(name_parts)

        # Filter out common words and keep domain-specific terms
        common_words = {'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by', 'a', 'an'}
        filtered_keywords = [kw for kw in keywords if kw not in common_words and len(kw) > 2]

        return filtered_keywords[:50]  # Limit to top 50

    def _identify_cost_patterns(self, schema_info: Dict[str, Any]) -> List[str]:
        """Identify cost-related patterns in the data"""
        cost_patterns = ['cost', 'price', 'amount', 'total', 'value']

        # Look for currency symbols and cost-related terms in sample data
        for item in schema_info.get('sample_data', []):
            if item.get('name'):
                name = item['name'].lower()
                if any(symbol in name for symbol in ['$', '€', '£', 'dollar', 'euro', 'pound']):
                    cost_patterns.append(name.split()[0])  # Add the term before currency

        return list(set(cost_patterns))

    def _identify_time_patterns(self, schema_info: Dict[str, Any]) -> List[str]:
        """Identify time-related patterns in the data"""
        time_patterns = ['date', 'time', 'when', 'schedule', 'due', 'completed']

        # Look for date patterns in sample data
        date_patterns = [
            r'\d{4}', r'\d{1,2}/\d{1,2}',
            r'january|february|march|april|may|june|july|august|september|october|november|december',
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        ]

        for item in schema_info.get('sample_data', []):
            if item.get('name'):
                name = item['name'].lower()
                for pattern in date_patterns:
                    if re.search(pattern, name):
                        time_patterns.append(name)
                        break

        return list(set(time_patterns))

    # ========================================================================
    # 🎯 PUBLIC API METHODS
    # ========================================================================

    def get_best_pattern_for_question(self, question: str) -> Optional[QueryPattern]:
        """
        PUBLIC API: Get the best pattern for a given question with enhanced intelligence
        """
        if not self.adapted_patterns:
            self.adapted_patterns = UniversalPatternLibrary.get_universal_patterns()

        return self._select_best_pattern_for_question(question)

    def generate_cypher_from_pattern(self, pattern: QueryPattern, question: str) -> Optional[str]:
        """
        PUBLIC API: Generate Cypher query from pattern and question
        """
        try:
            # Extract parameters from question
            if pattern.parameters:
                extracted_params = self._extract_parameters_from_question(question, pattern.parameters)

                # Fill in template with extracted parameters
                cypher_query = pattern.template
                for param, value in extracted_params.items():
                    cypher_query = cypher_query.replace(f'{{{param}}}', value)

                return cypher_query
            else:
                return pattern.template

        except Exception as e:
            logger.error(f"Error generating Cypher from pattern: {e}")
            return None

    def get_pattern_explanation(self, pattern: QueryPattern) -> str:
        """
        PUBLIC API: Get human-readable explanation of what a pattern does
        """
        explanation = f"Pattern: {pattern.description}\n"
        explanation += f"Category: {pattern.category.value}\n"
        explanation += f"Complexity: {pattern.complexity.value}\n"

        if pattern.example_nl:
            explanation += f"Example Question: {pattern.example_nl}\n"

        if pattern.parameters:
            explanation += f"Required Parameters: {', '.join(pattern.parameters)}\n"

        explanation += f"Confidence Score: {pattern.confidence_score:.2f}"

        return explanation


# ========================================================================
# 🎯 UNIVERSAL ASSET MANAGEMENT ENGINE (ENHANCED)
# ========================================================================

class UniversalAssetManagementEngine:
    """
    Universal asset management engine that adapts to any industry or domain.
    Enhanced with intelligent question classification and work-order awareness.
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.pattern_matcher = DomainAdaptivePatternMatcher(neo4j_driver)
        self.domain_context = None
        self.adapted_patterns = []

        # Initialize domain analysis
        self._initialize_domain_analysis()

    def _initialize_domain_analysis(self):
        """Initialize domain analysis and pattern adaptation"""
        try:
            # Analyze current domain
            self.domain_context = self.pattern_matcher.analyze_domain_and_schema()

            # Get universal patterns
            universal_patterns = UniversalPatternLibrary.get_universal_patterns()

            # Store adapted patterns
            self.adapted_patterns = universal_patterns
            self.pattern_matcher.adapted_patterns = universal_patterns

            logger.info(
                f"Initialized universal engine for {self.domain_context.industry.value} domain with {len(self.adapted_patterns)} patterns")

        except Exception as e:
            logger.error(f"Domain initialization failed: {e}")
            self.domain_context = DomainContext(
                industry=IndustryType.GENERIC,
                common_entities=["Entity"],
                common_relationships=["RELATED_TO"],
                domain_keywords=[],
                cost_patterns=["cost"],
                time_patterns=["date"]
            )

    def generate_adaptive_cypher(self, question: str, linked_entities: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate Cypher using adaptive patterns based on detected domain.
        ENHANCED with intelligent question classification and work-order awareness.
        """
        # Select best pattern for the question
        best_pattern = self.pattern_matcher.get_best_pattern_for_question(question)

        if not best_pattern:
            return {
                'cypher_query': None,
                'confidence_score': 0.0,
                'approach_used': 'no_pattern_match',
                'domain_detected': self.domain_context.industry.value if self.domain_context else 'unknown'
            }

        # Generate Cypher using the selected pattern
        try:
            cypher_query = self.pattern_matcher.generate_cypher_from_pattern(best_pattern, question)

            return {
                'cypher_query': cypher_query,
                'confidence_score': best_pattern.confidence_score,
                'approach_used': 'adaptive_pattern',
                'pattern_used': best_pattern.description,
                'domain_detected': self.domain_context.industry.value if self.domain_context else 'unknown',
                'pattern_category': best_pattern.category.value,
                'question_type': self.pattern_matcher._classify_question_type(question)
            }

        except Exception as e:
            logger.error(f"Pattern template filling failed: {e}")
            return {
                'cypher_query': None,
                'confidence_score': 0.0,
                'approach_used': 'template_error',
                'error': str(e)
            }

    def get_domain_summary(self) -> Dict[str, Any]:
        """Get summary of detected domain and adapted patterns"""
        if not self.domain_context:
            return {'error': 'Domain analysis not completed'}

        return {
            'detected_industry': self.domain_context.industry.value,
            'common_entities': self.domain_context.common_entities[:10],
            'common_relationships': self.domain_context.common_relationships[:10],
            'domain_keywords': self.domain_context.domain_keywords[:15],
            'total_patterns': len(self.adapted_patterns),
            'pattern_categories': [p.category.value for p in self.adapted_patterns],
            'work_order_patterns': len(
                [p for p in self.adapted_patterns if self.pattern_matcher._is_work_order_pattern(p)]),
            'adaptation_successful': True
        }


# ========================================================================
# 🎯 INTEGRATION CLASS FOR EXISTING SYSTEM
# ========================================================================

class UniversalEnhancedGraphRAGQA:
    """
    Universal version of Enhanced GraphRAG QA that adapts to any industry.
    Integrates seamlessly with existing GraphRAG system.
    """

    def __init__(self, base_graphrag_qa):
        self.base_qa = base_graphrag_qa
        self.universal_engine = UniversalAssetManagementEngine(base_graphrag_qa.driver)

        logger.info("Universal Enhanced GraphRAG QA initialized - Industry agnostic with intelligent pattern selection")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Enhanced question answering with universal domain adaptation and intelligent pattern selection"""

        logger.info(f"=== Universal GraphRAG Processing: {question} ===")

        # Step 1: Use base system for entity linking
        potential_mentions = self.base_qa._extract_potential_entities(question)
        linked_entities = self.base_qa._link_entities(potential_mentions) if potential_mentions else {}

        # Step 2: Try universal adaptive Cypher generation with enhanced intelligence
        adaptive_result = self.universal_engine.generate_adaptive_cypher(question, linked_entities)

        cypher_query = adaptive_result.get('cypher_query')
        confidence_score = adaptive_result.get('confidence_score', 0.0)

        # Step 3: Execute query if successful, otherwise fallback to base system
        if cypher_query and confidence_score > 0.5:
            try:
                graph_results = self.base_qa._query_neo4j(cypher_query)
                logger.info(f"Universal pattern query executed: {len(graph_results)} results")
            except Exception as e:
                logger.warning(f"Universal pattern query failed, using base system: {e}")
                return self.base_qa.answer_question(question)
        else:
            logger.info("Universal patterns not applicable, using base system")
            return self.base_qa.answer_question(question)

        # Step 4: Continue with base system for vector search and answer synthesis
        vector_top_k = self.base_qa.llm_config_extra.get("vector_search_top_k", 3)
        similar_chunks = self.base_qa._query_vector_db(question, top_k=vector_top_k)

        context_str = self.base_qa._format_context(graph_results, similar_chunks)
        answer_dict = self.base_qa._synthesize_answer(question, context_str, similar_chunks)

        # Step 5: Add universal enhancement metadata
        answer_dict.update({
            'cypher_query': cypher_query,
            'cypher_confidence': confidence_score,
            'generation_approach': adaptive_result.get('approach_used', 'fallback'),
            'pattern_used': adaptive_result.get('pattern_used'),
            'domain_detected': adaptive_result.get('domain_detected', 'unknown'),
            'pattern_category': adaptive_result.get('pattern_category'),
            'question_type': adaptive_result.get('question_type'),
            'linked_entities': linked_entities,
            'universal_enhancement': {
                'domain_adaptive': True,
                'industry_agnostic': True,
                'work_order_aware': True,
                'intelligent_routing': True,
                'pattern_confidence': confidence_score
            }
        })

        logger.info("=== Universal GraphRAG Processing Complete ===")
        return answer_dict

    def get_domain_info(self) -> Dict[str, Any]:
        """Get information about detected domain and available patterns"""
        return self.universal_engine.get_domain_summary()

    def switch_industry_context(self, industry: IndustryType) -> bool:
        """Manually switch to a specific industry context"""
        try:
            # Update domain context
            self.universal_engine.domain_context.industry = industry
            logger.info(f"Switched to {industry.value} industry context")
            return True
        except Exception as e:
            logger.error(f"Failed to switch industry context: {e}")
            return False


# ========================================================================
# 🎯 UTILITY FUNCTIONS AND FACTORY METHODS
# ========================================================================

def get_universal_pattern_matcher(neo4j_driver):
    """Factory function to create a configured pattern matcher"""
    return DomainAdaptivePatternMatcher(neo4j_driver)


def analyze_question_patterns(questions: List[str]) -> Dict[str, Any]:
    """Analyze a list of questions to understand common patterns"""
    analysis = {
        'total_questions': len(questions),
        'work_order_questions': 0,
        'cost_questions': 0,
        'inventory_questions': 0,
        'location_questions': 0,
        'specification_questions': 0,
        'general_questions': 0,
        'common_keywords': {}
    }

    # Create a temporary matcher for classification
    temp_matcher = DomainAdaptivePatternMatcher(None)

    # Analyze each question
    for question in questions:
        question_type = temp_matcher._classify_question_type(question)

        if question_type == 'work_order':
            analysis['work_order_questions'] += 1
        elif question_type == 'cost':
            analysis['cost_questions'] += 1
        elif question_type == 'inventory':
            analysis['inventory_questions'] += 1
        elif question_type == 'location':
            analysis['location_questions'] += 1
        elif question_type == 'specification':
            analysis['specification_questions'] += 1
        else:
            analysis['general_questions'] += 1

        # Count keywords
        words = question.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                analysis['common_keywords'][word] = analysis['common_keywords'].get(word, 0) + 1

    return analysis


# ========================================================================
# 🎯 MAIN EXECUTION AND TESTING
# ========================================================================

if __name__ == "__main__":
    print("=== Universal Asset Management Pattern System with Enhanced Intelligence ===")
    print("Industry-agnostic patterns with intelligent question classification and work-order awareness\n")

    # Show universal patterns
    universal_patterns = UniversalPatternLibrary.get_universal_patterns()

    print(f"📋 {len(universal_patterns)} Universal Patterns Available:")

    # Show work-order patterns first
    work_order_patterns = [p for p in universal_patterns[:6]]  # First 6 are work-order patterns
    print("\n🎯 Work-Order-Centric Patterns (High Priority):")
    for i, pattern in enumerate(work_order_patterns, 1):
        print(f"\n{i}. {pattern.description}")
        print(f"   Category: {pattern.category.value}")
        print(f"   Confidence: {pattern.confidence_score}")
        print(f"   Example: {pattern.example_nl}")

    # Show universal patterns
    universal_fallback_patterns = universal_patterns[6:]  # Remaining universal patterns
    print(f"\n🔄 Universal Fallback Patterns ({len(universal_fallback_patterns)}):")
    for i, pattern in enumerate(universal_fallback_patterns, 7):
        print(f"\n{i}. {pattern.description}")
        print(f"   Category: {pattern.category.value}")
        print(f"   Confidence: {getattr(pattern, 'confidence_score', 0.7)}")

    print("\n" + "=" * 80)
    print("🎯 Enhanced Features:")
    print("✅ Intelligent Question Classification - Routes questions to optimal patterns")
    print("✅ Enhanced Parameter Extraction - Better entity and ID recognition")
    print("✅ Smart Pattern Selection - Context-aware scoring and routing")
    print("✅ Work-Order Awareness - Prioritizes maintenance workflow patterns")
    print("✅ Cost/Invoice Intelligence - Proper routing for financial queries")
    print("✅ Industry Detection - Automatically detects oil&gas, manufacturing, etc.")
    print("✅ Schema Adaptation - Uses YOUR actual Neo4j labels and relationships")
    print("✅ Universal Compatibility - Works with any asset management domain")

    print("\n⚡ Question Type Detection:")
    print("- Cost/Financial: 'What is the total amount due on invoice INV-2024-7825?'")
    print("- Work Order: 'Who approved the AFE-2024-0078 authorization?'")
    print("- Specification: 'Which pump has a flow rate of 500 GPM?'")
    print("- Maintenance: 'Who maintains compressor unit 101?'")
    print("- Inventory: 'What spare parts do we have in stock?'")

    print("\n✨ The enhanced system automatically:")
    print("- Classifies questions by type (cost, work_order, inventory, location, etc.)")
    print("- Routes to optimal patterns based on question classification")
    print("- Extracts parameters with improved accuracy and format handling")
    print("- Boosts work-order pattern confidence for maintenance questions")
    print("- Provides detailed metadata about pattern selection and confidence")
    print("- Falls back gracefully to universal patterns for edge cases")

    print("\n🚀 Ready for enhanced universal deployment!")
    print("🎯 This version provides significant improvements in:")
    print("   - Parameter extraction accuracy (handles AFE-IDs, invoice numbers, etc.)")
    print("   - Pattern selection intelligence (cost vs maintenance vs inventory)")
    print("   - Question understanding (excludes inappropriate patterns)")
    print("   - Overall query success rate (fewer 'Query returned 0 records')")