#!/usr/bin/env python3
"""
Text Sanitization and Structuring Module for Knowledge Graph Pipeline
Handles OCR error correction, domain standardization, and relationship clarification
with intelligent chunking for large documents.
"""

import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import tiktoken  # For accurate token counting

# Assuming these imports are available in the project structure
try:
    from src.knowledge_graph.llm import call_llm, QuotaError
except ImportError as e:
    logging.error(f"Failed to import LLM modules: {e}")


    def call_llm(*args, **kwargs):
        raise NotImplementedError("call_llm is not available.")

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a text chunk for processing."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: str
    token_count: int
    document_context: str = ""  # Context from surrounding chunks


class TextSanitizer:
    """
    Handles text sanitization and structuring using cost-effective LLM calls
    with intelligent chunking for large documents.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text sanitizer.

        Args:
            config: Configuration dictionary containing LLM and sanitization settings
        """
        self.config = config
        self.sanitization_config = config.get("text_sanitization", {})
        self.llm_config = config.get("llm", {})

        # Chunking parameters
        self.max_tokens_per_chunk = self.sanitization_config.get("max_tokens_per_chunk", 3000)
        self.overlap_tokens = self.sanitization_config.get("overlap_tokens", 200)
        self.min_chunk_tokens = self.sanitization_config.get("min_chunk_tokens", 100)

        # Initialize tokenizer for accurate counting
        try:
            model_name = self.llm_config.get("model", "gpt-4o-mini")
            if "gpt" in model_name.lower():
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # General fallback
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None

        logger.info(f"TextSanitizer initialized with max_tokens_per_chunk={self.max_tokens_per_chunk}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using appropriate tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters for English
            return len(text) // 4

    def should_chunk_text(self, text: str) -> bool:
        """
        Determine if text needs to be chunked based on token count.

        Args:
            text: Input text to evaluate

        Returns:
            True if text should be chunked, False otherwise
        """
        token_count = self.count_tokens(text)
        threshold = self.max_tokens_per_chunk * 0.8  # Use 80% of limit as threshold

        should_chunk = token_count > threshold
        logger.info(f"Text token count: {token_count}, threshold: {threshold}, will_chunk: {should_chunk}")
        return should_chunk

    def create_intelligent_chunks(self, text: str, document_type: str = "unknown") -> List[ChunkInfo]:
        """
        Create intelligent chunks that respect document structure and maintain context.

        Args:
            text: Full text to chunk
            document_type: Type of document (invoice, report, etc.) for context-aware chunking

        Returns:
            List of ChunkInfo objects with text and metadata
        """
        logger.info(f"Creating intelligent chunks for {document_type} document")

        # Define document-specific breakpoints
        breakpoints = self._get_document_breakpoints(document_type)

        # Try structure-aware chunking first
        chunks = self._chunk_by_structure(text, breakpoints)

        # If structure-aware chunking produces chunks that are too large, fall back to token-based
        oversized_chunks = [chunk for chunk in chunks if self.count_tokens(chunk.text) > self.max_tokens_per_chunk]

        if oversized_chunks:
            logger.info(f"Found {len(oversized_chunks)} oversized chunks, applying token-based splitting")
            final_chunks = []
            for chunk in chunks:
                if self.count_tokens(chunk.text) > self.max_tokens_per_chunk:
                    sub_chunks = self._chunk_by_tokens(chunk.text, chunk.start_pos)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            chunks = final_chunks

        # Add context information to chunks
        chunks = self._add_context_to_chunks(chunks, text)

        logger.info(f"Created {len(chunks)} chunks with token counts: {[c.token_count for c in chunks]}")
        return chunks

    def _get_document_breakpoints(self, document_type: str) -> List[str]:
        """Get document-type specific section breakpoints."""
        breakpoint_patterns = {
            "invoice": [
                r"\n\s*#+\s*[A-Z][^#\n]*",  # Headers with #
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",  # ALL CAPS sections
                r"\n\s*\d+\.\s*[A-Z]",  # Numbered sections
                r"\n\s*Invoice\s+Details?", r"\n\s*Customer\s+Information",
                r"\n\s*Service\s+Operations", r"\n\s*Financial\s+Information"
            ],
            "report": [
                r"\n\s*#+\s*[A-Z][^#\n]*",  # Headers
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",  # ALL CAPS sections
                r"\n\s*\d+\.\s*[A-Z]",  # Numbered sections
                r"\n\s*Summary", r"\n\s*Details", r"\n\s*Conclusions?"
            ],
            "default": [
                r"\n\s*#+\s*[A-Z][^#\n]*",  # Generic headers
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",  # ALL CAPS sections
                r"\n\s*\d+\.\s*[A-Z]",  # Numbered sections
            ]
        }
        return breakpoint_patterns.get(document_type, breakpoint_patterns["default"])

    def _chunk_by_structure(self, text: str, breakpoints: List[str]) -> List[ChunkInfo]:
        """Chunk text by document structure using breakpoint patterns."""

        # Find all breakpoint positions
        break_positions = [0]  # Start of document

        for pattern in breakpoints:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            for match in matches:
                break_positions.append(match.start())

        # Add end of document
        break_positions.append(len(text))

        # Remove duplicates and sort
        break_positions = sorted(set(break_positions))

        chunks = []
        for i in range(len(break_positions) - 1):
            start_pos = break_positions[i]
            end_pos = break_positions[i + 1]
            chunk_text = text[start_pos:end_pos].strip()

            if len(chunk_text) > 10:  # Skip very small chunks
                token_count = self.count_tokens(chunk_text)
                chunk_info = ChunkInfo(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    chunk_id=f"struct_chunk_{i}",
                    token_count=token_count
                )
                chunks.append(chunk_info)

        # If no meaningful structure found, fall back to token-based chunking
        if len(chunks) <= 1:
            logger.info("Structure-based chunking found minimal sections, falling back to token-based")
            return self._chunk_by_tokens(text, 0)

        return chunks

    def _chunk_by_tokens(self, text: str, start_offset: int = 0) -> List[ChunkInfo]:
        """Fall back to token-based chunking with overlap."""

        chunks = []
        chunk_size = self.max_tokens_per_chunk - 100  # Leave room for prompt
        overlap_size = self.overlap_tokens

        # Simple sentence-aware chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_tokens = 0
        chunk_start = start_offset
        chunk_idx = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence would exceed limit, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(current_chunk),
                    chunk_id=f"token_chunk_{chunk_idx}",
                    token_count=current_tokens
                )
                chunks.append(chunk_info)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_idx += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens

        # Add final chunk if it has content
        if current_chunk.strip():
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk),
                chunk_id=f"token_chunk_{chunk_idx}",
                token_count=current_tokens
            )
            chunks.append(chunk_info)

        return chunks

    def _get_overlap_text(self, text: str, target_tokens: int) -> str:
        """Extract the last N tokens worth of text for overlap."""
        if not text:
            return ""

        sentences = text.split('. ')
        overlap_text = ""
        current_tokens = 0

        # Build overlap from the end backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                overlap_text = sentence + '. ' + overlap_text if overlap_text else sentence
                current_tokens += sentence_tokens
            else:
                break

        return overlap_text.strip()

    def _add_context_to_chunks(self, chunks: List[ChunkInfo], full_text: str) -> List[ChunkInfo]:
        """Add document context to each chunk for better LLM understanding."""

        # Extract document metadata/header for context
        doc_context = self._extract_document_context(full_text)

        for chunk in chunks:
            chunk.document_context = doc_context

        return chunks

    def _extract_document_context(self, text: str) -> str:
        """Extract key document context (first few lines, headers, etc.)."""
        lines = text.split('\n')[:10]  # First 10 lines usually contain key context
        context_lines = []

        for line in lines:
            line = line.strip()
            if line and len(line) > 5:  # Skip very short lines
                context_lines.append(line)
                if len(context_lines) >= 5:  # Limit context size
                    break

        return ' | '.join(context_lines)

    def sanitize_text(self, text: str, document_type: str = "unknown",
                      requests_session=None) -> str:
        """
        Main function to sanitize and structure text for KG pipeline.

        Args:
            text: Raw OCR text to sanitize
            document_type: Type of document for context-aware processing
            requests_session: Optional session for HTTP connection pooling

        Returns:
            Sanitized and structured text ready for KG extraction
        """
        logger.info(f"Starting text sanitization for {document_type} document (length: {len(text)} chars)")

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return text

        # Quick preprocessing for obvious fixes
        text = self._quick_preprocessing(text)

        # Determine if chunking is needed
        if not self.should_chunk_text(text):
            logger.info("Text is small enough for single LLM call")
            return self._sanitize_single_chunk(text, document_type, requests_session)

        # Handle large text with chunking
        logger.info("Text requires chunking for processing")
        chunks = self.create_intelligent_chunks(text, document_type)
        sanitized_chunks = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (tokens: {chunk.token_count})")
            try:
                sanitized_chunk = self._sanitize_single_chunk(
                    chunk.text, document_type, requests_session, chunk.document_context
                )
                sanitized_chunks.append(sanitized_chunk)

            except QuotaError as e:
                logger.error(f"Quota error on chunk {i + 1}: {e}")
                # For quota errors, return original chunk to avoid breaking pipeline
                sanitized_chunks.append(chunk.text)

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                # For other errors, return original chunk
                sanitized_chunks.append(chunk.text)

        # Combine sanitized chunks
        final_text = self._combine_sanitized_chunks(sanitized_chunks)
        logger.info(f"Text sanitization complete. Output length: {len(final_text)} chars")

        return final_text

    def _quick_preprocessing(self, text: str) -> str:
        """Apply quick rule-based fixes before LLM processing."""

        # Common OCR error patterns for Physical Asset Management documents
        quick_fixes = {
            # Asset identification patterns
            r'\b[Il]D[\s]*(\d+)': r'ID \1',  # Fix ID formatting
            r'\bSER[\s]*#[\s]*([A-Z0-9]+)': r'Serial# \1',  # Serial number formatting
            r'\bWO[\s]*#?[\s]*(\d+)': r'Work Order \1',  # Work order formatting

            # Common company/manufacturer names
            r'\bNational\s+0ilwell\s+Varco\b': 'National Oilwell Varco',
            r'\bSiemens\b': 'Siemens',
            r'\bGE\s+(?:Electric|Healthcare|Power)\b': lambda m: m.group(0).replace('GE', 'General Electric'),

            # Location and facility codes
            r'\bBldg[\s]*(\d+)': r'Building \1',
            r'\bFlr[\s]*(\d+)': r'Floor \1',
            r'\bRm[\s]*(\d+)': r'Room \1',

            # Maintenance terminology
            r'\bPM[\s]*(\d+)': r'Preventive Maintenance \1',
            r'\bCM[\s]*(\d+)': r'Corrective Maintenance \1',
            r'\bCBM\b': 'Condition Based Maintenance',
            r'\bPdM\b': 'Predictive Maintenance',

            # Common OCR character errors
            r'\bl\b(?=\s+[A-Z])': 'I',  # Lowercase l that should be I
            r'\bO(?=\d)': '0',  # O that should be 0 before numbers
            r'\b0(?=[A-Z])': 'O',  # 0 that should be O before letters
            r'(?<=\d)O(?=\d)': '0',  # O between digits should be 0
            r'(?<=\d)l(?=\d)': '1',  # l between digits should be 1

            # Equipment and part number patterns
            r'\bP/N[\s]*([A-Z0-9-]+)': r'Part Number \1',
            r'\bM/N[\s]*([A-Z0-9-]+)': r'Model Number \1',
            r'\bS/N[\s]*([A-Z0-9-]+)': r'Serial Number \1',

            # Status and priority codes
            r'\bCRITICAL\b': 'Critical Priority',
            r'\bURGENT\b': 'Urgent Priority',
            r'\bHIGH\b(?=\s+[Pp]riority)': 'High',
            r'\bMEDIUM\b(?=\s+[Pp]riority)': 'Medium',
            r'\bLOW\b(?=\s+[Pp]riority)': 'Low',

            # Spacing and formatting issues
            r'([A-Z])([a-z]+)([A-Z])': r'\1\2 \3',  # camelCase to spaced
            r'\s+': ' ',  # Multiple spaces to single space
            r'([A-Z]{2,})\s+([A-Z]{2,})': r'\1 \2',  # Space between acronyms

            # Date and time formatting
            r'(\d{1,2})/(\d{1,2})/(\d{4})': r'\1/\2/\3',  # Standardize date format
            r'(\d{1,2}):(\d{2})\s*([AP]M)': r'\1:\2 \3',  # Time formatting
        }

        for pattern, replacement in quick_fixes.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _sanitize_single_chunk(self, text: str, document_type: str,
                               requests_session=None, doc_context: str = "") -> str:
        """
        Sanitize a single chunk of text using LLM.

        Args:
            text: Text chunk to sanitize
            document_type: Document type for context
            requests_session: Optional session for connection pooling
            doc_context: Document context for this chunk

        Returns:
            Sanitized text chunk
        """

        system_prompt = self._get_sanitization_system_prompt(document_type)
        user_prompt = self._get_sanitization_user_prompt(text, document_type, doc_context)

        try:
            # Get LLM configuration
            model = self.llm_config.get("model", "gpt-4o-mini")
            api_key = self.llm_config.get("api_key")
            base_url = self.llm_config.get("base_url")
            max_tokens = self.sanitization_config.get("max_tokens", 4000)
            temperature = self.sanitization_config.get("temperature", 0.1)  # Low temp for consistency

            if not model or not api_key:
                raise ValueError("LLM model or API key missing in config")

            logger.debug(f"Calling LLM for text sanitization: model={model}, max_tokens={max_tokens}")

            response = call_llm(
                model=model,
                user_prompt=user_prompt,
                api_key=api_key,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=base_url,
                session=requests_session
            )

            if response and response.strip():
                logger.debug("LLM sanitization successful")
                return response.strip()
            else:
                logger.warning("LLM returned empty response, using original text")
                return text

        except Exception as e:
            logger.error(f"Error in LLM text sanitization: {e}")
            # Return original text on any error to avoid breaking pipeline
            return text

    def _combine_sanitized_chunks(self, chunks: List[str]) -> str:
        """
        Intelligently combine sanitized chunks back into a coherent document.

        Args:
            chunks: List of sanitized text chunks

        Returns:
            Combined text with proper section separation
        """
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        # Join chunks with section separators
        combined = []
        for i, chunk in enumerate(chunks):
            combined.append(chunk.strip())

            # Add section separator between chunks (except after last chunk)
            if i < len(chunks) - 1:
                combined.append("\n\n---\n\n")

        return "\n".join(combined)

    def _get_sanitization_system_prompt(self, document_type: str) -> str:
        """Generate system prompt for text sanitization based on document type."""

        base_prompt = """You are an expert text sanitization specialist for Physical Asset Management systems. Your task is to transform raw OCR-extracted text into clean, structured, and standardized text that will be used for knowledge graph generation.

Your responsibilities:
1. **OCR Error Correction**: Fix common OCR errors, especially equipment IDs, serial numbers, and technical codes
2. **Asset Standardization**: Ensure consistent naming for assets, locations, personnel, and equipment following ISO 55000 standards
3. **Technical Abbreviation Expansion**: Expand maintenance and asset management abbreviations while preserving originals
4. **Relationship Clarification**: Make asset hierarchies, maintenance relationships, and operational dependencies explicit
5. **Structure Enhancement**: Organize information following asset management document standards

Critical Guidelines:
- Preserve ALL factual information (asset IDs, serial numbers, dates, measurements, specifications)
- Use consistent asset naming conventions throughout
- Follow hierarchical asset structures (Site → Building → System → Equipment → Component)
- Make maintenance relationships explicit (operates, maintains, located_at, part_of)
- Apply ISO 55000, ISO 14224, and industry maintenance standards
- Ensure output supports asset tracking and maintenance knowledge extraction"""

        document_specific = {
            "maintenance_report": """
This document is a maintenance report. Focus on:
- Asset identification and hierarchical relationships (Site → Building → System → Equipment)
- Work order details and maintenance activities (PM, CM, inspections)
- Personnel assignments and responsibilities (technicians, operators, supervisors)
- Parts consumption and material usage (spare parts, consumables, tools)
- Equipment condition and performance metrics (operational status, measurements)
- Corrective vs preventive maintenance classification and scheduling""",

            "asset_inventory": """
This document is an asset inventory or specification. Focus on:
- Asset hierarchical classification and location (facility → area → equipment → component)
- Technical specifications and manufacturer information (model, serial, capacity)
- Installation dates and lifecycle information (commissioning, warranty, replacement)
- Maintenance schedules and requirements (PM intervals, inspection frequencies)
- Criticality and operational importance (business impact, safety classification)""",

            "inspection_report": """
This document is an inspection report. Focus on:
- Inspection procedures and compliance requirements (regulatory, safety, quality)
- Asset condition assessments and findings (wear, performance, defects)
- Regulatory compliance and certification status (ISO, OSHA, industry standards)
- Inspection personnel and qualifications (certified inspectors, technical specialists)
- Corrective actions and recommendations (repairs, replacements, monitoring)""",

            "work_order": """
This document is a work order. Focus on:
- Work request details and priority classification (emergency, urgent, routine)
- Asset assignments and location information (equipment tags, facility codes)
- Resource requirements and scheduling (labor hours, materials, tools)
- Approval workflows and status tracking (requested, approved, in-progress, complete)
- Cost center and budget allocation (maintenance budgets, project codes)""",

            "invoice": """
This document is a service/parts invoice. Focus on:
- Service provider and customer relationships (contractors, suppliers, operators)
- Asset-related services and parts (maintenance services, spare parts, consumables)
- Purchase orders and cost tracking (PO numbers, cost centers, budgets)
- Service locations and asset assignments (equipment served, facility locations)
- Warranty and contract information (service agreements, parts warranties)""",

            "unknown": """
Analyze the content to determine document type and apply appropriate asset management domain knowledge.
Consider equipment specifications, maintenance records, facility management, or operational documentation."""
        }

        specific_instruction = document_specific.get(document_type, document_specific["unknown"])

        return f"{base_prompt}\n\n{specific_instruction}"

    def _get_sanitization_user_prompt(self, text: str, document_type: str, doc_context: str = "") -> str:
        """Generate user prompt for text sanitization."""

        context_section = ""
        if doc_context:
            context_section = f"""
Document Context (for reference):
{doc_context}

"""

        return f"""{context_section}Please sanitize and structure the following raw OCR text for optimal knowledge graph extraction. Apply OCR error correction, asset standardization, abbreviation expansion, and relationship clarification as outlined in your instructions.

Focus on Physical Asset Management entities such as:
- **Assets**: Equipment, machinery, systems, components, infrastructure
- **Locations**: Sites, buildings, floors, rooms, facilities, geographic coordinates
- **Personnel**: Technicians, operators, inspectors, managers, contractors
- **Organizations**: Manufacturers, service providers, suppliers, operators
- **Maintenance**: Work orders, inspections, repairs, preventive maintenance, schedules
- **Parts & Materials**: Components, spare parts, consumables, inventory items
- **Procedures**: Maintenance procedures, inspection protocols, safety requirements
- **Measurements**: Performance metrics, condition indicators, operational parameters
- **Documents**: Manuals, specifications, certificates, reports, compliance records

Apply asset management standards and conventions:
- Asset hierarchy: Site → Building → System → Equipment → Component
- Asset identification: Consistent naming, manufacturer codes, serial numbers
- Maintenance codes: Work order types, priority levels, status tracking
- Location codes: Hierarchical facility identification
- Personnel roles: Maintenance technician, inspector, operator, supervisor

Raw OCR Text:
---
{text}
---

Provide the sanitized and structured text:"""


# Usage example and configuration
def create_default_config() -> Dict[str, Any]:
    """Create default configuration for text sanitization."""
    return {
        "text_sanitization": {
            "max_tokens_per_chunk": 3000,
            "overlap_tokens": 200,
            "min_chunk_tokens": 100,
            "max_tokens": 4000,
            "temperature": 0.1
        },
        "llm": {
            "model": "gpt-4o-mini",  # Cost-effective choice
            "api_key": "your-api-key-here",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "max_tokens": 4000,
            "temperature": 0.1
        }
    }


# Integration function for existing pipeline
def sanitize_and_structure_text(text: str, config: Dict[str, Any],
                                document_type: str = "unknown",
                                requests_session=None) -> str:
    """
    Main function to be called from processing_pipeline.py

    Args:
        text: Raw OCR text
        config: Configuration dictionary
        document_type: Type of document being processed
        requests_session: Optional session for connection pooling

    Returns:
        Sanitized and structured text ready for KG extraction
    """
    sanitizer = TextSanitizer(config)
    return sanitizer.sanitize_text(text, document_type, requests_session)