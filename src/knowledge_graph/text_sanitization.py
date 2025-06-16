#!/usr/bin/env python3
"""
Text Sanitization and Structuring Module for Knowledge Graph Pipeline
Handles OCR error correction with STRICT no-hallucination constraints.
"""

import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import tiktoken

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
    document_context: str = ""


class TextSanitizer:
    """
    Handles text sanitization with STRICT no-hallucination constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the text sanitizer."""
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
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None

        logger.info(f"TextSanitizer initialized with no-hallucination mode")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using appropriate tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4

    def should_chunk_text(self, text: str) -> bool:
        """Determine if text needs to be chunked based on token count."""
        token_count = self.count_tokens(text)
        threshold = self.max_tokens_per_chunk * 0.8

        should_chunk = token_count > threshold
        logger.info(f"Text token count: {token_count}, threshold: {threshold}, will_chunk: {should_chunk}")
        return should_chunk

    def create_intelligent_chunks(self, text: str, document_type: str = "unknown") -> List[ChunkInfo]:
        """Create intelligent chunks that respect document structure."""
        logger.info(f"Creating intelligent chunks for {document_type} document")

        breakpoints = self._get_document_breakpoints(document_type)
        chunks = self._chunk_by_structure(text, breakpoints)

        # Check for oversized chunks
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

        chunks = self._add_context_to_chunks(chunks, text)
        logger.info(f"Created {len(chunks)} chunks with token counts: {[c.token_count for c in chunks]}")
        return chunks

    def _get_document_breakpoints(self, document_type: str) -> List[str]:
        """Get document-type specific section breakpoints."""
        breakpoint_patterns = {
            "invoice": [
                r"\n\s*#+\s*[A-Z][^#\n]*",
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",
                r"\n\s*\d+\.\s*[A-Z]",
                r"\n\s*Invoice\s+Details?", r"\n\s*Customer\s+Information",
                r"\n\s*Service\s+Operations", r"\n\s*Financial\s+Information"
            ],
            "report": [
                r"\n\s*#+\s*[A-Z][^#\n]*",
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",
                r"\n\s*\d+\.\s*[A-Z]",
                r"\n\s*Summary", r"\n\s*Details", r"\n\s*Conclusions?"
            ],
            "default": [
                r"\n\s*#+\s*[A-Z][^#\n]*",
                r"\n\s*[A-Z][A-Z\s]{10,}:?\s*\n",
                r"\n\s*\d+\.\s*[A-Z]",
            ]
        }
        return breakpoint_patterns.get(document_type, breakpoint_patterns["default"])

    def _chunk_by_structure(self, text: str, breakpoints: List[str]) -> List[ChunkInfo]:
        """Chunk text by document structure using breakpoint patterns."""
        break_positions = [0]

        for pattern in breakpoints:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            for match in matches:
                break_positions.append(match.start())

        break_positions.append(len(text))
        break_positions = sorted(set(break_positions))

        chunks = []
        for i in range(len(break_positions) - 1):
            start_pos = break_positions[i]
            end_pos = break_positions[i + 1]
            chunk_text = text[start_pos:end_pos].strip()

            if len(chunk_text) > 10:
                token_count = self.count_tokens(chunk_text)
                chunk_info = ChunkInfo(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    chunk_id=f"struct_chunk_{i}",
                    token_count=token_count
                )
                chunks.append(chunk_info)

        if len(chunks) <= 1:
            logger.info("Structure-based chunking found minimal sections, falling back to token-based")
            return self._chunk_by_tokens(text, 0)

        return chunks

    def _chunk_by_tokens(self, text: str, start_offset: int = 0) -> List[ChunkInfo]:
        """Fall back to token-based chunking with overlap."""
        chunks = []
        chunk_size = self.max_tokens_per_chunk - 100
        overlap_size = self.overlap_tokens

        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_tokens = 0
        chunk_start = start_offset
        chunk_idx = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(current_chunk),
                    chunk_id=f"token_chunk_{chunk_idx}",
                    token_count=current_tokens
                )
                chunks.append(chunk_info)

                overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_idx += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens

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
        doc_context = self._extract_document_context(full_text)

        for chunk in chunks:
            chunk.document_context = doc_context

        return chunks

    def _extract_document_context(self, text: str) -> str:
        """Extract key document context (first few lines, headers, etc.)."""
        lines = text.split('\n')[:10]
        context_lines = []

        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                context_lines.append(line)
                if len(context_lines) >= 5:
                    break

        return ' | '.join(context_lines)

    def sanitize_text(self, text: str, document_type: str = "unknown",
                      requests_session=None) -> str:
        """
        Main function to sanitize text with STRICT no-hallucination constraints.
        """
        logger.info(f"Starting STRICT text sanitization for {document_type} document (length: {len(text)} chars)")

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return text

        # Apply rule-based preprocessing first
        text = self._rule_based_ocr_cleaning(text)

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
                sanitized_chunks.append(chunk.text)

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                sanitized_chunks.append(chunk.text)

        final_text = self._combine_sanitized_chunks(sanitized_chunks)
        logger.info(f"STRICT text sanitization complete. Output length: {len(final_text)} chars")

        return final_text

    def _rule_based_ocr_cleaning(self, text: str) -> str:
        """
        Apply deterministic OCR fixes before LLM processing.
        These are safe, predictable corrections that don't add information.
        """
        logger.debug("Applying rule-based OCR cleaning")

        # Character-level OCR fixes (common misrecognitions)
        character_fixes = {
            r'\b0f\b': 'of',  # 0 misread as O
            r'\bfank\b': 'tank',  # f misread
            r'\bchcmical\b': 'chemical',  # missing i
            r'\btech-s\b': 'technician',  # truncated word
            r'\bbbls\b': 'barrels',  # standard abbreviation
            r'\bl\b(?=\s+[0-9])': '1',  # lowercase l before numbers
            r'\bO(?=[0-9])': '0',  # O before numbers should be 0
            r'\b0(?=[A-Z])': 'O',  # 0 before letters should be O
            r'(?<=\d)O(?=\d)': '0',  # O between digits should be 0
            r'(?<=\d)l(?=\d)': '1',  # l between digits should be 1
        }

        # Spacing and punctuation fixes
        spacing_fixes = {
            r':get;oil\s*i0\s*scll': 'get oil to sell',  # Specific garbled phrase
            r'\b2s:\s*bbls\b': '25 barrels',  # Specific quantity fix
            r'\[\s*o\s*SWD': 'to SWD',  # Bracket/character error
            r'\s+': ' ',  # Multiple spaces to single
            r'([a-z])([A-Z])': r'\1 \2',  # Add space before capitals
        }

        # Apply character fixes
        for pattern, replacement in character_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Apply spacing fixes
        for pattern, replacement in spacing_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _sanitize_single_chunk(self, text: str, document_type: str,
                               requests_session=None, doc_context: str = "") -> str:
        """
        Sanitize a single chunk of text using LLM with STRICT constraints.
        """
        system_prompt = self._get_no_hallucination_system_prompt(document_type)
        user_prompt = self._get_no_hallucination_user_prompt(text, document_type, doc_context)

        try:
            model = self.llm_config.get("model", "gpt-4o-mini")
            api_key = self.llm_config.get("api_key")
            base_url = self.llm_config.get("base_url")
            max_tokens = self.sanitization_config.get("max_tokens", 4000)
            temperature = 0.05  # Very low temperature for consistency

            if not model or not api_key:
                raise ValueError("LLM model or API key missing in config")

            logger.debug(f"Calling LLM for STRICT text sanitization: model={model}")

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
            return text

    def _combine_sanitized_chunks(self, chunks: List[str]) -> str:
        """Intelligently combine sanitized chunks back into a coherent document."""
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        combined = []
        for i, chunk in enumerate(chunks):
            combined.append(chunk.strip())
            if i < len(chunks) - 1:
                combined.append("\n\n---\n\n")

        return "\n".join(combined)

    def _get_no_hallucination_system_prompt(self, document_type: str) -> str:
        """
        UPDATED: Generate STRICT no-hallucination system prompt.
        This is the key change to prevent fictional content generation.
        """

        return """You are a text correction specialist for Physical Asset Management documents. Your ONLY job is to fix OCR errors while preserving ALL original information exactly as written.

🚨 CRITICAL RULES - NEVER VIOLATE:
1. **NEVER ADD INFORMATION**: Do not invent asset IDs, work orders, personnel names, or activities not explicitly mentioned
2. **NEVER CREATE STRUCTURE**: Do not add sections, headers, or organize information that isn't already present  
3. **NEVER INFER ACTIVITIES**: Do not describe detailed processes not explicitly stated in the source
4. **NEVER ADD CONTEXT**: Do not add background information, standards, or domain knowledge
5. **NEVER CREATE ENTITIES**: Do not generate equipment codes, serial numbers, or organizational details

✅ ONLY ALLOWED CORRECTIONS:
1. **OCR Character Fixes**: 
   - "0f" → "of", "fank" → "tank", "chcmical" → "chemical"
   - "tech-s" → "technician", "bbls" → "barrels"
   - Fix obvious character recognition errors only

2. **Spacing and Punctuation**:
   - Add spaces between words wrongly merged: "oiltank" → "oil tank"
   - Fix basic punctuation for readability
   - Remove garbled characters: ":get;oil i0 scll" → "get oil to sell"

3. **Standard Abbreviation Recognition** (ONLY if universally clear):
   - "SWD" → "Salt Water Disposal" 
   - "PM" → "Preventive Maintenance" (only if clearly maintenance context)
   - Keep original abbreviation: "SWD (Salt Water Disposal)"

❌ FORBIDDEN ACTIONS:
- Do NOT add: Asset IDs (OT-001, WO-2025-001), Serial Numbers, Personnel names
- Do NOT create: Equipment lists, Structured sections, Missing information placeholders  
- Do NOT infer: Detailed procedures, Maintenance activities, Equipment relationships
- Do NOT expand: Brief mentions into detailed descriptions
- Do NOT organize: Raw information into structured formats
- Do NOT add: Work order numbers, dates not in original, organizational details

EXAMPLE - CORRECT APPROACH:
Input: "Drive to location, total 0f25 bbls 0 boltonis slowly pull from oil fank"
Output: "Drive to location, total of 25 barrels bolt-ons slowly pull from oil tank"

EXAMPLE - WRONG APPROACH (DO NOT DO THIS):
❌ "Work Order: WO-2025-001, Equipment: Oil Tank OT-001, Activity: Corrective Maintenance"

Your output should read like a cleaned version of the original text, NOT a structured maintenance report. Preserve the original style, length, and level of detail exactly."""

    def _get_no_hallucination_user_prompt(self, text: str, document_type: str, doc_context: str = "") -> str:
        """
        UPDATED: Generate STRICT user prompt that prevents hallucination.
        This is the second key change to enforce constraints.
        """

        context_section = ""
        if doc_context:
            context_section = f"Document Context (for reference only): {doc_context}\n\n"

        return f"""{context_section}Fix ONLY the OCR errors in this text. Do not add any information, structure, or details not present in the original.

⚠️ STRICT REQUIREMENTS:
- Fix character recognition errors only (0→o, fank→tank, chcmical→chemical)
- Correct obvious spacing and punctuation issues
- Do NOT add asset IDs, work orders, personnel names, or equipment codes
- Do NOT create sections or organize information  
- Do NOT expand brief mentions into detailed descriptions
- Keep the same level of detail and style as the original
- Output should be the same length and content as input, just with OCR errors fixed

Raw OCR Text:
---
{text}
---

Corrected Text (same content, fixed OCR errors only):"""


# Keep the existing utility functions
def create_default_config() -> Dict[str, Any]:
    """Create default configuration for text sanitization."""
    return {
        "text_sanitization": {
            "max_tokens_per_chunk": 3000,
            "overlap_tokens": 200,
            "min_chunk_tokens": 100,
            "max_tokens": 4000,
            "temperature": 0.05  # Very low for consistency
        },
        "llm": {
            "model": "gpt-4o-mini",
            "api_key": "your-api-key-here",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "max_tokens": 4000,
            "temperature": 0.05
        }
    }


def sanitize_and_structure_text(text: str, config: Dict[str, Any],
                                document_type: str = "unknown",
                                requests_session=None) -> str:
    """
    Main function to be called from processing_pipeline.py
    """
    sanitizer = TextSanitizer(config)
    return sanitizer.sanitize_text(text, document_type, requests_session)