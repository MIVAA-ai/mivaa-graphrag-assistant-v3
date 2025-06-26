# enhanced_ocr_pipeline.py - COMPLETE LLM OCR ONLY VERSION WITH METADATA

import logging
import tempfile
import hashlib
import base64
import io
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import Counter
import warnings
import os
import threading
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PDF processing
try:
    import fitz  # PyMuPDF
    PDF_PROCESSING_AVAILABLE = True
    logger.info("PyMuPDF (fitz) imported successfully")
except ImportError as e:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning(f"PyMuPDF (fitz) not available: {e}")

# Alternative import attempt
if not PDF_PROCESSING_AVAILABLE:
    try:
        import PyMuPDF as fitz
        PDF_PROCESSING_AVAILABLE = True
        logger.info("PyMuPDF imported successfully as alternative")
    except ImportError:
        logger.error("Neither 'fitz' nor 'PyMuPDF' could be imported")

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English model for entity detection
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# LLM Clients
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# OCR storage with proper singleton handling
try:
    from src.utils.ocr_storage import create_storage_manager, get_storage_manager
    OCR_STORAGE_AVAILABLE = True
except ImportError:
    OCR_STORAGE_AVAILABLE = False

    def create_storage_manager(*args, **kwargs):
        return None

    def get_storage_manager():
        return None

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OCRMethod(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    MISTRAL = "mistral"
    HYBRID = "hybrid"


@dataclass
class OCRResult:
    """Enhanced OCR result structure with comprehensive metadata"""
    # Core OCR results
    success: bool
    text: str
    confidence: float
    method_used: str
    processing_time: float
    text_regions_detected: int
    preprocessing_applied: List[str]
    error_message: Optional[str] = None
    saved_files: Optional[Dict] = None
    structured_data: Optional[Dict] = None
    detected_tables: Optional[List] = None
    invoice_fields: Optional[Dict] = None

    # Enhanced metadata fields
    file_metadata: Optional[Dict] = None
    content_metadata: Optional[Dict] = None
    quality_metrics: Optional[Dict] = None
    detected_entities: Optional[Dict] = None
    document_classification: Optional[Dict] = None
    processing_metadata: Optional[Dict] = None
    chunk_metadata: Optional[List[Dict]] = None


class TimeoutHandler:
    """Cross-platform timeout handler"""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.timed_out = False

    def __enter__(self):
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_occurred)
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        if self.timed_out:
            raise TimeoutError(f"LLM OCR processing timed out after {self.timeout_seconds} seconds")

    def _timeout_occurred(self):
        self.timed_out = True
        logger.error(f"LLM OCR processing timed out after {self.timeout_seconds} seconds")


class LLMOCRExtractor:
    """
    Enhanced LLM-based OCR extractor with comprehensive metadata extraction
    Superior accuracy and rich metadata compared to traditional OCR
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize all LLM clients
        self.gemini_client = None
        self.openai_client = None
        self.claude_client = None
        self.mistral_client = None

        # Configuration
        self.primary_method = config.get('LLM_OCR_PRIMARY_METHOD', 'gemini')
        self.fallback_enabled = config.get('LLM_OCR_FALLBACK_ENABLED', True)
        self.timeout_seconds = config.get('LLM_OCR_TIMEOUT', 60)
        self.max_retries = config.get('LLM_OCR_MAX_RETRIES', 2)

        # Enhanced metadata configuration
        self.extract_entities = config.get('EXTRACT_ENTITIES', True)
        self.classify_documents = config.get('CLASSIFY_DOCUMENTS', True)
        self.analyze_quality = config.get('ANALYZE_QUALITY', True)
        self.chunk_size = config.get('CHUNK_SIZE', 1000)

        # Initialize clients
        self._initialize_llm_clients()

        # Storage manager (singleton)
        self._storage_manager = None
        if OCR_STORAGE_AVAILABLE:
            try:
                self._storage_manager = get_storage_manager()
                if self._storage_manager is None:
                    self._storage_manager = create_storage_manager("ocr_outputs")
            except Exception as e:
                logger.warning(f"Could not initialize OCR storage: {e}")
                self._storage_manager = None

    def _initialize_llm_clients(self):
        """Initialize available LLM clients"""

        # Gemini client
        gemini_api_key = (
                self.config.get('GOOGLE_API_KEY') or
                self.config.get('LLM_API_KEY') or
                self.config.get('GEMINI_API_KEY') or
                self.config.get('gemini_api_key') or
                self.config.get('llm', {}).get('api_key') or
                self.config.get('llm', {}).get('ocr', {}).get('gemini_api_key')
        )

        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini 1.5 Flash initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini_client = None

        # OpenAI client
        openai_api_key = (
                self.config.get('OPENAI_API_KEY') or
                self.config.get('openai_api_key') or
                self.config.get('llm', {}).get('ocr', {}).get('openai_api_key')
        )

        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI GPT-4o Vision initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.openai_client = None

        # Claude client
        claude_api_key = (
                self.config.get('ANTHROPIC_API_KEY') or
                self.config.get('claude_api_key') or
                self.config.get('llm', {}).get('ocr', {}).get('anthropic_api_key')
        )

        if claude_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
                logger.info("Claude 3.5 Sonnet initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
                self.claude_client = None

        # Mistral client
        mistral_api_key = (
                self.config.get('MISTRAL_API_KEY') or
                self.config.get('mistral_api_key') or
                self.config.get('mistral', {}).get('api_key') or
                self.config.get('llm', {}).get('ocr', {}).get('mistral_api_key')
        )

        if mistral_api_key and MISTRAL_AVAILABLE:
            try:
                self.mistral_client = Mistral(api_key=mistral_api_key)
                logger.info("Mistral Pixtral initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral: {e}")
                self.mistral_client = None

    def _generate_file_metadata(self, uploaded_file) -> Dict[str, Any]:
        """Generate comprehensive file metadata"""
        file_content = uploaded_file.getvalue()

        return {
            "file_id": str(uuid.uuid4()),
            "original_filename": uploaded_file.name,
            "file_hash": hashlib.sha256(file_content).hexdigest(),
            "file_size_bytes": len(file_content),
            "mime_type": uploaded_file.type,
            "upload_timestamp": datetime.now().isoformat(),
            "file_extension": Path(uploaded_file.name).suffix.lower(),
        }

    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy and domain patterns"""
        if not text.strip():
            return {}

        entities = {
            "companies": [],
            "people": [],
            "locations": [],
            "dates": [],
            "money": [],
            "organizations": [],
            "wells": [],  # Oil & Gas specific
            "formations": [],  # Oil & Gas specific
            "equipment": [],  # Oil & Gas specific
        }

        # Use spaCy if available
        if nlp:
            try:
                doc = nlp(text)

                # Extract standard entities
                for ent in doc.ents:
                    if ent.label_ in ["ORG"]:
                        entities["companies"].append(ent.text.strip())
                    elif ent.label_ in ["PERSON"]:
                        entities["people"].append(ent.text.strip())
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities["locations"].append(ent.text.strip())
                    elif ent.label_ in ["DATE"]:
                        entities["dates"].append(ent.text.strip())
                    elif ent.label_ in ["MONEY"]:
                        entities["money"].append(ent.text.strip())

            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")

        # Extract domain-specific entities (Oil & Gas) using patterns
        try:
            well_patterns = [
                r'\b[A-Z]{1,3}[-\s]?\d{1,4}[A-Z]?\b',  # Well names like A-1, B-12A
                r'\bWell\s+[A-Z0-9\-]+\b',  # "Well ABC-123"
                r'\b\w+[-\s]\d+[A-Z]?\s+Well\b',  # "Smith-1A Well"
            ]

            formation_patterns = [
                r'\b\w+\s+Formation\b',  # "Daman Formation"
                r'\b\w+\s+Sand\b',  # "Uinta Sand"
                r'\b\w+\s+Shale\b',  # "Bakken Shale"
                r'\b\w+\s+Limestone\b',  # "Austin Limestone"
            ]

            equipment_patterns = [
                r'\bdrilling\s+rig\b',
                r'\bcompletion\s+tools?\b',
                r'\bpump\s+jack\b',
                r'\bblowout\s+preventer\b',
                r'\bchristmas\s+tree\b',
                r'\bperforation\s+gun\b',
                r'\bdownhole\s+motor\b',
            ]

            # Apply domain-specific patterns
            for pattern in well_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["wells"].extend([m.strip() for m in matches])

            for pattern in formation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["formations"].extend([m.strip() for m in matches])

            for pattern in equipment_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["equipment"].extend([m.strip() for m in matches])

        except Exception as e:
            logger.warning(f"Pattern-based entity extraction failed: {e}")

        # Remove duplicates and limit count
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # Limit to 10 per type

        # Remove empty lists
        entities = {k: v for k, v in entities.items() if v}

        return entities

    def _classify_document_type(self, text: str, filename: str) -> Dict[str, Any]:
        """Classify document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Document type indicators
        type_indicators = {
            "invoice": ["invoice", "bill", "payment", "amount due", "total:", "$", "invoice #", "invoice number"],
            "contract": ["agreement", "contract", "terms and conditions", "party", "whereas", "hereby", "witnesseth"],
            "report": ["report", "analysis", "summary", "findings", "conclusion", "executive summary"],
            "letter": ["dear", "sincerely", "regards", "letter", "correspondence", "yours truly"],
            "form": ["form", "application", "submit", "signature", "date:", "name:", "please fill"],
            "technical": ["specification", "procedure", "technical", "engineering", "design", "specifications"],
            "drilling_report": ["drilling", "wellbore", "completion", "production", "reservoir", "mud log"],
            "geological_report": ["formation", "geology", "seismic", "core", "log", "lithology", "stratigraphy"],
            "financial": ["balance sheet", "income statement", "cash flow", "revenue", "expenses", "profit"],
            "legal": ["legal", "lawsuit", "court", "attorney", "law", "jurisdiction", "plaintiff"],
        }

        # Calculate scores for each document type
        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            # Add filename bonus
            if any(word in filename_lower for word in indicators[:3]):  # Check first 3 indicators
                score += 2
            scores[doc_type] = score

        # Determine primary type
        if max(scores.values()) == 0:
            primary_type = "unknown"
            confidence = 0.0
        else:
            primary_type = max(scores.keys(), key=lambda k: scores[k])
            # Normalize confidence by text length
            confidence = min(scores[primary_type] / max(len(text.split()) / 100, 1), 1.0)

        return {
            "document_type": primary_type,
            "classification_confidence": confidence,
            "type_scores": scores,
            "category": self._get_document_category(primary_type)
        }

    def _get_document_category(self, doc_type: str) -> str:
        """Map document type to broader category"""
        category_mapping = {
            "invoice": "financial",
            "contract": "legal",
            "report": "technical",
            "drilling_report": "technical",
            "geological_report": "technical",
            "letter": "correspondence",
            "form": "administrative",
            "technical": "technical",
            "financial": "financial",
            "legal": "legal"
        }
        return category_mapping.get(doc_type, "general")

    def _analyze_text_quality(self, text: str, confidence: float) -> Dict[str, Any]:
        """Analyze text quality metrics"""
        if not text.strip():
            return {
                "quality_score": 0.0,
                "readability_score": 0.0,
                "complexity": "unknown",
                "issues": ["empty_text"]
            }

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Basic quality indicators
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0

        # Quality issues detection
        issues = []
        if confidence < 0.7:
            issues.append("low_ocr_confidence")
        if avg_word_length < 3:
            issues.append("short_words")
        if len(words) < 10:
            issues.append("insufficient_text")
        if re.search(r'[^\w\s\.,!?;:\-\(\)\"\']+', text):
            issues.append("special_characters")
        if len(sentences) < 2:
            issues.append("few_sentences")

        # Calculate overall quality score
        quality_factors = [
            confidence,  # OCR confidence
            min(len(words) / 100, 1.0),  # Text length factor
            min(avg_word_length / 5, 1.0),  # Word length factor
            1.0 - (len(issues) * 0.15)  # Issue penalty
        ]

        quality_score = max(0.0, sum(quality_factors) / len(quality_factors))

        # Complexity assessment
        if avg_sentence_length > 20 and avg_word_length > 5:
            complexity = "high"
        elif avg_sentence_length > 15 or avg_word_length > 4:
            complexity = "medium"
        else:
            complexity = "low"

        # Simple readability score (Flesch-like)
        readability = max(0.0, min(100.0, 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))))

        return {
            "quality_score": quality_score,
            "readability_score": readability,
            "complexity": complexity,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "issues": issues,
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text)
        }

    def _create_chunk_metadata(self, text: str) -> List[Dict[str, Any]]:
        """Create metadata for text chunks"""
        if not text.strip():
            return []

        chunks = []
        words = text.split()

        # Simple word-based chunking
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk_metadata = {
                "chunk_id": f"chunk_{i // self.chunk_size}",
                "chunk_text": chunk_text,
                "start_word": i,
                "end_word": min(i + self.chunk_size, len(words)),
                "word_count": len(chunk_words),
                "character_count": len(chunk_text)
            }
            chunks.append(chunk_metadata)

        return chunks

    def _convert_file_to_image(self, uploaded_file) -> bytes:
        """Convert uploaded file to image bytes for LLM processing"""
        file_type = uploaded_file.type
        file_content = uploaded_file.getvalue()

        if file_type == "application/pdf":
            if not PDF_PROCESSING_AVAILABLE:
                raise ValueError("PyMuPDF not available for PDF processing")

            # Convert PDF to image using PyMuPDF
            doc = fitz.open(stream=file_content, filetype="pdf")
            page = doc.load_page(0)  # First page

            # High DPI for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x scaling
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            doc.close()

            return img_data

        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            # Enhance image quality if needed
            if PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(file_content))

                    # Enhance image for better OCR
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)

                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.1)

                    # Convert to PNG for consistency
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', optimize=True)
                    return img_buffer.getvalue()

                except Exception as e:
                    logger.warning(f"Image enhancement failed: {e}, using original")

            return file_content

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_with_gemini(self, image_data: bytes, file_name: str) -> OCRResult:
        """Extract text using Gemini 1.5 Flash"""
        if not self.gemini_client:
            raise ValueError("Gemini client not available")

        start_time = time.time()

        try:
            # Prepare image for Gemini
            image_part = {
                "mime_type": "image/png",
                "data": image_data
            }

            # Advanced OCR prompt for structured extraction
            prompt = """Extract ALL text from this document with high accuracy. 

Instructions:
1. Extract every word, number, and symbol visible in the image
2. Maintain the original text structure and formatting
3. For tables, preserve row and column relationships
4. Include headers, footers, and any watermarks
5. If you see financial data (amounts, dates, invoice numbers), be extra careful with accuracy
6. Return only the extracted text without any commentary

Focus on perfect accuracy - this is for business document processing."""

            # Make API call with timeout
            with TimeoutHandler(self.timeout_seconds):
                response = self.gemini_client.generate_content([prompt, image_part])
                extracted_text = response.text

            processing_time = time.time() - start_time
            confidence = self._estimate_llm_confidence(extracted_text, "gemini")

            logger.info(f"Gemini extraction completed for {file_name}: "
                        f"{len(extracted_text)} chars, confidence: {confidence:.3f}, "
                        f"time: {processing_time:.2f}s")

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=confidence,
                method_used="gemini_1.5_flash",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["llm_vision_extraction"]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Gemini extraction failed for {file_name}: {e}")

            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                method_used="gemini_failed",
                processing_time=processing_time,
                text_regions_detected=0,
                preprocessing_applied=[],
                error_message=str(e)
            )

    def _extract_with_openai(self, image_data: bytes, file_name: str) -> OCRResult:
        """Extract text using OpenAI GPT-4o Vision"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        start_time = time.time()

        try:
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Advanced OCR prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL text from this document with perfect accuracy.

Requirements:
- Extract every visible word, number, and symbol
- Preserve document structure and formatting
- For tables, maintain row/column alignment
- Include all headers, footers, dates, amounts
- Be extremely careful with financial data
- Return only extracted text, no commentary

This is for business document processing - accuracy is critical."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            # Make API call with timeout
            with TimeoutHandler(self.timeout_seconds):
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )

                extracted_text = response.choices[0].message.content

            processing_time = time.time() - start_time
            confidence = self._estimate_llm_confidence(extracted_text, "openai")

            logger.info(f"OpenAI extraction completed for {file_name}: "
                        f"{len(extracted_text)} chars, confidence: {confidence:.3f}, "
                        f"time: {processing_time:.2f}s")

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=confidence,
                method_used="gpt_4o_vision",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["llm_vision_extraction"]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"OpenAI extraction failed for {file_name}: {e}")

            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                method_used="openai_failed",
                processing_time=processing_time,
                text_regions_detected=0,
                preprocessing_applied=[],
                error_message=str(e)
            )

    def _extract_with_claude(self, image_data: bytes, file_name: str) -> OCRResult:
        """Extract text using Claude 3.5 Sonnet"""
        if not self.claude_client:
            raise ValueError("Claude client not available")

        start_time = time.time()

        try:
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Make API call with timeout
            with TimeoutHandler(self.timeout_seconds):
                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract ALL text from this document with perfect accuracy.

Instructions:
- Extract every visible word, number, symbol, and character
- Preserve the original document structure and formatting
- For tables, maintain proper row and column relationships  
- Include all headers, footers, dates, amounts, and reference numbers
- Pay special attention to financial data and ensure 100% accuracy
- Return ONLY the extracted text with no additional commentary

This is for business document processing where accuracy is absolutely critical."""
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image
                                    }
                                }
                            ]
                        }
                    ]
                )

                extracted_text = response.content[0].text

            processing_time = time.time() - start_time
            confidence = self._estimate_llm_confidence(extracted_text, "claude")

            logger.info(f"Claude extraction completed for {file_name}: "
                        f"{len(extracted_text)} chars, confidence: {confidence:.3f}, "
                        f"time: {processing_time:.2f}s")

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=confidence,
                method_used="claude_3.5_sonnet",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["llm_vision_extraction"]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Claude extraction failed for {file_name}: {e}")

            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                method_used="claude_failed",
                processing_time=processing_time,
                text_regions_detected=0,
                preprocessing_applied=[],
                error_message=str(e)
            )

    def _extract_with_mistral(self, image_data: bytes, file_name: str) -> OCRResult:
        """Extract text using Mistral Pixtral"""
        if not self.mistral_client:
            raise ValueError("Mistral client not available")

        start_time = time.time()

        try:
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL text from this document with maximum accuracy.

Requirements:
- Extract every word, number, and symbol visible
- Preserve document structure and formatting
- For tables, maintain alignment and relationships
- Include headers, footers, and all data fields
- Be extra careful with financial amounts and dates
- Return only the extracted text without commentary

This is for business document processing - perfect accuracy required."""
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Make API call with timeout
            with TimeoutHandler(self.timeout_seconds):
                response = self.mistral_client.chat.complete(
                    model="pixtral-12b-2409",
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )

                extracted_text = response.choices[0].message.content

            processing_time = time.time() - start_time
            confidence = self._estimate_llm_confidence(extracted_text, "mistral")

            logger.info(f"Mistral extraction completed for {file_name}: "
                        f"{len(extracted_text)} chars, confidence: {confidence:.3f}, "
                        f"time: {processing_time:.2f}s")

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=confidence,
                method_used="mistral_pixtral",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["llm_vision_extraction"]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Mistral extraction failed for {file_name}: {e}")

            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                method_used="mistral_failed",
                processing_time=processing_time,
                text_regions_detected=0,
                preprocessing_applied=[],
                error_message=str(e)
            )

    def _estimate_llm_confidence(self, text: str, method: str) -> float:
        """Estimate confidence based on LLM output quality"""
        if not text or len(text.strip()) == 0:
            return 0.0

        # Base confidence by method (based on our testing)
        base_confidence = {
            "gemini": 0.95,
            "openai": 0.93,
            "claude": 0.92,
            "mistral": 0.88
        }.get(method, 0.85)

        # Quality indicators
        has_structure = any(char in text for char in ['\n', '\t', '  '])
        has_numbers = any(char.isdigit() for char in text)
        has_punctuation = any(char in text for char in '.,!?:;')
        word_count = len(text.split())

        # Adjust confidence based on content quality
        if word_count > 10 and has_structure and has_numbers:
            return min(base_confidence + 0.02, 0.99)
        elif word_count > 5 and has_punctuation:
            return base_confidence
        elif word_count > 0:
            return max(base_confidence - 0.05, 0.70)
        else:
            return 0.0

    def _get_available_methods(self) -> List[str]:
        """Get list of available LLM methods"""
        methods = []
        if self.gemini_client is not None:
            methods.append("gemini")
        if self.openai_client is not None:
            methods.append("openai")
        if self.claude_client is not None:
            methods.append("claude")
        if self.mistral_client is not None:
            methods.append("mistral")
        return methods

    def _perform_ocr_extraction(self, image_data: bytes, file_name: str, file_metadata: Dict) -> OCRResult:
        """Perform OCR extraction using available methods"""
        # Get available methods
        available_methods = self._get_available_methods()

        if not available_methods:
            return self._create_failed_result(file_metadata, "No LLM clients available")

        # Determine extraction order
        if self.primary_method in available_methods:
            methods_to_try = [self.primary_method]
            if self.fallback_enabled:
                methods_to_try.extend([m for m in available_methods if m != self.primary_method])
        else:
            methods_to_try = available_methods

        logger.info(f"Will try methods in order: {methods_to_try}")

        # Try extraction methods in order
        last_error = None
        for method in methods_to_try:
            try:
                logger.info(f"Trying {method.upper()} for {file_name}")

                if method == "gemini":
                    result = self._extract_with_gemini(image_data, file_name)
                elif method == "openai":
                    result = self._extract_with_openai(image_data, file_name)
                elif method == "claude":
                    result = self._extract_with_claude(image_data, file_name)
                elif method == "mistral":
                    result = self._extract_with_mistral(image_data, file_name)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue

                # If successful, return with file metadata
                if result.success and len(result.text.strip()) > 0:
                    result.file_metadata = file_metadata
                    logger.info(f"{method.upper()} extraction successful for {file_name}")
                    return result
                else:
                    logger.warning(f"{method.upper()} extraction failed or returned empty text")
                    last_error = result.error_message

            except Exception as e:
                logger.error(f"{method.upper()} extraction error: {e}")
                last_error = str(e)
                continue

        # All methods failed
        return self._create_failed_result(file_metadata, f"All extraction methods failed. Last error: {last_error}")

    def _enhance_result_with_metadata(self, result: OCRResult, file_metadata: Dict):
        """Add comprehensive metadata to OCR result"""
        text = result.text

        # Extract entities if enabled
        if self.extract_entities:
            result.detected_entities = self._extract_entities_from_text(text)

        # Classify document if enabled
        if self.classify_documents:
            result.document_classification = self._classify_document_type(text, file_metadata["original_filename"])

        # Analyze quality if enabled
        if self.analyze_quality:
            result.quality_metrics = self._analyze_text_quality(text, result.confidence)

        # Create content metadata
        result.content_metadata = {
            "word_count": len(text.split()),
            "character_count": len(text),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "line_count": len(text.split('\n')),
            "has_tables": bool(result.detected_tables),
            "has_financial_data": bool(result.invoice_fields),
            "language_detected": "en",  # Could be enhanced with language detection
        }

        # Create processing metadata
        result.processing_metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "pipeline_version": "v2.1.0",
            "entity_extraction_enabled": self.extract_entities,
            "classification_enabled": self.classify_documents,
            "quality_analysis_enabled": self.analyze_quality,
        }

        # Create chunk metadata
        result.chunk_metadata = self._create_chunk_metadata(text)

        logger.info(f"Enhanced metadata added for {file_metadata['original_filename']}")

    def _create_failed_result(self, file_metadata: Dict, error_message: str, processing_time: float = 0.0) -> OCRResult:
        """Create a failed OCR result with metadata"""
        return OCRResult(
            success=False,
            text="",
            confidence=0.0,
            method_used="failed",
            processing_time=processing_time,
            text_regions_detected=0,
            preprocessing_applied=[],
            error_message=error_message,
            file_metadata=file_metadata
        )

    def _save_enhanced_result(self, uploaded_file, result: OCRResult):
        """Save OCR result with enhanced metadata"""
        try:
            enhanced_metadata = {
                'file_metadata': result.file_metadata,
                'content_metadata': result.content_metadata,
                'quality_metrics': result.quality_metrics,
                'detected_entities': result.detected_entities,
                'document_classification': result.document_classification,
                'processing_metadata': result.processing_metadata,
                'chunk_metadata': result.chunk_metadata,
                'ocr_info': {
                    'method_used': result.method_used,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'text_length': len(result.text)
                }
            }

            saved_files = self._storage_manager.save_ocr_output(
                uploaded_file=uploaded_file,
                ocr_text=result.text,
                structured_data=enhanced_metadata
            )
            result.saved_files = saved_files
            logger.info(f"Enhanced OCR output saved with metadata")

        except Exception as e:
            logger.error(f"Failed to save enhanced OCR output: {e}")

    def extract_text(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """
        Enhanced main extraction method with comprehensive metadata
        """
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        logger.info(f"Starting enhanced LLM OCR extraction for {file_name} (type: {file_type})")

        # Generate file metadata
        file_metadata = self._generate_file_metadata(uploaded_file)

        # Handle text files directly
        if file_type in ['text/plain', 'text/csv', 'text/html', 'text/xml']:
            return self._handle_text_file_with_metadata(uploaded_file, file_metadata, save_to_disk)

        # Convert file to image
        try:
            image_data = self._convert_file_to_image(uploaded_file)
        except Exception as e:
            logger.error(f"Failed to convert {file_name} to image: {e}")
            return self._create_failed_result(file_metadata, f"File conversion failed: {str(e)}")

        # Perform OCR extraction
        result = self._perform_ocr_extraction(image_data, file_name, file_metadata)

        # If extraction successful, add enhanced metadata
        if result.success and result.text.strip():
            self._enhance_result_with_metadata(result, file_metadata)

        # Save enhanced result if requested
        if save_to_disk and self._storage_manager and result.success:
            self._save_enhanced_result(uploaded_file, result)

        return result

    def _handle_text_file_with_metadata(self, uploaded_file, file_metadata: Dict, save_to_disk: bool) -> OCRResult:
        """Handle text files with enhanced metadata"""
        start_time = time.time()
        file_name = uploaded_file.name

        try:
            text_content = uploaded_file.getvalue()

            if isinstance(text_content, bytes):
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                    try:
                        decoded_text = text_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    decoded_text = text_content.decode('utf-8', errors='replace')
            else:
                decoded_text = str(text_content)

            processing_time = time.time() - start_time

            logger.info(f"Text file processed directly: {file_name}, length: {len(decoded_text)}")

            result = OCRResult(
                success=True,
                text=decoded_text,
                confidence=1.0,
                method_used="direct_text",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["direct_text_extraction"],
                file_metadata=file_metadata
            )

            # Add enhanced metadata
            self._enhance_result_with_metadata(result, file_metadata)

            # Save to disk if requested
            if save_to_disk and self._storage_manager:
                self._save_enhanced_result(uploaded_file, result)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process text file {file_name}: {e}")

            return self._create_failed_result(file_metadata, f"Text file processing failed: {str(e)}", processing_time)

    # COMPATIBILITY METHODS - Maintain exact same interface as original
    def set_mistral_client(self, mistral_client):
        """Compatibility method - set Mistral client"""
        self.mistral_client = mistral_client
        logger.info("Mistral client updated for LLM OCR")

    def extract_text_with_fallback(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """Compatibility method - fallback is built into main extract_text method"""
        return self.extract_text(uploaded_file, save_to_disk)

    @property
    def confidence_threshold(self):
        """Return confidence threshold"""
        return self.config.get('LLM_OCR_CONFIDENCE_THRESHOLD', 0.7)

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return self._get_available_methods()


class EnhancedOCRPipeline:
    """
    BACKWARD COMPATIBILITY WRAPPER with Enhanced Metadata
    Drop-in replacement for the original EnhancedOCRPipeline
    Maintains exact same interface while using superior LLM OCR with metadata
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with LLM OCR backend"""
        self.config = config

        # Initialize the LLM OCR extractor
        self.llm_extractor = LLMOCRExtractor(config)

        # Compatibility properties - NO EASYOCR
        self._easyocr_reader = None  # Always None - we don't use EasyOCR
        self._mistral_client = None
        self.primary_method = config.get('LLM_OCR_PRIMARY_METHOD', 'gemini')
        self.fallback_enabled = config.get('LLM_OCR_FALLBACK_ENABLED', True)
        self.confidence_threshold = config.get('LLM_OCR_CONFIDENCE_THRESHOLD', 0.7)

        # Language settings (for compatibility only)
        self.easyocr_languages = ['en']  # Static for compatibility
        self.easyocr_gpu = False  # Static for compatibility

        # Storage manager
        self.ocr_storage = self.llm_extractor._storage_manager

        logger.info("Enhanced OCR Pipeline initialized with LLM OCR backend and metadata extraction")

        # Log available providers
        available = self.llm_extractor.get_available_providers()
        if available:
            logger.info(f"Available LLM providers: {', '.join(available)}")
        else:
            logger.warning("No LLM providers available - check API keys")

    @property
    def easyocr_reader(self):
        """Compatibility property - returns None since we use LLM OCR only"""
        return None

    @property
    def mistral_client(self):
        """Return Mistral client if available"""
        return self.llm_extractor.mistral_client

    def set_mistral_client(self, mistral_client):
        """Set Mistral client for compatibility"""
        self.llm_extractor.mistral_client = mistral_client
        self._mistral_client = mistral_client
        logger.info("Mistral client set in LLM OCR pipeline")

    def extract_text(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """
        MAIN EXTRACTION METHOD with Enhanced Metadata
        Maintains exact same interface as original EnhancedOCRPipeline
        """
        return self.llm_extractor.extract_text(uploaded_file, save_to_disk)

    def extract_text_with_fallback(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """
        FALLBACK EXTRACTION METHOD
        Maintains compatibility - fallback is built into main method
        """
        return self.llm_extractor.extract_text(uploaded_file, save_to_disk)

    def get_available_providers(self) -> List[str]:
        """Get available LLM providers"""
        return self.llm_extractor.get_available_providers()

    def get_status_info(self) -> Dict[str, Any]:
        """Get status information for UI display"""
        available_providers = self.get_available_providers()

        return {
            'primary_method': self.primary_method,
            'fallback_enabled': self.fallback_enabled,
            'available_providers': available_providers,
            'total_providers': len(available_providers),
            'confidence_threshold': self.confidence_threshold,
            'storage_available': self.ocr_storage is not None,
            'backend_type': 'LLM_OCR_ENHANCED',
            'easyocr_available': False,  # Always False
            'metadata_features': {
                'entity_extraction': self.llm_extractor.extract_entities,
                'document_classification': self.llm_extractor.classify_documents,
                'quality_analysis': self.llm_extractor.analyze_quality,
                'chunk_creation': True
            }
        }


# ENHANCED EXPORT FUNCTIONS

def export_for_vector_db(ocr_result: OCRResult) -> Dict[str, Any]:
    """Export OCR result in format suitable for vector database"""
    if not ocr_result.success:
        return {}

    return {
        'text': ocr_result.text,
        'file_id': ocr_result.file_metadata.get('file_id') if ocr_result.file_metadata else None,
        'original_filename': ocr_result.file_metadata.get('original_filename') if ocr_result.file_metadata else None,
        'file_type': ocr_result.file_metadata.get('mime_type') if ocr_result.file_metadata else None,
        'file_size_bytes': ocr_result.file_metadata.get('file_size_bytes') if ocr_result.file_metadata else None,
        'file_hash': ocr_result.file_metadata.get('file_hash') if ocr_result.file_metadata else None,
        'method_used': ocr_result.method_used,
        'confidence': ocr_result.confidence,
        'processing_time': ocr_result.processing_time,
        'document_type': ocr_result.document_classification.get(
            'document_type') if ocr_result.document_classification else None,
        'document_category': ocr_result.document_classification.get(
            'category') if ocr_result.document_classification else None,
        'classification_confidence': ocr_result.document_classification.get(
            'classification_confidence') if ocr_result.document_classification else None,
        'language_detected': ocr_result.content_metadata.get(
            'language_detected') if ocr_result.content_metadata else None,
        'page_count': 1,  # Could be enhanced for multi-page documents
        'has_tables': ocr_result.content_metadata.get('has_tables') if ocr_result.content_metadata else False,
        'has_financial_data': ocr_result.content_metadata.get(
            'has_financial_data') if ocr_result.content_metadata else False,
        'upload_timestamp': ocr_result.file_metadata.get('upload_timestamp') if ocr_result.file_metadata else None,
        'extraction_timestamp': ocr_result.processing_metadata.get(
            'extraction_timestamp') if ocr_result.processing_metadata else None,
        'text_quality_score': ocr_result.quality_metrics.get('quality_score') if ocr_result.quality_metrics else None,
        'readability_score': ocr_result.quality_metrics.get(
            'readability_score') if ocr_result.quality_metrics else None,
        'complexity_score': ocr_result.quality_metrics.get('complexity') if ocr_result.quality_metrics else None,
        'detected_entities': ocr_result.detected_entities,
        'word_count': ocr_result.content_metadata.get('word_count') if ocr_result.content_metadata else None,
        'character_count': ocr_result.content_metadata.get('character_count') if ocr_result.content_metadata else None,
        'chunk_ids': [chunk['chunk_id'] for chunk in ocr_result.chunk_metadata] if ocr_result.chunk_metadata else [],
        'quality_issues': ocr_result.quality_metrics.get('issues') if ocr_result.quality_metrics else []
    }


def export_for_graph_db(ocr_result: OCRResult) -> Tuple[List[Dict], Dict]:
    """Export OCR result as triples and metadata for graph database"""
    if not ocr_result.success:
        return [], {}

    # Generate triples from entities and document relationships
    triples = []
    if ocr_result.detected_entities and ocr_result.chunk_metadata:
        document_name = ocr_result.file_metadata.get('original_filename',
                                                     'unknown') if ocr_result.file_metadata else 'unknown'

        for chunk in ocr_result.chunk_metadata:
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['chunk_text']

            # Create entity-document relationships
            for entity_type, entities in ocr_result.detected_entities.items():
                for entity in entities:
                    if entity.lower() in chunk_text.lower():
                        triples.append({
                            'subject': entity,
                            'predicate': 'appears_in',
                            'object': document_name,
                            'subject_type': entity_type,
                            'object_type': 'document',
                            'chunk_id': chunk_id,
                            'chunk_text': chunk_text,
                            'confidence': ocr_result.confidence,
                            'inferred': False
                        })

                        # Create entity-chunk relationships
                        triples.append({
                            'subject': entity,
                            'predicate': 'mentioned_in_chunk',
                            'object': chunk_id,
                            'subject_type': entity_type,
                            'object_type': 'chunk',
                            'chunk_id': chunk_id,
                            'chunk_text': chunk_text,
                            'confidence': ocr_result.confidence,
                            'inferred': False
                        })

            # Create document-chunk relationships
            triples.append({
                'subject': document_name,
                'predicate': 'contains_chunk',
                'object': chunk_id,
                'subject_type': 'document',
                'object_type': 'chunk',
                'chunk_id': chunk_id,
                'chunk_text': chunk_text,
                'confidence': 1.0,
                'inferred': False
            })

    # Create document metadata for graph storage
    document_metadata = export_for_vector_db(ocr_result)

    return triples, document_metadata


def export_chunks_for_processing(ocr_result: OCRResult) -> List[Dict[str, Any]]:
    """Export chunks with metadata for further processing"""
    if not ocr_result.success or not ocr_result.chunk_metadata:
        return []

    chunks = []
    for chunk in ocr_result.chunk_metadata:
        chunk_data = {
            'chunk_id': chunk['chunk_id'],
            'chunk_text': chunk['chunk_text'],
            'word_count': chunk['word_count'],
            'character_count': chunk['character_count'],
            'start_word': chunk['start_word'],
            'end_word': chunk['end_word'],
            'source_file_id': ocr_result.file_metadata.get('file_id') if ocr_result.file_metadata else None,
            'source_filename': ocr_result.file_metadata.get('original_filename') if ocr_result.file_metadata else None,
            'extraction_method': ocr_result.method_used,
            'extraction_confidence': ocr_result.confidence,
            'document_type': ocr_result.document_classification.get(
                'document_type') if ocr_result.document_classification else None
        }
        chunks.append(chunk_data)

    return chunks


# FACTORY FUNCTIONS - Enhanced Configuration for LLM OCR Only

def create_enhanced_config() -> Dict[str, Any]:
    """Create optimized configuration for Enhanced LLM OCR with metadata"""
    return {
        # LLM OCR Configuration
        'LLM_OCR_PRIMARY_METHOD': 'gemini',  # Default to Gemini 1.5 Flash
        'LLM_OCR_FALLBACK_ENABLED': True,
        'LLM_OCR_TIMEOUT': 60,
        'LLM_OCR_MAX_RETRIES': 2,
        'LLM_OCR_CONFIDENCE_THRESHOLD': 0.7,

        # Enhanced Metadata Configuration
        'EXTRACT_ENTITIES': True,
        'CLASSIFY_DOCUMENTS': True,
        'ANALYZE_QUALITY': True,
        'CHUNK_SIZE': 1000,  # Words per chunk

        # Performance settings
        'ENABLE_OCR_TIMEOUT': True,
        'OCR_DEFAULT_TIMEOUT': 60,
        'ENABLE_PROGRESS_MONITORING': True,

        # Storage settings
        'OCR_STORAGE_SINGLETON': True,
        'SAVE_OCR_OUTPUTS': True,

        # File processing
        'VALIDATE_FILE_TYPES': True,
        'HANDLE_TEXT_FILES': True,
        'PDF_DPI': 200,  # For PDF to image conversion

        # Logging
        'REDUCE_DEBUG_LOGGING': True,
        'ENABLE_PERFORMANCE_LOGGING': True
    }


def create_ocr_pipeline(config: Dict[str, Any] = None) -> EnhancedOCRPipeline:
    """
    Factory function for creating Enhanced LLM OCR pipeline with metadata
    Maintains backward compatibility with original function
    """
    if config is None:
        config = create_enhanced_config()

    return EnhancedOCRPipeline(config)


# UTILITY FUNCTIONS

def get_supported_file_types() -> List[str]:
    """Get list of supported file types"""
    return [
        'application/pdf',
        'image/png',
        'image/jpeg',
        'image/jpg',
        'text/plain',
        'text/csv',
        'text/html',
        'text/xml'
    ]


def validate_api_keys(config: Dict[str, Any]) -> Dict[str, bool]:
    """Validate available API keys"""
    validation_results = {}

    # Check Gemini/Google API key
    gemini_key = (
            config.get('GOOGLE_API_KEY') or
            config.get('LLM_API_KEY') or
            config.get('GEMINI_API_KEY') or
            config.get('gemini_api_key')
    )
    validation_results['gemini'] = bool(gemini_key and GEMINI_AVAILABLE)

    # Check OpenAI API key
    openai_key = config.get('OPENAI_API_KEY')
    validation_results['openai'] = bool(openai_key and OPENAI_AVAILABLE)

    # Check Claude API key
    claude_key = config.get('ANTHROPIC_API_KEY')
    validation_results['claude'] = bool(claude_key and ANTHROPIC_AVAILABLE)

    # Check Mistral API key
    mistral_key = (
            config.get('MISTRAL_API_KEY') or
            config.get('mistral_api_key')
    )
    validation_results['mistral'] = bool(mistral_key and MISTRAL_AVAILABLE)

    return validation_results


def get_metadata_summary(ocr_result: OCRResult) -> Dict[str, Any]:
    """Get comprehensive metadata summary from OCR result"""
    if not ocr_result.success:
        return {"status": "failed", "error": ocr_result.error_message}

    summary = {
        "status": "success",
        "extraction_info": {
            "method": ocr_result.method_used,
            "confidence": ocr_result.confidence,
            "processing_time": ocr_result.processing_time,
            "text_length": len(ocr_result.text)
        },
        "file_info": ocr_result.file_metadata if ocr_result.file_metadata else {},
        "content_info": ocr_result.content_metadata if ocr_result.content_metadata else {},
        "quality_info": ocr_result.quality_metrics if ocr_result.quality_metrics else {},
        "classification": ocr_result.document_classification if ocr_result.document_classification else {},
        "entities": ocr_result.detected_entities if ocr_result.detected_entities else {},
        "chunks_created": len(ocr_result.chunk_metadata) if ocr_result.chunk_metadata else 0
    }

    return summary


# MAIN EXECUTION AND TESTING

if __name__ == "__main__":
    # Example usage and testing
    print("Enhanced LLM OCR Pipeline with Metadata - LLM OCR Only Version")
    print("=" * 60)

    # Create test configuration
    test_config = create_enhanced_config()

    # Show configuration
    print("Configuration:")
    for key, value in test_config.items():
        if 'API_KEY' not in key:
            print(f"  {key}: {value}")

    # Test API key validation
    api_status = validate_api_keys(test_config)
    print(f"\nAPI Key Status: {api_status}")

    # Test pipeline creation
    try:
        pipeline = create_ocr_pipeline(test_config)
        status = pipeline.get_status_info()
        print(f"\nPipeline Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Test provider availability
        providers = pipeline.get_available_providers()
        print(f"\nAvailable Providers: {providers}")

    except Exception as e:
        print(f"Pipeline creation failed: {e}")

    print(f"\nSupported File Types: {get_supported_file_types()}")

    print("\nEnhanced LLM OCR Pipeline with Metadata ready for integration!")
    print("Features: Entity Extraction | Document Classification | Quality Analysis | Chunking")
    print("LLM Providers: Gemini | OpenAI | Claude | Mistral")
    print("Ready for Vector DB and Graph DB integration with rich metadata!")