# src/utils/pipeline_validation.py

import logging
import json
import hashlib
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import your existing pipeline components
try:
    from src.knowledge_graph.text_utils import chunk_text
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError
    from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT, MAIN_USER_PROMPT
    import src.utils.processing_pipeline as pipeline
except ImportError as e:
    logging.warning(f"Could not import pipeline components for validation: {e}")


@dataclass
class ValidationResult:
    """Results from validating a single document."""
    file_name: str
    file_type: str
    file_size: int
    file_hash: str

    # OCR/Text Extraction Results
    mistral_ocr_success: bool
    mistral_text_length: int
    mistral_extraction_time: float
    ocr_error_message: Optional[str]

    # Text Processing Results
    chunking_success: bool
    num_chunks: int
    avg_chunk_length: float
    chunking_time: float

    # KG Extraction Results
    kg_extraction_success: bool
    num_triples_extracted: int
    extraction_time: float
    kg_error_message: Optional[str]

    # Quality Metrics
    text_quality_score: float
    contains_handwriting_indicators: bool
    contains_tables: bool
    contains_financial_data: bool

    # Overall Assessment
    processing_recommendation: str
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PipelineValidationProcessor:
    """Validates your actual pipeline components for accuracy and performance."""

    def __init__(self, config: Dict[str, Any], mistral_client: Any):
        self.config = config
        self.mistral_client = mistral_client
        self.logger = logging.getLogger(__name__)

        # Validation thresholds (configurable)
        self.min_text_length = 50
        self.max_extraction_time = 300  # 5 minutes
        self.min_quality_score = 0.6

        # Track common issues
        self.common_issues = {
            'ocr_failures': [],
            'chunking_issues': [],
            'kg_extraction_failures': [],
            'quality_concerns': []
        }

    def quick_validate_document(self, uploaded_file: Any) -> ValidationResult:
        """Quick validation for UI - simplified version with essential checks."""

        file_name = uploaded_file.name
        file_type = uploaded_file.type
        file_content = uploaded_file.getvalue()
        file_size = len(file_content)
        file_hash = hashlib.sha256(file_content).hexdigest()[:16]  # Short hash for UI

        # Initialize with defaults
        result = ValidationResult(
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_hash=file_hash,
            mistral_ocr_success=False,
            mistral_text_length=0,
            mistral_extraction_time=0.0,
            ocr_error_message=None,
            chunking_success=False,
            num_chunks=0,
            avg_chunk_length=0.0,
            chunking_time=0.0,
            kg_extraction_success=False,
            num_triples_extracted=0,
            extraction_time=0.0,
            kg_error_message=None,
            text_quality_score=0.0,
            contains_handwriting_indicators=False,
            contains_tables=False,
            contains_financial_data=False,
            processing_recommendation="Unknown",
            confidence_score=0.0
        )

        try:
            # Step 1: Test OCR/Text Extraction
            text_content = self._test_ocr_extraction_quick(uploaded_file, result)

            if text_content:
                # Step 2: Quick quality analysis
                self._analyze_text_quality_quick(text_content, result)

                # Step 3: Test chunking
                self._test_chunking_quick(text_content, result)

            # Step 4: Generate assessment
            self._generate_quick_assessment(result)

        except Exception as e:
            self.logger.error(f"Quick validation failed for {file_name}: {e}")
            result.ocr_error_message = str(e)
            result.processing_recommendation = "Failed - Validation Error"

        return result

    def _test_ocr_extraction_quick(self, uploaded_file: Any, result: ValidationResult) -> Optional[str]:
        """Quick OCR test using your actual pipeline."""

        if result.file_type == "text/plain":
            try:
                text_content = uploaded_file.getvalue().decode('utf-8', errors='replace')
                result.mistral_ocr_success = True
                result.mistral_text_length = len(text_content)
                result.mistral_extraction_time = 0.01
                return text_content
            except Exception as e:
                result.ocr_error_message = f"Text decode error: {str(e)}"
                return None

        # For PDFs and images
        if not self.mistral_client:
            result.ocr_error_message = "Mistral client not available"
            return None

        try:
            start_time = time.time()
            text_content = pipeline.process_uploaded_file_ocr(uploaded_file, self.mistral_client)
            result.mistral_extraction_time = time.time() - start_time

            if text_content:
                result.mistral_ocr_success = True
                result.mistral_text_length = len(text_content)
                return text_content
            else:
                result.ocr_error_message = "OCR returned no content"

        except Exception as e:
            result.ocr_error_message = str(e)
            self.logger.error(f"OCR failed for {result.file_name}: {e}")

        return None

    def _analyze_text_quality_quick(self, text_content: str, result: ValidationResult):
        """Quick text quality analysis."""

        if not text_content:
            return

        total_chars = len(text_content)
        quality_score = 0.0

        # Basic length check
        if total_chars > self.min_text_length:
            quality_score += 0.3

        # Business document indicators
        financial_indicators = [
            r'invoice', r'total', r'amount', r'\$[\d,]+', r'date',
            r'company', r'address', r'phone', r'email'
        ]
        found_indicators = sum(1 for pattern in financial_indicators
                               if re.search(pattern, text_content, re.IGNORECASE))

        if found_indicators >= 3:
            quality_score += 0.3
            result.contains_financial_data = True

        # Table detection
        if re.search(r'^\s*\|.*\|.*\|', text_content, re.MULTILINE) or \
                len(re.findall(r'\t', text_content)) > 5:
            quality_score += 0.2
            result.contains_tables = True

        # Handwriting indicators (OCR confusion patterns)
        handwriting_patterns = [
            r'[il1|]{3,}',  # Multiple similar characters
            r'[0oO]{3,}',  # O/0 confusion
            r'\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b'  # Fragmented text
        ]
        if any(re.search(pattern, text_content) for pattern in handwriting_patterns):
            result.contains_handwriting_indicators = True
            quality_score -= 0.1

        # Line structure quality
        lines = text_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            if avg_line_length > 20:
                quality_score += 0.2

        result.text_quality_score = max(0.0, min(1.0, quality_score))

    def _test_chunking_quick(self, text_content: str, result: ValidationResult):
        """Quick chunking test."""

        try:
            start_time = time.time()
            chunks = chunk_text(
                text_content,
                chunk_size=self.config.get('CHUNK_SIZE', 1000),
                chunk_overlap=self.config.get('CHUNK_OVERLAP', 100)
            )
            result.chunking_time = time.time() - start_time

            if chunks:
                result.chunking_success = True
                result.num_chunks = len(chunks)
                result.avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)

        except Exception as e:
            self.logger.error(f"Chunking failed for {result.file_name}: {e}")

    def _generate_quick_assessment(self, result: ValidationResult):
        """Generate quick assessment and confidence score."""

        # Calculate confidence based on key factors
        confidence = 0.0

        if result.mistral_ocr_success:
            confidence += 0.4

        if result.mistral_text_length > 200:
            confidence += 0.2

        confidence += result.text_quality_score * 0.25

        if result.chunking_success and result.num_chunks > 0:
            confidence += 0.15

        result.confidence_score = min(1.0, confidence)

        # Generate recommendation
        if result.confidence_score >= 0.8:
            result.processing_recommendation = "Excellent - Process automatically"
        elif result.confidence_score >= 0.6:
            result.processing_recommendation = "Good - Process with monitoring"
        elif result.confidence_score >= 0.4:
            result.processing_recommendation = "Fair - Manual review recommended"
        elif result.confidence_score >= 0.2:
            result.processing_recommendation = "Poor - Manual processing needed"
        else:
            result.processing_recommendation = "Failed - Check document quality"

    def validate_batch_quick(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Quick batch validation for UI."""

        results = []
        start_time = time.time()

        for uploaded_file in uploaded_files:
            try:
                result = self.quick_validate_document(uploaded_file)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Validation failed for {uploaded_file.name}: {e}")
                # Create minimal failed result
                failed_result = ValidationResult(
                    file_name=uploaded_file.name,
                    file_type=uploaded_file.type,
                    file_size=len(uploaded_file.getvalue()),
                    file_hash="error",
                    mistral_ocr_success=False,
                    mistral_text_length=0,
                    mistral_extraction_time=0.0,
                    ocr_error_message=str(e),
                    chunking_success=False,
                    num_chunks=0,
                    avg_chunk_length=0.0,
                    chunking_time=0.0,
                    kg_extraction_success=False,
                    num_triples_extracted=0,
                    extraction_time=0.0,
                    kg_error_message=str(e),
                    text_quality_score=0.0,
                    contains_handwriting_indicators=False,
                    contains_tables=False,
                    contains_financial_data=False,
                    processing_recommendation="Failed - Validation Error",
                    confidence_score=0.0
                )
                results.append(failed_result)

        total_time = time.time() - start_time

        # Generate summary report
        total_files = len(results)
        successful_ocr = sum(1 for r in results if r.mistral_ocr_success)
        avg_confidence = sum(r.confidence_score for r in results) / total_files if total_files > 0 else 0
        financial_docs = sum(1 for r in results if r.contains_financial_data)

        return {
            'summary': {
                'total_files': total_files,
                'total_validation_time': total_time,
                'ocr_success_rate': successful_ocr / total_files if total_files > 0 else 0,
                'average_confidence_score': avg_confidence,
                'financial_documents': financial_docs
            },
            'results': [result.to_dict() for result in results],
            'recommendations': self._generate_batch_recommendations(results)
        }

    def _generate_batch_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations for batch processing."""

        recommendations = []

        if not results:
            return ["No files processed"]

        total_files = len(results)
        excellent_files = sum(1 for r in results if r.confidence_score >= 0.8)
        poor_files = sum(1 for r in results if r.confidence_score < 0.4)
        handwriting_files = sum(1 for r in results if r.contains_handwriting_indicators)

        if excellent_files == total_files:
            recommendations.append("🚀 All files ready for automated processing!")
        elif excellent_files > total_files * 0.7:
            recommendations.append(f"Most files ({excellent_files}/{total_files}) ready for automated processing")
        else:
            recommendations.append(f"⚠️ Only {excellent_files}/{total_files} files ready for automated processing")

        if poor_files > 0:
            recommendations.append(f"{poor_files} files have quality issues and need manual review")

        if handwriting_files > 0:
            recommendations.append(f"✍️ {handwriting_files} files may contain handwriting - consider manual data entry")

        return recommendations


# Convenience function for Streamlit integration
def create_validator(config: Dict[str, Any], mistral_client: Any) -> PipelineValidationProcessor:
    """Factory function to create validator instance."""
    return PipelineValidationProcessor(config, mistral_client)