# enhanced_ocr_pipeline.py - BALANCED QUALITY & PERFORMANCE VERSION
"""
Enhanced OCR Pipeline with balanced approach:
- FIXED: Critical performance issues (storage loops, LLM config, warnings)
- PRESERVED: All quality enhancement features
- OPTIMIZED: Intelligent preprocessing with quality controls
- MAINTAINED: Full backward compatibility
"""

import logging
import cv2
import numpy as np
import easyocr
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading

# Suppress specific EasyOCR overflow warnings while preserving other warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="easyocr.utils",
                        message="overflow encountered in scalar add")

# PDF processing
try:
    import pdf2image
except ImportError:
    pdf2image = None
    print("Warning: pdf2image not installed. PDF processing will be limited.")

# Advanced image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from PIL.ImageFilter import SHARPEN, SMOOTH, DETAIL

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Some image enhancements will be limited.")

# OpenCV advanced features
try:
    import cv2.ximgproc as ximgproc

    OPENCV_CONTRIB_AVAILABLE = True
except ImportError:
    OPENCV_CONTRIB_AVAILABLE = False

# Super resolution (optional)
try:
    from cv2 import dnn_superres

    SUPER_RESOLUTION_AVAILABLE = True
except ImportError:
    SUPER_RESOLUTION_AVAILABLE = False

# OCR storage - FIXED: Proper import handling
try:
    from src.utils.ocr_storage import create_storage_manager

    OCR_STORAGE_AVAILABLE = True
except ImportError:
    print("Warning: OCR storage not available. Install dependencies or check path.")
    create_storage_manager = None
    OCR_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


# MAINTAINED: Original class names for backward compatibility
class OCRMethod(Enum):
    EASYOCR = "easyocr"
    MISTRAL = "mistral"
    HYBRID = "hybrid"


@dataclass
class OCRResult:
    """PRESERVED: Original OCRResult structure with all fields"""
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


class DocumentAnalyzer:
    """
    PRESERVED: Advanced document analysis for quality enhancement
    """

    @staticmethod
    def analyze_document_type(image: np.ndarray, file_name: str) -> Dict[str, Any]:
        """Analyze document characteristics for optimal processing"""
        try:
            # Document type hints from filename
            file_lower = file_name.lower()
            doc_hints = {
                'is_invoice': any(term in file_lower for term in ['invoice', 'bill', 'receipt']),
                'is_form': any(term in file_lower for term in ['form', 'application', 'wo-', 'work']),
                'is_table': any(term in file_lower for term in ['table', 'data', 'report']),
                'is_handwritten': any(term in file_lower for term in ['hand', 'written', 'signature'])
            }

            # Image characteristics
            height, width = image.shape[:2] if len(image.shape) >= 2 else (0, 0)

            # Check if grayscale
            is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and
                                                     np.allclose(image[:, :, 0], image[:, :, 1]) and
                                                     np.allclose(image[:, :, 1], image[:, :, 2]))

            # Estimate text density and quality
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Edge detection for text quality assessment
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height) if width * height > 0 else 0

            # Brightness and contrast analysis
            brightness = np.mean(gray)
            contrast = np.std(gray)

            return {
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'is_grayscale': is_grayscale,
                'brightness': brightness,
                'contrast': contrast,
                'edge_density': edge_density,
                'estimated_quality': 'high' if contrast > 50 and edge_density > 0.01 else 'low',
                'document_hints': doc_hints,
                'recommended_preprocessing': DocumentAnalyzer._recommend_preprocessing(
                    brightness, contrast, edge_density, doc_hints
                )
            }

        except Exception as e:
            logger.warning(f"Document analysis failed: {e}")
            return {
                'width': 0, 'height': 0, 'aspect_ratio': 1.0,
                'is_grayscale': True, 'brightness': 128, 'contrast': 30,
                'edge_density': 0.01, 'estimated_quality': 'medium',
                'document_hints': {}, 'recommended_preprocessing': ['basic']
            }

    @staticmethod
    def _recommend_preprocessing(brightness: float, contrast: float,
                                 edge_density: float, doc_hints: Dict) -> List[str]:
        """Recommend preprocessing based on document analysis"""
        recommendations = ['basic']  # Always include basic preprocessing

        # Quality-based recommendations
        if contrast < 30:
            recommendations.append('contrast_enhancement')

        if brightness < 80 or brightness > 200:
            recommendations.append('brightness_adjustment')

        if edge_density < 0.005:
            recommendations.append('sharpening')

        # Document type recommendations
        if doc_hints.get('is_invoice') or doc_hints.get('is_table'):
            recommendations.append('table_enhancement')

        if doc_hints.get('is_handwritten'):
            recommendations.append('handwriting_enhancement')

        if doc_hints.get('is_form'):
            recommendations.append('form_enhancement')

        return recommendations


class QualityPreservingPreprocessor:
    """
    ENHANCED: Quality-preserving preprocessor with intelligent optimization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # PRESERVED: All quality features with intelligent defaults
        self.enable_super_resolution = config.get('ENABLE_SUPER_RESOLUTION', True)
        self.enable_adaptive_preprocessing = config.get('ENABLE_ADAPTIVE_PREPROCESSING', True)
        self.enable_table_detection = config.get('ENABLE_TABLE_DETECTION', True)
        self.enable_handwriting_enhancement = config.get('ENABLE_HANDWRITING_ENHANCEMENT', True)
        self.enable_multi_scale_processing = config.get('ENABLE_MULTI_SCALE_PROCESSING', True)

        # OPTIMIZED: Intelligent quality controls
        self.max_variants_per_image = config.get('MAX_PREPROCESSING_VARIANTS', 6)  # Reduced from unlimited
        self.quality_threshold = config.get('PREPROCESSING_QUALITY_THRESHOLD', 0.7)
        self.adaptive_variant_selection = config.get('ADAPTIVE_VARIANT_SELECTION', True)

    def enhanced_preprocessing(self, image: np.ndarray, file_name: str,
                               document_analysis: Dict[str, Any]) -> Tuple[List[np.ndarray], List[str]]:
        """
        BALANCED: Enhanced preprocessing with quality preservation and performance optimization
        """
        processed_images = []
        applied_techniques = []

        try:
            orig_height, orig_width = image.shape[:2]
            logger.info(f"Starting enhanced preprocessing for {file_name}: {orig_width}x{orig_height}")

            # Get recommendations from document analysis
            recommendations = document_analysis.get('recommended_preprocessing', ['basic'])
            doc_quality = document_analysis.get('estimated_quality', 'medium')

            # Base image preparation
            base_image = self._prepare_base_image(image)
            applied_techniques.append('base_preparation')

            # PRESERVED: All preprocessing variants with intelligent selection
            variants_to_create = self._select_optimal_variants(recommendations, doc_quality)

            for variant_type in variants_to_create:
                try:
                    if variant_type == 'enhanced_contrast':
                        variant = self._create_enhanced_contrast_variant(base_image)
                        processed_images.append(variant)
                        applied_techniques.append('enhanced_contrast')

                    elif variant_type == 'table_optimized' and self.enable_table_detection:
                        variant = self._create_table_optimized_variant(base_image)
                        processed_images.append(variant)
                        applied_techniques.append('table_enhancement')

                    elif variant_type == 'handwriting_optimized' and self.enable_handwriting_enhancement:
                        variant = self._create_handwriting_variant(base_image)
                        processed_images.append(variant)
                        applied_techniques.append('handwriting_enhancement')

                    elif variant_type == 'super_resolution' and self.enable_super_resolution:
                        variant = self._create_super_resolution_variant(base_image, orig_width, orig_height)
                        if variant is not None:
                            processed_images.append(variant)
                            applied_techniques.append('super_resolution')

                    elif variant_type == 'multi_scale' and self.enable_multi_scale_processing:
                        variants = self._create_multi_scale_variants(base_image, orig_width, orig_height)
                        processed_images.extend(variants)
                        applied_techniques.append('multi_scale_processing')

                    # Stop if we've reached the maximum number of variants
                    if len(processed_images) >= self.max_variants_per_image:
                        break

                except Exception as e:
                    logger.warning(f"Failed to create {variant_type} variant: {e}")
                    continue

            # PRESERVED: Border padding for all variants
            final_variants = []
            for img in processed_images:
                bordered = self._add_border_padding(img)
                final_variants.append(bordered)

            applied_techniques.append('border_padding')

            logger.info(f"Enhanced preprocessing complete for {file_name}: "
                        f"Generated {len(final_variants)} quality variants, "
                        f"techniques: {', '.join(applied_techniques)}")

            return final_variants, applied_techniques

        except Exception as e:
            logger.error(f"Enhanced preprocessing failed for {file_name}: {e}")
            # Fallback to basic preprocessing
            return [self._prepare_base_image(image)], ['basic_fallback']

    def _select_optimal_variants(self, recommendations: List[str], doc_quality: str) -> List[str]:
        """OPTIMIZED: Intelligently select variants based on document analysis"""
        variants = ['enhanced_contrast']  # Always include enhanced contrast

        # Quality-based selection
        if doc_quality == 'low':
            variants.extend(['table_optimized', 'super_resolution'])
        elif doc_quality == 'high':
            variants.append('table_optimized')  # High quality docs need less processing

        # Recommendation-based selection
        if 'table_enhancement' in recommendations:
            variants.append('table_optimized')
        if 'handwriting_enhancement' in recommendations:
            variants.append('handwriting_optimized')
        if 'sharpening' in recommendations or doc_quality == 'low':
            variants.append('super_resolution')

        # Multi-scale for complex documents
        if len(recommendations) > 3 or 'form_enhancement' in recommendations:
            variants.append('multi_scale')

        # Remove duplicates and limit
        unique_variants = list(dict.fromkeys(variants))
        return unique_variants[:self.max_variants_per_image]

    def _prepare_base_image(self, image: np.ndarray) -> np.ndarray:
        """PRESERVED: High-quality base image preparation"""
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # PRESERVED: Advanced contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

        # PRESERVED: Noise reduction while preserving text
        processed = cv2.medianBlur(processed, 3)

        return processed

    def _create_enhanced_contrast_variant(self, image: np.ndarray) -> np.ndarray:
        """PRESERVED: Enhanced contrast variant"""
        # Advanced unsharp masking
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # Additional contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
        enhanced = clahe.apply(sharpened)

        return enhanced

    def _create_table_optimized_variant(self, image: np.ndarray) -> np.ndarray:
        """PRESERVED: Table detection optimized variant"""
        # Morphological operations for table structure
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Detect horizontal and vertical lines
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_horizontal)
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_vertical)

        # Combine with original
        table_enhanced = cv2.addWeighted(image, 0.7, horizontal + vertical, 0.3, 0)

        # Enhance text within tables
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(table_enhanced, -1, kernel)

        return sharpened

    def _create_handwriting_variant(self, image: np.ndarray) -> np.ndarray:
        """PRESERVED: Handwriting enhancement variant"""
        # Bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)

        # Enhance thin strokes
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(smoothed, -1, kernel)

        # Adaptive thresholding for handwriting
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

        return adaptive

    def _create_super_resolution_variant(self, image: np.ndarray,
                                         orig_width: int, orig_height: int) -> Optional[np.ndarray]:
        """PRESERVED: Super resolution variant with optimization"""
        try:
            # Only apply super resolution if image is small or low quality
            if orig_width < 1000 or orig_height < 1000:
                scale_factor = 2.0 if orig_width < 500 else 1.5

                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)

                # High-quality interpolation
                upscaled = cv2.resize(image, (new_width, new_height),
                                      interpolation=cv2.INTER_CUBIC)

                # Apply sharpening to upscaled image
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(upscaled, -1, kernel)

                return sharpened

            return None  # Skip super resolution for large images

        except Exception as e:
            logger.warning(f"Super resolution failed: {e}")
            return None

    def _create_multi_scale_variants(self, image: np.ndarray,
                                     orig_width: int, orig_height: int) -> List[np.ndarray]:
        """PRESERVED: Multi-scale processing variants"""
        variants = []

        try:
            # Create different scale variants for complex documents
            scales = [1.2, 1.5] if orig_width < 1200 else [1.1]

            for scale in scales:
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)

                scaled = cv2.resize(image, (new_width, new_height),
                                    interpolation=cv2.INTER_LANCZOS4)

                # Apply different enhancement to each scale
                if scale == 1.2:
                    # Light enhancement
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(scaled)
                else:
                    # Stronger enhancement
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
                    enhanced = clahe.apply(scaled)

                variants.append(enhanced)

        except Exception as e:
            logger.warning(f"Multi-scale processing failed: {e}")

        return variants

    def _add_border_padding(self, image: np.ndarray) -> np.ndarray:
        """PRESERVED: Border padding for better OCR"""
        border_size = 50  # Increased for better OCR accuracy
        return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                  cv2.BORDER_CONSTANT, value=255)


# FIXED: Singleton storage manager to prevent initialization loops
class StorageManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def initialize(self, storage_path: str = "ocr_outputs"):
        """Initialize storage only once"""
        if not self._initialized and OCR_STORAGE_AVAILABLE:
            try:
                self.storage = create_storage_manager(storage_path)
                self._initialized = True
                logger.info("OCR storage manager initialized (singleton)")
            except Exception as e:
                logger.error(f"Failed to initialize OCR storage: {e}")
                self.storage = None
        elif not OCR_STORAGE_AVAILABLE:
            self.storage = None

    def get_storage(self):
        return getattr(self, 'storage', None)


# MAINTAINED: Original class name for backward compatibility
class EnhancedOCRPipeline:
    """
    BALANCED: Enhanced OCR Pipeline with quality preservation and performance optimization
    - FIXED: All critical issues (storage loops, LLM config, warnings)
    - PRESERVED: All quality enhancement features
    - OPTIMIZED: Intelligent preprocessing selection
    - MAINTAINED: Full backward compatibility
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with balanced quality and performance settings"""
        self.config = config
        self.easyocr_reader = None
        self.mistral_client = None

        # FIXED: Use singleton storage manager
        self.storage_manager = StorageManager()
        self.storage_manager.initialize()

        # PRESERVED: All OCR configuration options
        self.primary_method = config.get('OCR_PRIMARY_METHOD', 'easyocr')
        self.fallback_enabled = config.get('OCR_FALLBACK_ENABLED', True)
        self.easyocr_enabled = config.get('EASYOCR_ENABLED', True)
        self.easyocr_gpu = config.get('EASYOCR_GPU', True)
        self.easyocr_languages = config.get('EASYOCR_LANGUAGES', ['en'])

        # BALANCED: Quality-focused thresholds with performance consideration
        self.confidence_threshold = config.get('OCR_CONFIDENCE_THRESHOLD', 0.2)  # Slightly higher
        self.text_threshold = config.get('OCR_TEXT_THRESHOLD', 0.3)  # Balanced
        self.low_text_threshold = config.get('OCR_LOW_TEXT_THRESHOLD', 0.25)  # Balanced

        # PRESERVED: Quality enhancement features
        self.document_analyzer = DocumentAnalyzer()
        self.preprocessor = QualityPreservingPreprocessor(config)

        # Initialize components
        self._initialize_easyocr()

    def _initialize_easyocr(self):
        """FIXED: Initialize EasyOCR with warning suppression"""
        if not self.easyocr_enabled:
            logger.info("EasyOCR disabled in configuration")
            return

        try:
            logger.info(f"Initializing EasyOCR with languages: {self.easyocr_languages}, GPU: {self.easyocr_gpu}")

            # FIXED: Suppress warnings during initialization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="easyocr")

                self.easyocr_reader = easyocr.Reader(
                    self.easyocr_languages,
                    gpu=self.easyocr_gpu,
                    verbose=False,
                    download_enabled=True
                )

            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None

    def set_mistral_client(self, mistral_client):
        """PRESERVED: Set Mistral client for fallback OCR"""
        self.mistral_client = mistral_client
        logger.info("Mistral client set for OCR fallback")

    def _extract_with_advanced_easyocr(self, uploaded_file) -> OCRResult:
        """
        BALANCED: Advanced EasyOCR extraction with quality preservation
        """
        if not self.easyocr_reader:
            return OCRResult(
                success=False, text="", confidence=0.0, method_used="easyocr",
                processing_time=0.0, text_regions_detected=0, preprocessing_applied=[],
                error_message="EasyOCR not initialized"
            )

        start_time = time.time()
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        logger.info(f"Processing {file_name} with Advanced EasyOCR...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / file_name

                # Save uploaded file
                with open(temp_path, "wb") as tmp:
                    tmp.write(uploaded_file.getvalue())

                # Load image based on file type
                if file_type == "application/pdf":
                    if pdf2image is None:
                        raise ValueError("pdf2image not installed. Cannot process PDF files.")

                    # BALANCED: Quality PDF processing
                    images = pdf2image.convert_from_bytes(
                        uploaded_file.getvalue(),
                        dpi=250,  # Higher DPI for quality
                        first_page=1,
                        last_page=1,
                        fmt='png'
                    )

                    if not images:
                        raise ValueError("No pages extracted from PDF")

                    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                else:
                    image = cv2.imread(str(temp_path))

                if image is None:
                    raise ValueError(f"Could not load image from {file_name}")

                # PRESERVED: Document analysis for optimal processing
                document_analysis = self.document_analyzer.analyze_document_type(image, file_name)

                # BALANCED: Enhanced preprocessing with quality focus
                processed_images, preprocessing_applied = self.preprocessor.enhanced_preprocessing(
                    image, file_name, document_analysis
                )

                # PRESERVED: Extract text from all variants
                all_results = []
                for i, img in enumerate(processed_images):
                    try:
                        # FIXED: EasyOCR with warning suppression
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning, module="easyocr")

                            results = self.easyocr_reader.readtext(
                                img,
                                detail=1,
                                paragraph=False,
                                text_threshold=self.text_threshold,
                                low_text=self.low_text_threshold,
                                link_threshold=0.4,
                                width_ths=0.7,
                                height_ths=0.7,
                                decoder='beamsearch',  # Higher quality decoder
                                beamWidth=5,
                                batch_size=1
                            )

                        # Process results with quality filtering
                        for (bbox, text, confidence) in results:
                            cleaned_text = text.strip()
                            if confidence >= self.confidence_threshold and len(cleaned_text) > 0:
                                all_results.append({
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'variant': i,
                                    'preprocessing': preprocessing_applied[i] if i < len(
                                        preprocessing_applied) else 'unknown'
                                })

                    except Exception as e:
                        logger.warning(f"Error processing image variant {i}: {e}")
                        continue

                # PRESERVED: Advanced result combination for quality
                combined_text, avg_confidence, total_regions = self._combine_results_with_quality_focus(all_results)

                processing_time = time.time() - start_time

                logger.info(f"Advanced EasyOCR completed for {file_name}: "
                            f"{total_regions} regions processed, "
                            f"{len(combined_text)} chars extracted, "
                            f"avg confidence: {avg_confidence:.3f}, "
                            f"time: {processing_time:.2f}s")

                return OCRResult(
                    success=True,
                    text=combined_text,
                    confidence=avg_confidence,
                    method_used="advanced_easyocr",
                    processing_time=processing_time,
                    text_regions_detected=total_regions,
                    preprocessing_applied=preprocessing_applied,
                    structured_data={'document_analysis': document_analysis}
                )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Advanced EasyOCR failed for {file_name}: {e}")
            return OCRResult(
                success=False, text="", confidence=0.0, method_used="easyocr",
                processing_time=processing_time, text_regions_detected=0,
                preprocessing_applied=[], error_message=str(e)
            )

    def _combine_results_with_quality_focus(self, all_results: List[Dict]) -> Tuple[str, float, int]:
        """
        PRESERVED: Advanced result combination with quality focus
        """
        if not all_results:
            return "", 0.0, 0

        # Group by text similarity for deduplication
        text_groups = {}
        for result in all_results:
            text = result['text'].strip().lower()
            if text not in text_groups:
                text_groups[text] = []
            text_groups[text].append(result)

        # Select best result from each group
        final_results = []
        for text, group in text_groups.items():
            if len(text) <= 1:  # Skip single characters
                continue

            # Select result with highest confidence
            best_result = max(group, key=lambda x: x['confidence'])
            final_results.append(best_result)

        # Sort by confidence and position (if bbox available)
        final_results.sort(key=lambda x: (-x['confidence'], x.get('bbox', [[0, 0]])[0][1]))

        # Combine texts
        combined_texts = [r['text'] for r in final_results]
        final_text = ' '.join(combined_texts)

        # Calculate weighted average confidence
        if final_results:
            total_weight = sum(len(r['text']) for r in final_results)
            if total_weight > 0:
                avg_confidence = sum(r['confidence'] * len(r['text']) for r in final_results) / total_weight
            else:
                avg_confidence = np.mean([r['confidence'] for r in final_results])
        else:
            avg_confidence = 0.0

        return final_text, avg_confidence, len(all_results)

    def extract_text(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """
        PRESERVED: Main extraction method with all features
        """
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        logger.info(f"Starting OCR extraction for {file_name} (type: {file_type})")

        # Handle text files directly
        if file_type in ['text/plain', 'text/csv', 'text/html', 'text/xml']:
            return self._handle_text_file_directly(uploaded_file, save_to_disk)

        # Proceed with advanced OCR
        if self.primary_method == "easyocr" and self.easyocr_reader:
            result = self._extract_with_advanced_easyocr(uploaded_file)
        else:
            result = OCRResult(
                success=False, text="", confidence=0.0, method_used="none",
                processing_time=0.0, text_regions_detected=0, preprocessing_applied=[],
                error_message="No OCR method available"
            )

        # FIXED: Save results without repeated storage initialization
        if save_to_disk and result.success:
            storage = self.storage_manager.get_storage()
            if storage:
                try:
                    # PRESERVED: Comprehensive metadata saving
                    ocr_metadata = {
                        'ocr_method': result.method_used,
                        'confidence': result.confidence,
                        'processing_time': result.processing_time,
                        'text_length': len(result.text),
                        'text_regions_detected': result.text_regions_detected,
                        'preprocessing_applied': result.preprocessing_applied,
                        'extracted_at': datetime.now().isoformat(),
                        'file_info': {
                            'name': file_name,
                            'type': file_type,
                            'size': len(uploaded_file.getvalue())
                        }
                    }

                    if result.structured_data:
                        ocr_metadata.update(result.structured_data)

                    saved_files = storage.save_ocr_output(
                        uploaded_file=uploaded_file,
                        ocr_text=result.text,
                        structured_data=ocr_metadata
                    )
                    result.saved_files = saved_files
                    logger.info(f"OCR output saved for {file_name}: {len(saved_files)} files")

                except Exception as e:
                    logger.error(f"Failed to save OCR output for {file_name}: {e}")

        return result

    def _handle_text_file_directly(self, uploaded_file, save_to_disk: bool) -> OCRResult:
        """
        PRESERVED: Direct text file handling
        """
        start_time = time.time()
        file_name = uploaded_file.name

        try:
            # Read the text content directly
            text_content = uploaded_file.getvalue()

            # Handle different encodings
            if isinstance(text_content, bytes):
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                    try:
                        decoded_text = text_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use utf-8 with error handling
                    decoded_text = text_content.decode('utf-8', errors='replace')
            else:
                decoded_text = str(text_content)

            processing_time = time.time() - start_time

            logger.info(f"Text file processed directly: {file_name}, length: {len(decoded_text)}")

            result = OCRResult(
                success=True,
                text=decoded_text,
                confidence=1.0,  # Perfect confidence for text files
                method_used="direct_text",
                processing_time=processing_time,
                text_regions_detected=1,
                preprocessing_applied=["direct_text_extraction"]
            )

            # FIXED: Save text file results
            if save_to_disk:
                storage = self.storage_manager.get_storage()
                if storage:
                    try:
                        ocr_metadata = {
                            'file_type': 'text',
                            'processing_method': 'direct_text_extraction',
                            'extracted_at': datetime.now().isoformat()
                        }

                        saved_files = storage.save_ocr_output(
                            uploaded_file=uploaded_file,
                            ocr_text=result.text,
                            structured_data=ocr_metadata
                        )
                        result.saved_files = saved_files
                        logger.info(f"Text file output saved: {len(saved_files)} files")
                    except Exception as e:
                        logger.error(f"Failed to save text file output: {e}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process text file {file_name}: {e}")

            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                method_used="direct_text",
                processing_time=processing_time,
                text_regions_detected=0,
                preprocessing_applied=[],
                error_message=f"Text file processing failed: {str(e)}"
            )

    def extract_text_with_fallback(self, uploaded_file, save_to_disk: bool = True) -> OCRResult:
        """
        PRESERVED: Extraction with Mistral fallback for maximum quality
        """
        # Try primary method first
        primary_result = self.extract_text(uploaded_file, save_to_disk=False)  # Don't save twice

        # If primary method succeeded with good confidence, return it
        if primary_result.success and primary_result.confidence >= 0.7:
            if save_to_disk:
                # Save the successful result
                storage = self.storage_manager.get_storage()
                if storage and primary_result.text:
                    try:
                        saved_files = storage.save_ocr_output(
                            uploaded_file=uploaded_file,
                            ocr_text=primary_result.text,
                            structured_data={'method': 'primary_success'}
                        )
                        primary_result.saved_files = saved_files
                    except Exception as e:
                        logger.error(f"Failed to save primary result: {e}")

            return primary_result

        # Try Mistral fallback if enabled and available
        if self.fallback_enabled and self.mistral_client:
            logger.info(f"Primary OCR confidence low ({primary_result.confidence:.3f}), trying Mistral fallback...")

            try:
                fallback_result = self._extract_with_mistral_fallback(uploaded_file)

                # Compare results and choose the best one
                if fallback_result.success:
                    if fallback_result.confidence > primary_result.confidence or not primary_result.success:
                        logger.info(
                            f"Mistral fallback provided better result (confidence: {fallback_result.confidence:.3f})")

                        if save_to_disk:
                            storage = self.storage_manager.get_storage()
                            if storage:
                                try:
                                    saved_files = storage.save_ocr_output(
                                        uploaded_file=uploaded_file,
                                        ocr_text=fallback_result.text,
                                        structured_data={'method': 'mistral_fallback'}
                                    )
                                    fallback_result.saved_files = saved_files
                                except Exception as e:
                                    logger.error(f"Failed to save fallback result: {e}")

                        return fallback_result
                    else:
                        logger.info("Primary result still better than fallback")

            except Exception as e:
                logger.error(f"Mistral fallback failed: {e}")

        # Return primary result (even if low confidence) if fallback didn't help
        if save_to_disk and primary_result.success:
            storage = self.storage_manager.get_storage()
            if storage:
                try:
                    saved_files = storage.save_ocr_output(
                        uploaded_file=uploaded_file,
                        ocr_text=primary_result.text,
                        structured_data={'method': 'primary_only'}
                    )
                    primary_result.saved_files = saved_files
                except Exception as e:
                    logger.error(f"Failed to save primary result: {e}")

        return primary_result

    def _extract_with_mistral_fallback(self, uploaded_file) -> OCRResult:
        """
        PRESERVED: Mistral fallback extraction for complex documents
        """
        if not self.mistral_client:
            return OCRResult(
                success=False, text="", confidence=0.0, method_used="mistral",
                processing_time=0.0, text_regions_detected=0, preprocessing_applied=[],
                error_message="Mistral client not available"
            )

        start_time = time.time()
        file_name = uploaded_file.name

        try:
            import base64

            # Convert image to base64 for Mistral
            file_content = uploaded_file.getvalue()

            if uploaded_file.type == "application/pdf":
                # Convert PDF to image first
                if pdf2image is None:
                    raise ValueError("pdf2image not installed for PDF processing")

                images = pdf2image.convert_from_bytes(file_content, dpi=200, first_page=1, last_page=1)
                if not images:
                    raise ValueError("No pages extracted from PDF")

                # Convert PIL image to bytes
                import io
                img_buffer = io.BytesIO()
                images[0].save(img_buffer, format='PNG')
                file_content = img_buffer.getvalue()

            # Encode to base64
            base64_content = base64.b64encode(file_content).decode('utf-8')

            # Prepare Mistral request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract all text from this image. Focus on accuracy and maintaining the original text structure. Return only the extracted text without any additional commentary."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{'png' if uploaded_file.type != 'application/pdf' else 'png'};base64,{base64_content}"
                            }
                        }
                    ]
                }
            ]

            # Call Mistral API
            response = self.mistral_client.chat(
                model="pixtral-12b-2409",
                messages=messages,
                max_tokens=2000
            )

            extracted_text = response.choices[0].message.content

            processing_time = time.time() - start_time

            # Estimate confidence based on text quality
            confidence = self._estimate_mistral_confidence(extracted_text)

            logger.info(f"Mistral fallback completed for {file_name}: "
                        f"{len(extracted_text)} chars extracted, "
                        f"estimated confidence: {confidence:.3f}, "
                        f"time: {processing_time:.2f}s")

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=confidence,
                method_used="mistral_fallback",
                processing_time=processing_time,
                text_regions_detected=1,  # Mistral doesn't provide region info
                preprocessing_applied=["mistral_vision"]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Mistral fallback failed for {file_name}: {e}")
            return OCRResult(
                success=False, text="", confidence=0.0, method_used="mistral",
                processing_time=processing_time, text_regions_detected=0,
                preprocessing_applied=[], error_message=str(e)
            )

    def _estimate_mistral_confidence(self, text: str) -> float:
        """
        PRESERVED: Estimate confidence for Mistral-extracted text
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        # Basic quality indicators
        has_complete_words = len([w for w in text.split() if len(w) > 2]) > 0
        has_punctuation = any(p in text for p in '.!?,:;')
        has_proper_spacing = '  ' not in text.replace('\n', ' ')

        # Calculate base confidence
        base_confidence = 0.8  # Mistral is generally high quality

        if has_complete_words:
            base_confidence += 0.1
        if has_punctuation:
            base_confidence += 0.05
        if has_proper_spacing:
            base_confidence += 0.05

        return min(base_confidence, 0.95)  # Cap at 95%


def create_enhanced_config() -> Dict[str, Any]:
    """
    BALANCED: Create configuration that preserves quality while optimizing performance
    """
    return {
        # Basic OCR settings
        'EASYOCR_ENABLED': True,
        'EASYOCR_GPU': True,
        'EASYOCR_LANGUAGES': ['en'],
        'OCR_PRIMARY_METHOD': 'easyocr',
        'OCR_FALLBACK_ENABLED': True,  # PRESERVED for quality

        # BALANCED: Quality-focused thresholds
        'OCR_CONFIDENCE_THRESHOLD': 0.2,  # Lower for quality
        'OCR_TEXT_THRESHOLD': 0.3,  # Balanced
        'OCR_LOW_TEXT_THRESHOLD': 0.25,  # Balanced

        # PRESERVED: All quality enhancement features
        'ENABLE_SUPER_RESOLUTION': True,
        'ENABLE_ADAPTIVE_PREPROCESSING': True,
        'ENABLE_TABLE_DETECTION': True,
        'ENABLE_HANDWRITING_ENHANCEMENT': True,
        'ENABLE_MULTI_SCALE_PROCESSING': True,

        # OPTIMIZED: Performance controls for quality features
        'MAX_PREPROCESSING_VARIANTS': 6,  # Limit variants but keep quality
        'PREPROCESSING_QUALITY_THRESHOLD': 0.7,
        'ADAPTIVE_VARIANT_SELECTION': True,

        # PDF processing - BALANCED quality
        'PDF_DPI': 250,  # Higher DPI for quality documents

        # Advanced features - PRESERVED
        'ENABLE_DOCUMENT_ANALYSIS': True,
        'ENABLE_ADVANCED_DEDUPLICATION': True,
        'ENABLE_QUALITY_ASSESSMENT': True,

        # File type handling
        'VALIDATE_FILE_TYPES': True,
        'HANDLE_TEXT_FILES': True,

        # FIXED: Storage optimization
        'OCR_STORAGE_SINGLETON': True,
        'REDUCE_DEBUG_LOGGING': True
    }


# MAINTAINED: Backward compatibility
def create_ocr_pipeline(config: Dict[str, Any] = None) -> EnhancedOCRPipeline:
    """
    Factory function for creating OCR pipeline with backward compatibility
    """
    if config is None:
        config = create_enhanced_config()

    return EnhancedOCRPipeline(config)