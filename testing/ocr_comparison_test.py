#!/usr/bin/env python3
"""
OCR Pipeline Comparison Test Script - FIXED VERSION with Correct API
Compares MinerU vs Enhanced EasyOCR performance on the same documents
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
from dataclasses import dataclass, asdict
import hashlib
import sys

# Fix Windows Unicode issues
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_comparison.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Standard result format for OCR comparison"""
    method: str
    success: bool
    text: str
    char_count: int
    confidence: float
    processing_time: float
    error_message: str = ""
    structured_data: Optional[Dict] = None
    regions_detected: int = 0
    total_regions: int = 0
    file_size_mb: float = 0.0

    def to_dict(self):
        return asdict(self)


class MinerUExtractor:
    """MinerU PDF extraction wrapper using current API"""

    def __init__(self):
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if MinerU is installed and available"""
        try:
            # Test the new API imports from v1.3.12
            import magic_pdf
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod

            logger.info("✓ MinerU v1.3.12+ API is available")
            return True

        except ImportError as e:
            logger.warning(f"✗ MinerU v1.3.12+ API not available: {e}")

            # Try older API as fallback
            try:
                from magic_pdf.pipe.UNIPipe import UNIPipe
                from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
                logger.info("✓ MinerU legacy API is available")
                return True
            except ImportError as e2:
                logger.warning(f"✗ MinerU legacy API also not available: {e2}")
                logger.info("Install with: pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com")
                return False

    def extract(self, pdf_path: str, output_dir: str = None) -> OCRResult:
        """Extract content using MinerU with current API"""
        if not self.available:
            return OCRResult(
                method="mineru",
                success=False,
                text="",
                char_count=0,
                confidence=0.0,
                processing_time=0.0,
                error_message="MinerU not installed"
            )

        start_time = time.time()
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        try:
            # Try new API first
            return self._extract_with_new_api(pdf_path, output_dir, start_time, file_size_mb)
        except Exception as e1:
            logger.warning(f"New API failed: {e1}")
            try:
                # Fallback to old API
                return self._extract_with_old_api(pdf_path, output_dir, start_time, file_size_mb)
            except Exception as e2:
                processing_time = time.time() - start_time
                logger.error(f"Both APIs failed. New: {e1}, Old: {e2}")
                return OCRResult(
                    method="mineru",
                    success=False,
                    text="",
                    char_count=0,
                    confidence=0.0,
                    processing_time=processing_time,
                    error_message=f"All APIs failed: {str(e2)}",
                    file_size_mb=file_size_mb
                )

    def _extract_with_new_api(self, pdf_path: str, output_dir: str, start_time: float,
                              file_size_mb: float) -> OCRResult:
        """Extract using v1.3.12+ API"""
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "mineru_output")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare writers
        local_image_dir = os.path.join(output_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(output_dir)

        # Read PDF bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)

        # Create dataset
        ds = PymuDocDataset(pdf_bytes)

        # Classify and process
        parse_method = ds.classify()
        logger.info(f"MinerU detected parse method: {parse_method}")

        if parse_method == SupportedPdfParseMethod.OCR:
            # OCR mode
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            # Text mode
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # Generate markdown
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        image_dir = os.path.basename(local_image_dir)
        md_content = pipe_result.dump_md(md_writer, f"{pdf_name}.md", image_dir)

        processing_time = time.time() - start_time

        # Extract text content
        if hasattr(pipe_result, 'get_text'):
            text_content = pipe_result.get_text()
        elif isinstance(md_content, str):
            text_content = md_content
        else:
            text_content = str(md_content) if md_content else ""

        return OCRResult(
            method="mineru_v1.3",
            success=True,
            text=text_content,
            char_count=len(text_content),
            confidence=0.95,  # MinerU doesn't provide confidence scores
            processing_time=processing_time,
            structured_data={"parse_method": str(parse_method)},
            file_size_mb=file_size_mb
        )

    def _extract_with_old_api(self, pdf_path: str, output_dir: str, start_time: float,
                              file_size_mb: float) -> OCRResult:
        """Extract using legacy API as fallback"""
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "mineru_output")
        os.makedirs(output_dir, exist_ok=True)

        # Process with old API
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        output_path = os.path.join(output_dir, pdf_name)
        os.makedirs(output_path, exist_ok=True)

        image_dir = os.path.join(output_path, "images")
        os.makedirs(image_dir, exist_ok=True)

        image_writer = DiskReaderWriter(image_dir)

        # Read PDF
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # Create pipeline
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)

        # Process
        pipe.pipe_classify()
        pipe.pipe_parse()

        # Generate markdown
        md_content = pipe.pipe_mk_markdown(os.path.basename(image_dir), drop_mode="none")

        processing_time = time.time() - start_time

        return OCRResult(
            method="mineru_legacy",
            success=True,
            text=md_content,
            char_count=len(md_content),
            confidence=0.95,
            processing_time=processing_time,
            file_size_mb=file_size_mb
        )


class EasyOCRExtractor:
    """Enhanced EasyOCR extraction wrapper (your current pipeline)"""

    def __init__(self):
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            import cv2
            import fitz  # PyMuPDF
            logger.info("✓ EasyOCR pipeline is available")
            return True
        except ImportError as e:
            logger.warning(f"✗ EasyOCR not available: {e}")
            return False

    def extract(self, pdf_path: str) -> OCRResult:
        """Extract content using Enhanced EasyOCR (simulating your current pipeline)"""
        if not self.available:
            return OCRResult(
                method="easyocr",
                success=False,
                text="",
                char_count=0,
                confidence=0.0,
                processing_time=0.0,
                error_message="EasyOCR not installed"
            )

        start_time = time.time()
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        try:
            import easyocr
            import cv2
            import fitz
            import numpy as np

            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            # Open PDF
            doc = fitz.open(pdf_path)

            all_text = []
            total_regions = 0
            regions_detected = 0
            confidences = []

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # Convert to image with enhancement
                mat = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scale
                img_data = mat.tobytes("png")

                # Convert to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Apply your enhanced preprocessing (simplified version)
                # Invoice enhancement
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Border padding
                border_size = 50
                enhanced = cv2.copyMakeBorder(
                    enhanced, border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT, value=255
                )

                # OCR with lower confidence threshold (your key fix)
                results = reader.readtext(
                    enhanced,
                    width_ths=0.5,  # Your optimized settings
                    height_ths=0.5,
                    text_threshold=0.2,  # Lower threshold
                    low_text=0.2  # Your key improvement
                )

                total_regions += len(results)

                # Filter results by confidence (your 0.2 threshold)
                for (bbox, text, confidence) in results:
                    if confidence >= 0.2:  # Your working threshold
                        all_text.append(text)
                        confidences.append(confidence)
                        regions_detected += 1

            doc.close()

            # Combine text
            combined_text = " ".join(all_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            processing_time = time.time() - start_time

            return OCRResult(
                method="easyocr_enhanced",
                success=True,
                text=combined_text,
                char_count=len(combined_text),
                confidence=avg_confidence,
                processing_time=processing_time,
                regions_detected=regions_detected,
                total_regions=total_regions,
                file_size_mb=file_size_mb
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"EasyOCR extraction failed: {e}")
            logger.error(traceback.format_exc())

            return OCRResult(
                method="easyocr_enhanced",
                success=False,
                text="",
                char_count=0,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e),
                file_size_mb=file_size_mb
            )


class OCRComparator:
    """Main comparison class"""

    def __init__(self, output_dir: str = "ocr_comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.mineru = MinerUExtractor()
        self.easyocr = EasyOCRExtractor()

        self.results = []

    def compare_single_document(self, pdf_path: str) -> Dict[str, OCRResult]:
        """Compare both methods on a single document"""
        logger.info(f"Comparing extraction methods for: {pdf_path}")

        results = {}

        # Test EasyOCR (your current pipeline)
        logger.info("Testing Enhanced EasyOCR...")
        easyocr_result = self.easyocr.extract(pdf_path)
        results['easyocr'] = easyocr_result

        # Test MinerU
        logger.info("Testing MinerU...")
        mineru_output = self.output_dir / "mineru_temp"
        mineru_result = self.mineru.extract(pdf_path, str(mineru_output))
        results['mineru'] = mineru_result

        return results

    def compare_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Compare both methods on multiple documents"""
        logger.info(f"Starting comparison of {len(pdf_paths)} documents...")

        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(pdf_paths),
            'results': [],
            'summary': {}
        }

        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing document {i}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")

            try:
                # Get file hash for identification
                file_hash = self._get_file_hash(pdf_path)

                # Compare methods
                results = self.compare_single_document(pdf_path)

                # Store results
                document_result = {
                    'file_path': pdf_path,
                    'file_name': os.path.basename(pdf_path),
                    'file_hash': file_hash,
                    'file_size_mb': os.path.getsize(pdf_path) / (1024 * 1024),
                    'easyocr': results['easyocr'].to_dict(),
                    'mineru': results['mineru'].to_dict(),
                    'comparison': self._compare_results(results['easyocr'], results['mineru'])
                }

                comparison_results['results'].append(document_result)

                # Log immediate results
                self._log_comparison(document_result)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                logger.error(traceback.format_exc())

        # Generate summary
        comparison_results['summary'] = self._generate_summary(comparison_results['results'])

        # Save results
        self._save_results(comparison_results)

        return comparison_results

    def _compare_results(self, easyocr_result: OCRResult, mineru_result: OCRResult) -> Dict[str, Any]:
        """Compare two OCR results"""
        comparison = {}

        # Success rate
        comparison['both_successful'] = easyocr_result.success and mineru_result.success
        comparison['easyocr_successful'] = easyocr_result.success
        comparison['mineru_successful'] = mineru_result.success

        if easyocr_result.success and mineru_result.success:
            # Character count comparison
            comparison['char_count_difference'] = mineru_result.char_count - easyocr_result.char_count
            comparison['char_count_improvement_pct'] = (
                    (mineru_result.char_count - easyocr_result.char_count) /
                    max(easyocr_result.char_count, 1) * 100
            )

            # Processing time comparison
            comparison['time_difference'] = mineru_result.processing_time - easyocr_result.processing_time
            comparison['time_ratio'] = mineru_result.processing_time / max(easyocr_result.processing_time, 0.001)

            # Confidence comparison
            comparison['confidence_difference'] = mineru_result.confidence - easyocr_result.confidence

            # Text similarity (basic)
            comparison['text_similarity'] = self._calculate_similarity(
                easyocr_result.text, mineru_result.text
            )

            # Winner determination
            comparison['winner'] = self._determine_winner(easyocr_result, mineru_result)

        return comparison

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _determine_winner(self, easyocr_result: OCRResult, mineru_result: OCRResult) -> str:
        """Determine which method performed better"""
        score_easyocr = 0
        score_mineru = 0

        # Character count (weight: 40%)
        if mineru_result.char_count > easyocr_result.char_count:
            score_mineru += 4
        elif easyocr_result.char_count > mineru_result.char_count:
            score_easyocr += 4

        # Confidence (weight: 30%)
        if mineru_result.confidence > easyocr_result.confidence:
            score_mineru += 3
        elif easyocr_result.confidence > mineru_result.confidence:
            score_easyocr += 3

        # Processing speed (weight: 20%, faster is better)
        if easyocr_result.processing_time < mineru_result.processing_time:
            score_easyocr += 2
        elif mineru_result.processing_time < easyocr_result.processing_time:
            score_mineru += 2

        # Structured data availability (weight: 10%)
        if mineru_result.structured_data and not easyocr_result.structured_data:
            score_mineru += 1
        elif easyocr_result.structured_data and not mineru_result.structured_data:
            score_easyocr += 1

        if score_mineru > score_easyocr:
            return "mineru"
        elif score_easyocr > score_mineru:
            return "easyocr"
        else:
            return "tie"

    def _log_comparison(self, document_result: Dict[str, Any]):
        """Log comparison results for a single document"""
        logger.info("=" * 60)
        logger.info(f"File: {document_result['file_name']}")
        logger.info(f"Size: {document_result['file_size_mb']:.2f} MB")

        easyocr = document_result['easyocr']
        mineru = document_result['mineru']
        comparison = document_result['comparison']

        logger.info(
            f"EasyOCR: {easyocr['char_count']} chars, {easyocr['confidence']:.3f} conf, {easyocr['processing_time']:.2f}s")
        logger.info(
            f"MinerU:  {mineru['char_count']} chars, {mineru['confidence']:.3f} conf, {mineru['processing_time']:.2f}s")

        if comparison.get('both_successful', False):
            logger.info(f"Char improvement: {comparison['char_count_improvement_pct']:.1f}%")
            logger.info(f"Time ratio: {comparison['time_ratio']:.2f}x")
            logger.info(f"Winner: {comparison['winner']}")
        else:
            logger.info(f"EasyOCR Success: {comparison['easyocr_successful']}")
            logger.info(f"MinerU Success: {comparison['mineru_successful']}")

        logger.info("=" * 60)

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_documents': len(results),
            'successful_comparisons': 0,
            'easyocr_wins': 0,
            'mineru_wins': 0,
            'ties': 0,
            'avg_char_improvement_pct': 0.0,
            'avg_time_ratio': 0.0,
            'total_processing_time': {'easyocr': 0.0, 'mineru': 0.0}
        }

        successful_comparisons = []

        for result in results:
            if result['comparison'].get('both_successful', False):
                summary['successful_comparisons'] += 1
                successful_comparisons.append(result)

                winner = result['comparison']['winner']
                if winner == 'easyocr':
                    summary['easyocr_wins'] += 1
                elif winner == 'mineru':
                    summary['mineru_wins'] += 1
                else:
                    summary['ties'] += 1

            # Accumulate processing times
            summary['total_processing_time']['easyocr'] += result['easyocr']['processing_time']
            summary['total_processing_time']['mineru'] += result['mineru']['processing_time']

        # Calculate averages
        if successful_comparisons:
            summary['avg_char_improvement_pct'] = sum(
                r['comparison']['char_count_improvement_pct'] for r in successful_comparisons
            ) / len(successful_comparisons)

            summary['avg_time_ratio'] = sum(
                r['comparison']['time_ratio'] for r in successful_comparisons
            ) / len(successful_comparisons)

        return summary

    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for identification"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_results(self, results: Dict[str, Any]):
        """Save comparison results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"ocr_comparison_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {filename}")

        # Also save a summary report
        self._generate_report(results, timestamp)

    def _generate_report(self, results: Dict[str, Any], timestamp: str):
        """Generate a human-readable report"""
        report_file = self.output_dir / f"ocr_comparison_report_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("OCR PIPELINE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"Total Documents: {results['total_documents']}\n\n")

            summary = results['summary']
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Successful Comparisons: {summary['successful_comparisons']}\n")
            f.write(f"EasyOCR Wins: {summary['easyocr_wins']}\n")
            f.write(f"MinerU Wins: {summary['mineru_wins']}\n")
            f.write(f"Ties: {summary['ties']}\n")
            f.write(f"Average Character Improvement: {summary['avg_char_improvement_pct']:.1f}%\n")
            f.write(f"Average Time Ratio (MinerU/EasyOCR): {summary['avg_time_ratio']:.2f}x\n")
            f.write(f"Total EasyOCR Time: {summary['total_processing_time']['easyocr']:.2f}s\n")
            f.write(f"Total MinerU Time: {summary['total_processing_time']['mineru']:.2f}s\n\n")

            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results['results']:
                f.write(f"\nFile: {result['file_name']}\n")
                f.write(f"Size: {result['file_size_mb']:.2f} MB\n")

                if result['comparison'].get('both_successful', False):
                    f.write(
                        f"EasyOCR: {result['easyocr']['char_count']} chars, {result['easyocr']['confidence']:.3f} conf\n")
                    f.write(
                        f"MinerU:  {result['mineru']['char_count']} chars, {result['mineru']['confidence']:.3f} conf\n")
                    f.write(f"Winner: {result['comparison']['winner']}\n")
                    f.write(f"Improvement: {result['comparison']['char_count_improvement_pct']:.1f}%\n")
                else:
                    f.write(f"EasyOCR Success: {result['easyocr']['success']}\n")
                    f.write(f"MinerU Success: {result['mineru']['success']}\n")
                    if result['mineru']['error_message']:
                        f.write(f"MinerU Error: {result['mineru']['error_message']}\n")

        logger.info(f"Report saved to: {report_file}")


def check_magic_pdf_config():
    """Check if magic-pdf.json config exists and help create it if not"""
    import os

    # Common locations for magic-pdf.json
    possible_locations = [
        os.path.expanduser("~/magic-pdf.json"),
        os.path.join(os.getcwd(), "magic-pdf.json"),
        "magic-pdf.json"
    ]

    config_found = False
    for location in possible_locations:
        if os.path.exists(location):
            logger.info(f"Found magic-pdf.json at: {location}")
            config_found = True
            break

    if not config_found:
        logger.warning("magic-pdf.json configuration file not found!")
        logger.info("MinerU might need a configuration file to work properly.")
        logger.info("You can:")
        logger.info(
            "1. Download the template: wget https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json")
        logger.info("2. Copy it to your home directory: cp magic-pdf.template.json ~/magic-pdf.json")
        logger.info("3. Or let the script try to work without it (some features may be limited)")


def main():
    """Main function to run the comparison"""
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Compare MinerU vs EasyOCR performance")
    parser.add_argument("pdf_paths", nargs="+", help="Paths to PDF files to test (supports wildcards)")
    parser.add_argument("--output-dir", default="ocr_comparison_results", help="Output directory for results")
    parser.add_argument("--check-config", action="store_true", help="Check for magic-pdf.json configuration")

    args = parser.parse_args()

    # Check configuration if requested
    if args.check_config:
        check_magic_pdf_config()

    # Expand wildcards and validate PDF files
    valid_paths = []
    for pattern in args.pdf_paths:
        if '*' in pattern:
            # Handle wildcards
            expanded_paths = glob.glob(pattern)
            for path in expanded_paths:
                if os.path.exists(path) and path.lower().endswith('.pdf'):
                    valid_paths.append(path)
        else:
            # Handle direct paths
            if os.path.exists(pattern) and pattern.lower().endswith('.pdf'):
                valid_paths.append(pattern)
            else:
                logger.warning(f"Skipping invalid file: {pattern}")

    if not valid_paths:
        logger.error("No valid PDF files found!")
        logger.info("Tip: Use --check-config to verify your MinerU setup")
        return

    logger.info(f"Found {len(valid_paths)} PDF files to process")

    # Check for config issues
    check_magic_pdf_config()

    # Run comparison
    comparator = OCRComparator(args.output_dir)
    results = comparator.compare_documents(valid_paths)

    # Print final summary
    summary = results['summary']
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"Documents Processed: {summary['total_documents']}")
    print(f"Successful Comparisons: {summary['successful_comparisons']}")

    if summary['successful_comparisons'] > 0:
        print(f"Winners:")
        print(f"   EasyOCR: {summary['easyocr_wins']}")
        print(f"   MinerU:  {summary['mineru_wins']}")
        print(f"   Ties:    {summary['ties']}")
        print(f"Avg Character Improvement: {summary['avg_char_improvement_pct']:.1f}%")
        print(f"Avg Time Ratio: {summary['avg_time_ratio']:.2f}x")
    else:
        print("No successful comparisons - check the logs for errors")
        print("Common issues:")
        print("- Missing magic-pdf.json configuration file")
        print("- MinerU installation incomplete")
        print("- Missing model files")

    print("=" * 60)


if __name__ == "__main__":
    main()