# src/utils/ocr_storage.py

import os
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OCRStorageManager:
    """Manages storage of OCR outputs to local filesystem."""

    def __init__(self, base_storage_dir: str = "ocr_outputs"):
        """Initialize storage manager with base directory."""
        self.base_dir = Path(base_storage_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.text_dir = self.base_dir / "extracted_text"
        self.json_dir = self.base_dir / "structured_data"
        self.csv_dir = self.base_dir / "csv_exports"
        self.raw_files_dir = self.base_dir / "original_files"

        for directory in [self.text_dir, self.json_dir, self.csv_dir, self.raw_files_dir]:
            directory.mkdir(exist_ok=True)

        logger.info(f"OCR storage initialized at: {self.base_dir.absolute()}")

    def save_ocr_output(self,
                        uploaded_file: Any,
                        ocr_text: str,
                        structured_data: Optional[Dict] = None) -> Dict[str, str]:
        """Save OCR output and return file paths."""

        # Generate unique filename based on original file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(uploaded_file.name).stem
        file_hash = self._generate_file_hash(uploaded_file.getvalue())

        base_filename = f"{original_name}_{timestamp}_{file_hash[:8]}"

        saved_files = {}

        try:
            # 1. Save original file
            original_path = self.raw_files_dir / f"{base_filename}{Path(uploaded_file.name).suffix}"
            with open(original_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            saved_files['original_file'] = str(original_path)

            # 2. Save extracted text
            text_path = self.text_dir / f"{base_filename}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            saved_files['extracted_text'] = str(text_path)

            # 3. Save metadata as JSON
            metadata = {
                'timestamp': timestamp,
                'original_filename': uploaded_file.name,
                'file_size_bytes': len(uploaded_file.getvalue()),
                'file_hash': file_hash,
                'file_type': uploaded_file.type,
                'text_length': len(ocr_text),
                'extraction_timestamp': datetime.now().isoformat(),
                'structured_data': structured_data or {}
            }

            json_path = self.json_dir / f"{base_filename}_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            saved_files['metadata'] = str(json_path)

            # 4. Save structured data as separate JSON if provided
            if structured_data:
                structured_path = self.json_dir / f"{base_filename}_structured.json"
                with open(structured_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)
                saved_files['structured_data'] = str(structured_path)

            logger.info(f"OCR output saved for {uploaded_file.name}: {len(saved_files)} files")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to save OCR output for {uploaded_file.name}: {e}")
            raise

    def save_batch_summary(self, batch_results: list) -> str:
        """Save summary of batch processing to CSV."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.csv_dir / f"batch_summary_{timestamp}.csv"

        # Prepare CSV data
        csv_data = []
        for result in batch_results:
            csv_data.append({
                'timestamp': result.get('timestamp', ''),
                'filename': result.get('original_filename', ''),
                'file_type': result.get('file_type', ''),
                'file_size_kb': round(result.get('file_size_bytes', 0) / 1024, 2),
                'text_length': result.get('text_length', 0),
                'success': 'Yes' if result.get('text_length', 0) > 0 else 'No',
                'text_file_path': result.get('saved_files', {}).get('extracted_text', ''),
                'json_file_path': result.get('saved_files', {}).get('metadata', '')
            })

        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                writer = csv.DictWriter(csvfile, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

        logger.info(f"Batch summary saved to: {csv_path}")
        return str(csv_path)

    def load_ocr_text(self, filename_pattern: str) -> Optional[str]:
        """Load previously saved OCR text by filename pattern."""

        text_files = list(self.text_dir.glob(f"*{filename_pattern}*.txt"))

        if not text_files:
            logger.warning(f"No OCR text files found matching: {filename_pattern}")
            return None

        # Return most recent if multiple matches
        latest_file = max(text_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read()

    def list_saved_files(self, limit: int = 20) -> list:
        """List recently saved OCR files."""

        files_info = []

        for text_file in sorted(self.text_dir.glob("*.txt"),
                                key=lambda p: p.stat().st_mtime,
                                reverse=True)[:limit]:

            # Get corresponding metadata if exists
            metadata_file = self.json_dir / f"{text_file.stem}_metadata.json"
            metadata = {}

            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            files_info.append({
                'filename': text_file.name,
                'text_file_path': str(text_file),
                'size_kb': round(text_file.stat().st_size / 1024, 2),
                'modified_time': datetime.fromtimestamp(text_file.stat().st_mtime).isoformat(),
                'original_filename': metadata.get('original_filename', 'Unknown'),
                'text_length': metadata.get('text_length', 0)
            })

        return files_info

    def _generate_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content."""
        return hashlib.sha256(file_content).hexdigest()

    def export_to_excel(self, output_path: Optional[str] = None) -> str:
        """Export all OCR data to Excel file."""

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.base_dir / f"ocr_export_{timestamp}.xlsx")

        try:
            import pandas as pd

            # Collect all metadata
            all_data = []
            for json_file in self.json_dir.glob("*_metadata.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                    # Add text content
                    text_file = self.text_dir / f"{json_file.stem.replace('_metadata', '')}.txt"
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as tf:
                            metadata['extracted_text'] = tf.read()

                    all_data.append(metadata)

            # Create DataFrame and save to Excel
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_excel(output_path, index=False)
                logger.info(f"Excel export saved to: {output_path}")
            else:
                logger.warning("No data found for Excel export")

            return output_path

        except ImportError:
            logger.error("pandas required for Excel export. Install with: pip install pandas openpyxl")
            raise
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise


# Convenience functions for easy integration
def create_storage_manager(storage_dir: str = "ocr_outputs") -> OCRStorageManager:
    """Factory function to create storage manager."""
    return OCRStorageManager(storage_dir)


def save_ocr_result(uploaded_file: Any,
                    ocr_text: str,
                    storage_manager: OCRStorageManager,
                    structured_data: Optional[Dict] = None) -> Dict[str, str]:
    """Quick function to save OCR result."""
    return storage_manager.save_ocr_output(uploaded_file, ocr_text, structured_data)