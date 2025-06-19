# src/utils/ocr_storage.py - FINAL VERSION WITH SINGLETON FIX

import os
import json
import csv
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import shutil

# FIXED: Reduced logging verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# FIXED: Global singleton instance and lock to prevent repeated initialization
_storage_manager_instance = None
_storage_manager_lock = threading.Lock()
_initialization_count = 0


class OCRStorageManager:
    """
    FIXED: Manages storage of OCR outputs to local filesystem with singleton pattern.
    This prevents the repeated initialization that was causing log spam.
    """

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

        # FIXED: Only log initialization once per instance
        if not hasattr(self, '_initialized'):
            logger.info(f"OCR storage initialized at: {self.base_dir.absolute()}")
            self._initialized = True

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
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                f.write(uploaded_file.getvalue())
                uploaded_file.seek(0)  # Reset for other uses
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
                'file_type': getattr(uploaded_file, 'type', 'unknown'),
                'text_length': len(ocr_text),
                'extraction_timestamp': datetime.now().isoformat(),
                'structured_data': structured_data or {}
            }

            json_path = self.json_dir / f"{base_filename}_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            saved_files['metadata'] = str(json_path)

            # 4. Save structured data as separate JSON if provided
            if structured_data:
                structured_path = self.json_dir / f"{base_filename}_structured.json"
                with open(structured_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False, default=str)
                saved_files['structured_data'] = str(structured_path)

            # 5. Save summary file for quick reference
            summary_path = self.text_dir / f"{base_filename}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Summary for: {uploaded_file.name}\n")
                f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text Length: {len(ocr_text)} characters\n")
                f.write(f"File Hash: {file_hash}\n")
                f.write(f"Method: {structured_data.get('ocr_method', 'unknown') if structured_data else 'unknown'}\n")
                f.write(
                    f"Confidence: {structured_data.get('confidence', 'unknown') if structured_data else 'unknown'}\n")
                f.write("\n--- Extracted Text Preview (first 500 chars) ---\n")
                f.write(ocr_text[:500])
                if len(ocr_text) > 500:
                    f.write("\n... (truncated)")
            saved_files['summary'] = str(summary_path)

            # REDUCED: Less verbose logging - only log success summary
            logger.info(f"OCR output saved for {uploaded_file.name}: {len(saved_files)} files")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to save OCR output for {uploaded_file.name}: {e}")
            raise

    def save_batch_summary(self, batch_results: List[Dict]) -> str:
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
        # Filter out summary files
        text_files = [f for f in text_files if not f.name.endswith('_summary.txt')]

        if not text_files:
            logger.warning(f"No OCR text files found matching: {filename_pattern}")
            return None

        # Return most recent if multiple matches
        latest_file = max(text_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load OCR text from {latest_file}: {e}")
            return None

    def list_saved_files(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recently saved OCR files with enhanced information.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of dictionaries with file information
        """
        files_info = []

        # Get all text files (excluding summaries)
        text_files = [f for f in self.text_dir.glob("*.txt") if not f.name.endswith('_summary.txt')]

        for text_file in sorted(text_files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:

            # Get corresponding metadata if exists
            metadata_file = self.json_dir / f"{text_file.stem}_metadata.json"
            metadata = {}

            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read metadata for {text_file.name}: {e}")

            # Calculate file size
            try:
                size_kb = round(text_file.stat().st_size / 1024, 2)
                modified_time = datetime.fromtimestamp(text_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logger.warning(f"Could not get file stats for {text_file.name}: {e}")
                size_kb = 0
                modified_time = 'Unknown'

            files_info.append({
                'filename': text_file.name,
                'text_file_path': str(text_file),
                'size_kb': size_kb,
                'modified_time': modified_time,
                'original_filename': metadata.get('original_filename', 'Unknown'),
                'text_length': metadata.get('text_length', 0),
                'file_hash': metadata.get('file_hash', ''),
                'extraction_timestamp': metadata.get('extraction_timestamp', ''),
                'ocr_method': metadata.get('structured_data', {}).get('ocr_method', 'unknown'),
                'confidence': metadata.get('structured_data', {}).get('confidence', 0)
            })

        return files_info

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            stats = {
                'base_directory': str(self.base_dir.absolute()),
                'total_files': 0,
                'total_size_mb': 0,
                'file_types': {},
                'directories': {}
            }

            # Count files in each directory
            for dir_name, dir_path in [
                ('text_files', self.text_dir),
                ('json_files', self.json_dir),
                ('csv_files', self.csv_dir),
                ('original_files', self.raw_files_dir)
            ]:
                dir_stats = {'count': 0, 'size_mb': 0}

                if dir_path.exists():
                    for item in dir_path.rglob("*"):
                        if item.is_file():
                            stats['total_files'] += 1
                            dir_stats['count'] += 1

                            file_size = item.stat().st_size
                            size_mb = file_size / (1024 * 1024)
                            stats['total_size_mb'] += size_mb
                            dir_stats['size_mb'] += size_mb

                            # Count by file extension
                            ext = item.suffix.lower()
                            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1

                stats['directories'][dir_name] = dir_stats

            # Round total size
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            for dir_stats in stats['directories'].values():
                dir_stats['size_mb'] = round(dir_stats['size_mb'], 2)

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'error': str(e)}

    def cleanup_old_files(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up OCR files older than specified days.

        Args:
            days_old: Files older than this many days will be deleted

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)

            deleted_files = 0
            errors = 0
            total_size_freed = 0

            for directory in [self.text_dir, self.json_dir, self.raw_files_dir]:
                if directory.exists():
                    for file_path in directory.iterdir():
                        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                            try:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                deleted_files += 1
                                total_size_freed += file_size
                                logger.debug(f"Deleted old file: {file_path.name}")
                            except Exception as e:
                                logger.error(f"Failed to delete {file_path.name}: {e}")
                                errors += 1

            return {
                'deleted_files': deleted_files,
                'errors': errors,
                'size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
                'days_threshold': days_old
            }

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return {'error': str(e)}

    def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file information by hash.

        Args:
            file_hash: Hash of the file (full or partial)

        Returns:
            Dictionary with file information or None if not found
        """
        try:
            for metadata_file in self.json_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    stored_hash = metadata.get('file_hash', '')
                    if stored_hash and (stored_hash == file_hash or stored_hash.startswith(file_hash)):
                        # Add file paths
                        base_name = metadata_file.stem.replace('_metadata', '')
                        metadata['text_file'] = str(self.text_dir / f"{base_name}.txt")
                        metadata['metadata_file'] = str(metadata_file)
                        metadata['original_file'] = str(
                            self.raw_files_dir / f"{base_name}{Path(metadata['original_filename']).suffix}")
                        return metadata

                except Exception as e:
                    logger.warning(f"Could not read metadata file {metadata_file.name}: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"Failed to search for file by hash {file_hash}: {e}")
            return None

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
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Add text content
                    text_file = self.text_dir / f"{json_file.stem.replace('_metadata', '')}.txt"
                    if text_file.exists():
                        try:
                            with open(text_file, 'r', encoding='utf-8') as tf:
                                text_content = tf.read()
                                metadata['extracted_text'] = text_content[:1000] + "..." if len(
                                    text_content) > 1000 else text_content
                        except Exception as e:
                            logger.warning(f"Could not read text file {text_file.name}: {e}")
                            metadata['extracted_text'] = "Error reading text"

                    all_data.append(metadata)

                except Exception as e:
                    logger.warning(f"Could not process {json_file.name}: {e}")
                    continue

            # Create DataFrame and save to Excel
            if all_data:
                df = pd.DataFrame(all_data)
                # Flatten structured_data if it exists
                if 'structured_data' in df.columns:
                    structured_df = pd.json_normalize(df['structured_data'])
                    structured_df.columns = [f"structured_{col}" for col in structured_df.columns]
                    df = pd.concat([df.drop('structured_data', axis=1), structured_df], axis=1)

                df.to_excel(output_path, index=False, engine='openpyxl')
                logger.info(f"Excel export saved to: {output_path}")
            else:
                logger.warning("No data found for Excel export")

            return output_path

        except ImportError:
            logger.error("pandas and openpyxl required for Excel export. Install with: pip install pandas openpyxl")
            raise
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise

    def _generate_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content."""
        return hashlib.sha256(file_content).hexdigest()

    def export_metadata_json(self, output_path: Optional[str] = None) -> str:
        """Export all metadata to a single JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.base_dir / f"metadata_export_{timestamp}.json")

        try:
            all_metadata = []
            for metadata_file in self.json_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    metadata['storage_file'] = metadata_file.name
                    all_metadata.append(metadata)
                except Exception as e:
                    logger.warning(f"Could not read metadata file {metadata_file.name}: {e}")

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_files': len(all_metadata),
                'storage_base_dir': str(self.base_dir.absolute()),
                'files': all_metadata
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Metadata exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Metadata export failed: {e}")
            raise


# FIXED: Singleton factory function to prevent repeated initialization
def create_storage_manager(storage_dir: str = "ocr_outputs") -> OCRStorageManager:
    """
    FIXED: Factory function to create or get singleton storage manager.
    This prevents multiple initializations that were causing log spam.
    """
    global _storage_manager_instance, _initialization_count

    with _storage_manager_lock:
        _initialization_count += 1

        if _storage_manager_instance is None:
            _storage_manager_instance = OCRStorageManager(storage_dir)
            logger.info("OCR storage manager initialized (singleton)")
        else:
            # FIXED: Only log debug message for subsequent calls
            logger.debug(f"OCR storage manager already initialized (call #{_initialization_count})")

        return _storage_manager_instance


def save_ocr_result(uploaded_file: Any,
                    ocr_text: str,
                    storage_manager: OCRStorageManager,
                    structured_data: Optional[Dict] = None) -> Dict[str, str]:
    """Quick function to save OCR result."""
    return storage_manager.save_ocr_output(uploaded_file, ocr_text, structured_data)


# FIXED: Additional utility functions for singleton management
def get_storage_manager() -> Optional[OCRStorageManager]:
    """Get the current singleton storage manager instance if it exists."""
    return _storage_manager_instance


def is_storage_initialized() -> bool:
    """Check if the storage manager singleton has been initialized."""
    return _storage_manager_instance is not None


def reset_storage_manager():
    """Reset the singleton instance (mainly for testing purposes)."""
    global _storage_manager_instance, _initialization_count

    with _storage_manager_lock:
        _storage_manager_instance = None
        _initialization_count = 0
        logger.debug("OCR storage manager singleton reset")


def get_initialization_count() -> int:
    """Get the number of times create_storage_manager has been called."""
    return _initialization_count


# Convenience functions for common operations without requiring manager instance
def save_ocr_result_singleton(uploaded_file: Any,
                              ocr_text: str,
                              structured_data: Optional[Dict] = None) -> Dict[str, str]:
    """
    Convenience function to save OCR result using the singleton manager.
    Creates manager if it doesn't exist.
    """
    manager = create_storage_manager()
    return manager.save_ocr_output(uploaded_file, ocr_text, structured_data)


def list_recent_ocr_files(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Convenience function to list recent OCR files.
    Returns empty list if no manager exists.
    """
    manager = get_storage_manager()
    if manager:
        return manager.list_saved_files(limit)
    return []


def get_ocr_storage_stats() -> Dict[str, Any]:
    """
    Convenience function to get storage statistics.
    Returns empty dict if no manager exists.
    """
    manager = get_storage_manager()
    if manager:
        return manager.get_storage_stats()
    return {}


# Context manager for temporary storage
class TemporaryOCRStorage:
    """Context manager for temporary OCR storage that cleans up automatically."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("temp_ocr_storage")
        self.manager = None
        self.should_cleanup = False

    def __enter__(self) -> OCRStorageManager:
        self.manager = OCRStorageManager(str(self.base_dir))
        self.should_cleanup = True
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.should_cleanup and self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
                logger.debug(f"Cleaned up temporary OCR storage: {self.base_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary storage {self.base_dir}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example usage of the OCR storage manager
    import io

    print("Testing OCR Storage Manager...")

    # Test singleton behavior
    manager1 = create_storage_manager("test_ocr_storage")
    manager2 = create_storage_manager("test_ocr_storage")

    print(f"Singleton test: {manager1 is manager2}")  # Should be True
    print(f"Initialization count: {get_initialization_count()}")

    # Test saving a mock file
    mock_file_content = b"This is a test document content for OCR processing."
    mock_file = io.BytesIO(mock_file_content)
    mock_file.name = "test_document.txt"
    mock_file.type = "text/plain"

    test_text = "This is extracted text from the test document."
    test_metadata = {
        'ocr_method': 'test_method',
        'confidence': 0.95,
        'processing_time': 1.23
    }

    saved_files = manager1.save_ocr_output(mock_file, test_text, test_metadata)
    print(f"Saved files: {list(saved_files.keys())}")

    # Test listing files
    recent_files = manager1.list_saved_files(limit=5)
    print(f"Recent files count: {len(recent_files)}")

    # Test statistics
    stats = manager1.get_storage_stats()
    print(f"Storage stats: {stats}")

    # Test cleanup
    print("Testing cleanup...")
    reset_storage_manager()
    print(f"After reset - is initialized: {is_storage_initialized()}")

    print("OCR Storage Manager test completed.")