# integration_setup_windows_fix.py - WINDOWS COMPATIBLE VERSION

"""
Quick Windows-compatible fix for the integration setup.
This handles Unicode encoding issues on Windows systems.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger(__name__)


def quick_integration_check():
    """Quick integration check without Unicode issues."""
    print("Enhanced Document Ingestion System - Quick Integration Check")
    print("=" * 70)

    # Check required files
    required_files = {
        'Main App': 'GraphRAG_Document_AI_Platform.py',
        'Config': 'config.toml',
        'Enhanced OCR': 'enhanced_ocr_pipeline.py',
        'Audit DB': 'src/utils/audit_db_manager.py',
        'Processing': 'src/utils/processing_pipeline.py',
        'OCR Storage': 'src/utils/ocr_storage.py',
        'Text Utils': 'src/knowledge_graph/text_utils.py',
        'LLM': 'src/knowledge_graph/llm.py',
        'Graph QA': 'graph_rag_qa.py'
    }

    print("\n[1] FILE VALIDATION:")
    print("-" * 30)
    missing_files = []
    for name, filepath in required_files.items():
        if Path(filepath).exists():
            print(f"FOUND     {name}")
        else:
            print(f"MISSING   {name}")
            missing_files.append(filepath)

    # Check dependencies
    print("\n[2] DEPENDENCY CHECK:")
    print("-" * 30)
    required_deps = [
        'streamlit', 'pandas', 'neo4j', 'chromadb',
        'sentence_transformers', 'mistralai', 'anthropic', 'openai'
    ]

    missing_deps = []
    for dep in required_deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"INSTALLED {dep}")
        except ImportError:
            print(f"MISSING   {dep}")
            missing_deps.append(dep)

    # Test system components
    print("\n[3] SYSTEM TESTS:")
    print("-" * 30)

    # Test config loading
    try:
        from GraphRAG_Document_AI_Platform import load_config
        config = load_config()
        if config and config.get('_CONFIG_VALID'):
            print("PASSED    Configuration Loading")
        else:
            print("FAILED    Configuration Loading")
    except Exception as e:
        print(f"FAILED    Configuration Loading: {e}")

    # Test OCR pipeline
    try:
        from GraphRAG_Document_AI_Platform import get_enhanced_ocr_pipeline, load_config
        config = load_config()
        ocr_pipeline = get_enhanced_ocr_pipeline(config)
        if ocr_pipeline:
            providers = ocr_pipeline.get_available_providers()
            print(f"PASSED    OCR Pipeline ({len(providers)} providers)")
        else:
            print("FAILED    OCR Pipeline")
    except Exception as e:
        print(f"FAILED    OCR Pipeline: {e}")

    # Test Neo4j connection
    try:
        from GraphRAG_Document_AI_Platform import init_neo4j_exporter, load_config
        config = load_config()
        neo4j_exporter = init_neo4j_exporter(
            config.get('NEO4J_URI'),
            config.get('NEO4J_USER'),
            config.get('NEO4J_PASSWORD')
        )
        if neo4j_exporter:
            print("PASSED    Neo4j Connection")
        else:
            print("FAILED    Neo4j Connection")
    except Exception as e:
        print(f"FAILED    Neo4j Connection: {e}")

    # Summary
    print("\n[4] INTEGRATION SUMMARY:")
    print("-" * 30)

    if not missing_files and not missing_deps:
        print("SUCCESS: System appears ready for enhanced features!")
        print("         You can proceed with installing the enhanced components.")
        return True
    else:
        print("ISSUES FOUND:")
        if missing_files:
            print(f"  - {len(missing_files)} required files missing")
        if missing_deps:
            print(f"  - {len(missing_deps)} required dependencies missing")
            print(f"    Install with: pip install {' '.join(missing_deps)}")
        return False


def install_missing_dependency():
    """Install the missing google-generativeai dependency."""
    print("\nInstalling missing dependency...")
    print("-" * 40)

    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'google-generativeai'],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("SUCCESS: google-generativeai installed successfully!")
            return True
        else:
            print(f"FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to install dependency: {e}")
        return False


def create_windows_startup_script():
    """Create Windows-compatible startup script."""
    print("\nCreating Windows startup script...")

    batch_content = '''@echo off
echo Starting Enhanced Document Ingestion System...

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\\Scripts\\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start Streamlit application
echo Launching Streamlit application...
streamlit run GraphRAG_Document_AI_Platform.py --server.fileWatcherType none

echo Enhanced Document Ingestion System stopped.
pause
'''

    try:
        script_path = Path("start_enhanced_system.bat")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)

        print(f"SUCCESS: Created {script_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create startup script: {e}")
        return False


def create_requirements_file():
    """Create requirements.txt file."""
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "neo4j>=5.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "mistralai>=0.0.8",
        "google-generativeai>=0.3.0",
        "anthropic>=0.7.0",
        "openai>=1.0.0",
        "langchain-text-splitters>=0.0.1",
        "spacy>=3.6.0",
        "tomli>=2.0.0",
        "plotly>=5.0.0",
        "openpyxl>=3.1.0",
        "fuzzywuzzy>=0.18.0",
        "python-levenshtein>=0.20.0",
        "Pillow>=9.0.0",
        "PyMuPDF>=1.23.0",
        "pathlib>=1.0.0",
        "numpy>=1.24.0",
        "requests>=2.28.0"
    ]

    try:
        requirements_path = Path("requirements.txt")
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements))

        print(f"SUCCESS: Created requirements.txt with {len(requirements)} dependencies")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create requirements.txt: {e}")
        return False


def backup_current_files():
    """Backup current files before replacing."""
    print("\nBacking up current files...")

    backup_dir = Path("backup_original_files")
    backup_dir.mkdir(exist_ok=True)

    files_to_backup = [
        ("pages/2_Document_Ingestion.py", "original_document_ingestion.py")
    ]

    for source, backup_name in files_to_backup:
        source_path = Path(source)
        if source_path.exists():
            backup_path = backup_dir / backup_name
            try:
                shutil.copy2(source_path, backup_path)
                print(f"BACKED UP: {source} -> {backup_name}")
            except Exception as e:
                print(f"WARNING: Could not backup {source}: {e}")
        else:
            print(f"SKIPPED: {source} (file not found)")


def show_manual_installation_steps():
    """Show manual installation steps."""
    print("\n" + "=" * 70)
    print("MANUAL INSTALLATION STEPS")
    print("=" * 70)

    print("""
Based on your system status, here are the steps to complete the installation:

STEP 1: Install missing dependency
  pip install google-generativeai

STEP 2: Save the enhanced components
  You need to save these 3 files to your project:

  A) pages/2_Document_Ingestion.py (Enhanced frontend)
  B) src/utils/realtime_progress.py (Progress tracking)  
  C) src/utils/enhanced_processing_pipeline.py (Enhanced backend)

STEP 3: Test the system
  streamlit run GraphRAG_Document_AI_Platform.py

STEP 4: Navigate to Document Ingestion page
  - Upload some documents
  - Watch the real-time progress monitoring
  - Enjoy the enhanced features!

TROUBLESHOOTING:
- If you get Unicode errors, ensure you're using UTF-8 encoding
- If imports fail, check that all files are saved in correct locations
- If Neo4j connection fails, ensure Neo4j is running on localhost:7687

SUCCESS INDICATORS:
- Real-time progress bars during OCR processing
- Live job monitoring with auto-refresh
- Enhanced job history with detailed analytics
- OCR performance metrics display
""")


def main():
    """Main function with error handling."""
    try:
        print("Enhanced Document Ingestion System - Windows Integration")
        print("=" * 70)

        # Quick system check
        system_ready = quick_integration_check()

        if system_ready:
            print("\n" + "=" * 70)
            print("GREAT NEWS: Your system looks ready!")
            print("=" * 70)

            # Create utility files
            create_requirements_file()
            create_windows_startup_script()
            backup_current_files()

            print("""
NEXT STEPS:
1. Save the 3 enhanced component files (provided separately)
2. Run: start_enhanced_system.bat
3. Navigate to Document Ingestion page
4. Upload documents and enjoy real-time monitoring!
""")
        else:
            # Try to fix the missing dependency
            print("\nAttempting to fix missing dependency...")
            if install_missing_dependency():
                print("Dependency installed! Please run this script again.")
            else:
                show_manual_installation_steps()

    except Exception as e:
        print(f"\nERROR during integration: {e}")
        print("Please try the manual installation steps above.")


if __name__ == "__main__":
    main()