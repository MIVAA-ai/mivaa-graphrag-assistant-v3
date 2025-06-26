#!/usr/bin/env python3
"""
Simple Import Scanner - Extract imports from Python files and generate requirements.txt
Usage: python scan_imports.py [directory]
"""

import os
import re
import sys
from pathlib import Path


def extract_imports_from_file(file_path):
    """Extract import statements from a Python file using regex"""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find import statements
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        ]

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue

            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    imports.add(module)

    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")

    return imports


def scan_directory(directory):
    """Scan all Python files in directory and subdirectories"""
    all_imports = set()
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if
                   d not in ['__pycache__', '.git', '.venv', 'venv', 'node_modules', 'dist', 'build']]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                python_files.append(file_path)
                imports = extract_imports_from_file(file_path)
                all_imports.update(imports)

    return all_imports, python_files


def filter_external_packages(imports):
    """Filter out standard library modules and get external packages"""

    # Common standard library modules
    stdlib_modules = {
        'os', 'sys', 'json', 'time', 'datetime', 'logging', 'threading', 'pathlib',
        'typing', 'collections', 'itertools', 'functools', 'operator', 'copy',
        'pickle', 'hashlib', 'base64', 'tempfile', 'shutil', 'glob', 're', 'math',
        'random', 'string', 'io', 'traceback', 'warnings', 'weakref', 'gc',
        'subprocess', 'multiprocessing', 'asyncio', 'concurrent', 'queue',
        'socket', 'ssl', 'urllib', 'http', 'email', 'html', 'xml', 'csv',
        'configparser', 'argparse', 'getopt', 'platform', 'ctypes', 'struct',
        'array', 'bisect', 'heapq', 'enum', 'dataclasses', 'abc', 'contextlib',
        'unittest', 'doctest', 'pdb', 'profile', 'cProfile', 'timeit'
    }

    # Package name mappings (import name -> package name)
    package_mappings = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'sentence_transformers': 'sentence-transformers',
        'nest_asyncio': 'nest-asyncio',
        'google': 'google-generativeai',  # Simplified mapping
        'langchain_text_splitters': 'langchain-text-splitters',
        'llama_index': 'llama-index',
        'spacy': 'spacy',
        'streamlit': 'streamlit',
        'requests': 'requests',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'scipy': 'scipy',
        'torch': 'torch',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'transformers': 'transformers',
        'datasets': 'datasets'
    }

    external_packages = set()
    local_modules = set()

    for module in imports:
        if module.startswith('src') or module.startswith('.'):
            local_modules.add(module)
        elif module not in stdlib_modules and not module.startswith('_'):
            # Map to package name
            package_name = package_mappings.get(module, module)
            external_packages.add(package_name)

    return external_packages, local_modules


def generate_requirements(packages, output_file='requirements_scanned.txt'):
    """Generate requirements.txt file"""

    # Categorize packages
    categories = {
        'Core Processing': [],
        'ML/AI Libraries': [],
        'Web/API': [],
        'Data Processing': [],
        'Other Dependencies': []
    }

    ml_keywords = ['torch', 'tensorflow', 'sklearn', 'transformers', 'sentence', 'spacy', 'nltk', 'keras']
    web_keywords = ['streamlit', 'flask', 'django', 'fastapi', 'requests', 'urllib3']
    data_keywords = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy']
    core_keywords = ['pillow', 'opencv', 'pdf2image', 'easyocr', 'chromadb', 'neo4j']

    for package in sorted(packages):
        package_lower = package.lower()

        if any(keyword in package_lower for keyword in core_keywords):
            categories['Core Processing'].append(package)
        elif any(keyword in package_lower for keyword in ml_keywords):
            categories['ML/AI Libraries'].append(package)
        elif any(keyword in package_lower for keyword in web_keywords):
            categories['Web/API'].append(package)
        elif any(keyword in package_lower for keyword in data_keywords):
            categories['Data Processing'].append(package)
        else:
            categories['Other Dependencies'].append(package)

    with open(output_file, 'w') as f:
        f.write("# Auto-generated requirements.txt\n")
        f.write(f"# Scanned from Python files\n\n")

        for category, packages_list in categories.items():
            if packages_list:
                f.write(f"# {category}\n")
                for package in packages_list:
                    f.write(f"{package}\n")
                f.write("\n")


def main():
    """Main function"""
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    directory = Path(directory).resolve()

    print(f"Scanning Python files in: {directory}")

    # Scan for imports
    all_imports, python_files = scan_directory(directory)

    print(f"Found {len(python_files)} Python files")
    print(f"Total unique imports: {len(all_imports)}")

    # Filter external packages
    external_packages, local_modules = filter_external_packages(all_imports)

    print(f"External packages: {len(external_packages)}")
    print(f"Local modules: {len(local_modules)}")

    print("\nExternal packages found:")
    for package in sorted(external_packages):
        print(f"  - {package}")

    if local_modules:
        print("\nLocal modules found:")
        for module in sorted(local_modules):
            print(f"  - {module}")

    # Generate requirements.txt
    output_file = '../requirements_scanned.txt'
    generate_requirements(external_packages, output_file)
    print(f"\nRequirements file generated: {output_file}")


if __name__ == "__main__":
    main()