# AI-Powered Document Assistant for Subsurface Geoscience Data

## Document GraphRAG Q&A Assistant

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)

A comprehensive solution for extracting, organizing, and querying knowledge from geoscience documents using Knowledge Graphs and Retrieval Augmented Generation (RAG).

## Overview

The Document GraphRAG Q&A Assistant is a Streamlit application designed to ingest various document types (PDFs, text files, images), extract knowledge from them to build a knowledge graph and a vector store, and then allow users to ask natural language questions against this processed information. It leverages Large Language Models (LLMs) for knowledge graph triple extraction, Cypher query generation, and answer synthesis, combining the strengths of graph-based RAG and semantic vector search.

## Key Features

* **Multi-Page Streamlit Interface:**
  * **Data Ingestion Page:** For uploading documents, managing processing, and viewing detailed audit history
  * **Chat Assistant Page:** For asking natural language questions about ingested documents

* **Document Processing Pipeline:**
  * Supports PDF, TXT, PNG, JPG, JPEG file types
  * OCR for image-based documents and PDFs using Mistral AI
  * Advanced text chunking strategies with LangChain's `RecursiveCharacterTextSplitter`
  * Knowledge Graph (KG) triple extraction (Subject-Predicate-Object) using configurable LLMs
  * Optional KG standardization and inference steps
  * Storage of extracted triples into a Neo4j graph database
  * Generation of embeddings for text chunks using Sentence Transformers
  * Storage of text chunks and embeddings in a ChromaDB vector store

* **Background Ingestion:** Document processing runs in background threads, keeping the UI responsive

* **Persistent Audit Trail:**
  * Uses an SQLite database to log all ingestion jobs
  * Tracks job status, file processing status, timestamps, extracted counts, and error messages

* **GraphRAG Q&A Engine:**
  * Entity linking from user questions to the Neo4j graph
  * LLM-powered Cypher query generation based on graph schema
  * Retry mechanisms with query correction and evaluation
  * Semantic vector search against ChromaDB
  * Combines context from both graph results and vector search
  * LLM-powered answer synthesis based on the combined context
  * Displays sources and generated Cypher queries for transparency

* **Additional Features:**
  * Few-shot learning for Cypher generation
  * Caching for KG extraction results
  * Configuration-driven setup
  * Efficient resource management
  * Experimental KG visualization
  * Persistent chat history

## Architecture Overview

The application consists of the following main components:

1. **Streamlit Frontend:**
   * `GraphRAG_Document_AI_Platform.py`: Main application entry point
   * `pages/1_knowledge_chat_assistant.py`: Chat interface
   * `pages/2_document_ingestion.py`: Document uploading and audit trail display

2. **Backend Processing & Logic:**
   * `processing_pipeline.py`: Orchestrates document ingestion workflow
   * `audit_db_manager.py`: Manages interactions with SQLite audit database
   * `graph_rag_qa.py`: Core engine for answering questions
   * `neo4j_exporter.py`: Stores extracted triples into Neo4j
   * Various utility modules for LLM interaction, text processing, and visualization

3. **Databases:**
   * **Neo4j:** Stores the knowledge graph (entities and relationships)
   * **ChromaDB:** Stores vector embeddings of text chunks
   * **SQLite:** Stores the audit trail for data ingestion jobs

4. **External Services:**
   * **LLMs (e.g., Gemini, Mistral):** Used for KG extraction, Cypher generation, answer synthesis, and OCR

## Setup and Installation

### Prerequisites

* Python 3.10 or higher
* Neo4j Desktop or Server instance running and accessible
* (Optional but Recommended for Windows) Microsoft C++ Build Tools

### Installation Steps

1. **Clone Repository:**
   ```bash
   # git clone https://github.com/MIVAA-ai/mivaa-graphrag-assistant.git
   # cd mivaa-graphrag-assistant
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   
   # For spaCy model if implementing coreference:
   python -m spacy download en_core_web_trf
   ```

4. **Configuration:**

   Create `config.toml` in the project root:
   ```toml
   [llm]
   model = "gemini-1.5-flash-latest"
   api_key = "YOUR_LLM_API_KEY"
   base_url = "YOUR_LLM_BASE_URL_IF_NEEDED"
   
       [llm.triple_extraction]
       model = "gemini-1.5-flash-latest"
       api_key = "YOUR_LLM_API_KEY"
       base_url = "YOUR_LLM_BASE_URL_IF_NEEDED"
       max_tokens = 2000
       temperature = 0.1
   
       [llm.ocr]
       mistral_api_key = "YOUR_MISTRAL_API_KEY"
   
   [embeddings]
   model_name = "all-MiniLM-L6-v2"
   
   [chunking]
   chunk_size = 1000
   overlap = 100
   
   [vector_db]
   persist_directory = "./chroma_db_pipeline"
   collection_name = "doc_pipeline_embeddings"
   
   [database]
   name = "neo4j"
   
   [caching]
   enabled = true
   
   [standardization]
   enabled = true
   
   [inference]
   enabled = true
   
   [nlp]
   COREFERENCE_RESOLUTION_ENABLED = false
   SPACY_MODEL_NAME = "en_core_web_trf"
   ```

   Create `graph_config.ini` for database connections:
   ```ini
   [neo4j]
   uri = bolt://localhost:7687
   user = neo4j
   password = your_neo4j_password
   
   [vector_db]
   ```

5. **Initialize Audit Database:**
   * The audit database schema will be created automatically on first run

## Running the Application

1. **Ensure Neo4j is running**

2. **Activate your Python virtual environment:**
   ```bash
   # Windows
   .\.venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run GraphRAG_Document_AI_Platform.py
   ```

   For development (to avoid file watcher issues):
   ```bash
   streamlit run GraphRAG_Document_AI_Platform.py --server.fileWatcherType none
   ```

4. **Access the application** in your web browser (usually at http://localhost:8501)

## Usage

### Data Ingestion Page

1. Navigate to the "Data Ingestion" page from the sidebar
2. Click "Select documents to process" to upload your files (PDFs, TXT, images)
3. Optionally, toggle "Use Processing Cache"
4. Click "🚀 Start Ingestion Job"
5. The job will run in the background with status updates
6. Review the "Ingestion History" table for past and current jobs
7. Select a "Job ID" to view detailed processing status for each file
8. For completed jobs, use "📊 Generate & Show Graph" to visualize the knowledge graph

### Chat Assistant Page

1. Navigate to the "Chat Assistant" page
2. The page will indicate available data context
3. Type your question into the chat input box and press Enter
4. Review the assistant's answer
5. Expand "Show Sources" and "Show Cypher Query Used" for additional context
6. Chat history is saved automatically to chat_history.json
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
