# 🧠 InsightMuse RAG - Research Paper Analysis System

InsightMuse RAG is an advanced research paper analysis system that combines Retrieval-Augmented Generation (RAG) with sophisticated PDF processing and question-answering capabilities. The system uses FAISS for efficient similarity search, sentence transformers for embeddings, and OpenAI's GPT models for generation.

## 🏗️ System Architecture

### Core Components

1. **RAG System** (`src/rag_system.py`)

   - Uses FAISS for vector similarity search
   - Implements sentence-transformer embeddings (all-MiniLM-L6-v2)
   - Integrates with OpenAI's GPT-3.5-turbo for generation
   - Features:
     - TLDR generation with context-aware prompting
     - Question answering with relevant context retrieval
     - Dynamic context management for large documents

2. **PDF Processing** (`src/pdf_extractor.py`)

   - Extracts structured sections from research papers
   - Handles image extraction and processing
   - Maintains document structure and hierarchy

3. **RAG Pipeline** (`src/rag_pipeline.py`)

   - Manages the end-to-end RAG workflow
   - Handles text simplification and context retrieval
   - Integrates with the visualization system

4. **Visualization System** (`src/visualizer.py`)
   - Generates visual representations of research content
   - Supports multiple visualization types
   - Integrates with the RAG pipeline

### Data Processing Scripts

1. **Index Building** (`scripts/build_scitldr_index.py`)

   - Creates FAISS indices for the SciTLDR dataset
   - Processes and embeds research paper content
   - Manages vector storage and retrieval

2. **Data Loading** (`scripts/load_scitldr_data.py`)
   - Handles SciTLDR dataset loading
   - Preprocesses data for index building
   - Manages data versioning

## 🔍 RAG Implementation Details

### Vector Search System

```python
def search_similar(self, query: str, k: int = 3) -> List[Dict]:
    # Get query embedding using sentence-transformers
    query_embedding = self.encoder.encode([query])[0]

    # Search FAISS index
    D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)

    # Return results with metadata
    return [self.stored_data[idx] for idx in I[0] if idx < len(self.stored_data)]
```

### Prompting Techniques

1. **TLDR Generation**

   ```python
   prompt = f"""
   Based on these example TLDRs from similar research papers:
   {context}

   Please provide a concise TLDR summary of the following research paper section:
   {text}

   Focus on the key points and main findings. Keep it brief but informative.
   """
   ```

2. **Question Answering**

   ```python
   prompt = f"""
   Context from the research paper:
   {full_context}

   Question: {question}

   Please provide a detailed answer based on the context. Include specific references
   to the paper where relevant.
   """
   ```

## 🚀 Setup and Installation

### Prerequisites

- Python 3.11.7 (specified in runtime.txt)
- OpenAI API key
- Git

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/insightmuse-rag.git
   cd insightmuse-rag
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Building the Index

1. Prepare the SciTLDR dataset:

   ```bash
   python scripts/load_scitldr_data.py
   ```

2. Build the FAISS index:
   ```bash
   python scripts/build_scitldr_index.py
   ```

## 📚 API Reference

### FastAPI Endpoints

1. **PDF Upload** (`/upload_pdf`)

   - Accepts PDF files
   - Returns structured sections and metadata
   - Initiates background TLDR generation

2. **Question Answering** (`/qna`)
   - Accepts questions about uploaded papers
   - Returns AI-generated answers with relevant context
   - Supports section-specific queries

### RAG System Methods

1. `ResearchPaperRAG`
   - `generate_tldr(text: str) -> str`
   - `answer_question(question: str, context: Dict[str, str]) -> str`
   - `search_similar(query: str, k: int = 3) -> List[Dict]`
   - `get_relevant_context(query: str, k: int = 5) -> str`

## 🔧 Configuration

### Vector Index Settings

- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Index type: FAISS FlatL2
- Default k for similarity search: 3

### OpenAI Settings

- Model: gpt-3.5-turbo
- Temperature: 0.3 (for consistent outputs)
- Max tokens: 150 (TLDR), 500 (QA)

## 🧪 Testing

Run the test scripts:

```bash
python scripts/test_query.py  # Test RAG queries
python scripts/test_diagram.py  # Test visualization
```

## 📦 Deployment

The system is configured for deployment on Render:

- Uses uvicorn as the ASGI server
- Configured with Procfile and runtime.txt
- Environment variables managed through Render dashboard

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for their powerful language models
- The Streamlit team for their excellent web framework
- The open-source community for their valuable tools and libraries

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/insightmuse-rag](https://github.com/yourusername/insightmuse-rag)
