# 🧠 InsightMuse - Research Paper Explainer & Visualizer

InsightMuse is an intelligent tool that helps researchers, students, and academics understand complex research papers through simplified explanations and visual representations. It combines Retrieval-Augmented Generation (RAG) with advanced visualization techniques to make research more accessible.

## ✨ Features

- **📄 PDF Processing**: Upload and process research papers in PDF format
- **📊 CSV Support**: Batch process multiple abstracts from CSV files
- **🧠 Intelligent Simplification**: Convert complex research text into clear, accessible language
- **📈 Smart Visualization**: Generate appropriate visualizations based on content type:
  - Flowcharts for methodologies and procedures
  - Statistical charts for results and findings
  - Summaries for discussions and conclusions
- **🔍 Section-wise Analysis**: Process specific sections of research papers
- **🎯 Interactive Interface**: User-friendly web interface built with Streamlit

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

### Installation

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

4. Set up your environment variables:
   Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_api_key_here
```

### Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Choose your input method:

   - Upload a PDF research paper
   - Upload a CSV file with abstracts

4. Select the section you want to analyze (for PDFs)

5. Click "Simplify & Visualize" to generate explanations and visualizations

## 🏗️ Architecture

InsightMuse is built using a modular architecture:

- **Web Interface** (`app.py`): Streamlit-based user interface
- **RAG Pipeline** (`src/rag_pipeline.py`): Handles text simplification and retrieval
- **Visualization System** (`src/visualizer.py`): Generates appropriate visualizations
- **PDF Processing** (`src/pdf_extractor.py`): Handles PDF document parsing
- **Vector Index** (`index/`): Stores embeddings for efficient retrieval

## 🛠️ Technical Stack

- **Natural Language Processing**: OpenAI GPT-3.5, Sentence Transformers
- **Vector Search**: FAISS
- **Visualization**: Graphviz, Matplotlib
- **Web Framework**: Streamlit
- **PDF Processing**: PyMuPDF
- **Data Processing**: Pandas

## 📁 Project Structure

```
insightmuse-rag/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── src/               # Core implementation
│   ├── rag_pipeline.py    # RAG implementation
│   ├── visualizer.py      # Visualization logic
│   ├── pdf_extractor.py   # PDF processing
│   ├── prompt.py          # Prompt templates
│   ├── build_index.py     # Vector index creation
│   └── load_data.py       # Data loading utilities
├── index/             # Vector index storage
├── data/              # Data storage
└── outputs/           # Generated visualizations
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for their powerful language models
- The Streamlit team for their excellent web framework
- The open-source community for their valuable tools and libraries

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/insightmuse-rag](https://github.com/yourusername/insightmuse-rag)
