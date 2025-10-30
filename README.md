# 🤖 Agentic RAG Research Assistant

A multi-agent system that conducts comprehensive research on any topic using retrieval-augmented generation (RAG) with intelligent orchestration.

## 🎯 Features

- **Multi-Agent Architecture**: 6 specialized agents working together
- **Web Search Integration**: Automatically finds and processes relevant sources
- **FAISS Vector Search**: Fast semantic search using HNSW indexing
- **Autonomous Research**: Agents coordinate to build comprehensive reports
- **Citation & Fact Checking**: Verifies claims and adds proper citations
- **Persistent Knowledge**: Builds growing knowledge base for follow-up queries

## 🏗️ Architecture
```
User Query
    ↓
Query Analyzer → Break into sub-questions
    ↓
Web Searcher → Find relevant sources
    ↓
Document Processor → Extract, chunk, embed content
    ↓
Retrieval Agent → Store in FAISS, retrieve relevant chunks
    ↓
Synthesis Agent → Generate comprehensive report
    ↓
Fact Checker → Verify claims and add citations
    ↓
Final Report
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Usage
```python
from src.orchestrator import ResearchOrchestrator

# Initialize system
orchestrator = ResearchOrchestrator()

# Conduct research
report = orchestrator.research(
    query="What are the latest trends in recommendation systems?"
)

# Report is automatically saved to data/reports/
print(report)
```

## 📚 Project Structure
```
agentic-rag-research-assistant/
├── src/
│   ├── config.py              # Configuration
│   ├── vector_store.py        # FAISS wrapper
│   ├── utils.py               # Helper functions
│   ├── orchestrator.py        # Multi-agent coordinator
│   └── agents/                # Specialized agents
│       ├── query_analyzer.py
│       ├── web_searcher.py
│       ├── document_processor.py
│       ├── retrieval_agent.py
│       ├── synthesis_agent.py
│       └── fact_checker.py
├── data/                      # Data storage
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
└── examples/                  # Usage examples
```

## 🛠️ Technologies

- **LangChain**: Agent orchestration
- **OpenAI GPT-4**: Language model for agents
- **FAISS**: Vector similarity search with HNSW
- **BeautifulSoup**: Web scraping
- **Python 3.11+**: Core language

## 📖 Documentation

Coming soon...

## 🤝 Contributing

This is a portfolio project, but feel free to fork and adapt!

## 📄 License

MIT License

## 👤 Author

[Your Name]
- LinkedIn: [[Profile]](https://www.linkedin.com/in/rohit-pulipaka/)
- GitHub: [@ropulipaka](https://github.com/ropulipaka)