# 🤖 Agentic RAG Research Assistant

A production-ready multi-agent system that conducts comprehensive research on any topic using retrieval-augmented generation (RAG) with intelligent LLM-powered orchestration and cost optimization.

## ✨ Key Features

- **🧠 Intelligent Orchestration**: LLM-powered planning that adapts execution strategy to query complexity
- **💰 Cost Optimization**: Up to 95% cost savings through intelligent model routing
- **⚡ High Performance**: HNSW vector search with sub-second retrieval from 10,000+ documents
- **🔍 Multi-Agent Pipeline**: 5 specialized agents working in harmony
- **🌐 Web Search Integration**: Automatic source discovery via Tavily API
- **📚 Source Citations**: Full citation tracking with relevance scores
- **💾 Persistent Knowledge**: Growing FAISS vector database for continuous learning
- **📊 Comprehensive Metrics**: Real-time cost and time tracking per agent

## 🎯 What Makes This Special

This isn't just another RAG system. It demonstrates:

✅ **Production-grade architecture** with proper separation of concerns  
✅ **Intelligent cost management** - uses GPT-4o-mini for planning, GPT-4.1 for synthesis, GPT-5 only when needed  
✅ **Scalable vector search** - HNSW indexing handles 10,000+ documents with <1s retrieval  
✅ **Dynamic query decomposition** - automatically breaks complex queries into focused sub-questions  
✅ **Enterprise-ready logging** - separate logs for agents, models, vector store, and errors  
✅ **Meta-level intelligence** - the orchestrator plans its own execution strategy  

## 🏗️ Architecture
```
User Query
    ↓
📋 Orchestrator (LLM Planning)
    ├─ Analyzes query complexity
    ├─ Selects optimal strategy (cost/balanced/quality)
    └─ Plans execution parameters dynamically
    ↓
🔍 Query Analyzer
    └─ Generates 2-15 focused sub-questions (based on complexity)
    ↓
🌐 Web Searcher (Tavily API)
    └─ Finds 10-300 relevant sources (based on sub-questions)
    ↓
📄 Document Processor
    └─ Creates 50-2000+ clean, overlapping chunks (sentence-aware)
    ↓
💾 Vector Store (FAISS HNSW)
    └─ Indexes with text-embedding-3-small
    ↓
🎯 Retrieval Agent
    └─ Retrieves top-k most relevant chunks (5-20 based on plan)
    ↓
✍️ Synthesis Agent
    └─ Generates comprehensive answer (length varies by complexity)
    ↓
📊 Final Report with Citations
```

## 📊 Performance Benchmarks

### Example: Complex Query
```
Query: "Explain how two-tower models work in recommendation systems 
        and compare them to collaborative filtering approaches"

Orchestrator Analysis:
├─ Detected complexity: Complex
├─ Selected strategy: Quality-optimized
└─ Planned parameters: 7 sub-queries, 10 results each

Results:
├─ Sub-queries generated: 7
├─ Search results found: 70 (7 × 10)
├─ Chunks processed: 713
├─ Chunks retrieved: 10
├─ Answer length: 4,638 characters
├─ Sources cited: 35
├─ Total time: 58.55s
└─ Total cost: $0.0109

Agent Breakdown:
├─ Query Analysis:     2.69s  |  $0.0002
├─ Web Search:        27.71s  |  $0.0000 (Tavily)
├─ Doc Processing:     0.04s  |  $0.0000
├─ Vector Indexing:    2.97s  |  $0.0005 (embeddings)
├─ Context Retrieval:  0.65s  |  $0.0000
└─ Synthesis:         21.41s  |  $0.0100 (GPT-4.1)
```

### Example: Simple Query
```
Query: "What is FAISS?"

Orchestrator Analysis:
├─ Detected complexity: Simple
├─ Selected strategy: Cost-optimized
└─ Planned parameters: 3 sub-queries, 5 results each

Results:
├─ Sub-queries: 3
├─ Search results: 15 (3 × 5)
├─ Chunks processed: ~80
├─ Chunks retrieved: 5
├─ Total time: ~15s
└─ Total cost: ~$0.003

*Numbers vary based on query complexity*
```

### Cost Comparison

| Approach | Model Strategy | Cost per Query | Savings |
|----------|---------------|----------------|---------|
| Naive (always GPT-5) | Single model | $0.10-0.30 | - |
| **Orchestrated (balanced)** | **Intelligent routing** | **$0.003-0.02** | **85-95%** |
| Orchestrated (cost-optimized) | Cheapest models | $0.001-0.005 | 95-99% |
| Orchestrated (quality-optimized) | Best models | $0.10-0.30 | 0-20% (highest quality) |

*Actual costs depend on query complexity and number of sources*

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key (free tier available)

### Installation
```bash
# Clone repository
git clone https://github.com/ropulipaka/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add:
#   OPENAI_API_KEY=your_key_here
#   TAVILY_API_KEY=your_key_here
```

### Usage

#### Run Full Orchestrator Test
```bash
python -m src.orchestrator
```

#### Use in Your Code
```python
from src.orchestrator import OrchestratorAgent

# Initialize with strategy
orchestrator = OrchestratorAgent(global_strategy="balanced")

# Conduct research
result = await orchestrator.research(
    query="How do recommendation systems work at Meta?",
    user_strategy="cost_optimized"  # Optional override
)

# Access results
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
print(f"Cost: ${result['metrics']['total_cost_usd']}")
print(f"Time: {result['metrics']['total_time_seconds']}s")
```

#### Test Individual Agents
```bash
# Test each agent independently
python -m src.agents.query_analyzer
python -m src.agents.web_searcher
python -m src.agents.document_processor
python -m src.agents.retrieval_agent
python -m src.agents.synthesis_agent
```

## 🎯 Routing Strategies

The orchestrator supports 4 strategies:

| Strategy | Use Case | Models Used | Cost Range | Quality |
|----------|----------|-------------|------------|---------|
| `cost_optimized` | High volume, budget-conscious | GPT-4o-mini | $0.001-0.005 | Good |
| `balanced` | ⭐ **Recommended** | GPT-4o-mini + GPT-4.1 | $0.003-0.02 | Great |
| `quality_optimized` | Critical research, no budget limit | GPT-4.1 + GPT-5 | $0.10-0.30 | Excellent |
| `latency_optimized` | Real-time applications | Fastest models | $0.005-0.03 | Good |

*Costs vary by query complexity*

## 🔄 How Query Complexity Affects Execution

The orchestrator dynamically adjusts all parameters based on the query:

| Query Type | Sub-Questions | Sources per Query | Total Sources | Typical Cost | Example |
|------------|--------------|-------------------|---------------|--------------|---------|
| **Simple** | 2-3 | 5 | 10-15 | $0.001-0.003 | "What is FAISS?" |
| **Medium** | 4-6 | 10 | 40-60 | $0.005-0.015 | "How do recommendation systems work?" |
| **Complex** | 7-10 | 10-15 | 70-150 | $0.01-0.05 | "Compare collaborative filtering and two-tower models" |
| **Very Complex** | 10-15 | 15-20 | 150-300 | $0.05-0.30 | "Design a complete RecSys for TikTok at scale" |

**Everything adapts automatically:**
- Number of sub-questions
- Search results per question  
- Chunks to retrieve
- Model selection (GPT-4o-mini vs GPT-4.1 vs GPT-5)
- Synthesis depth (brief vs detailed vs comprehensive)

## 📦 Project Structure
```
agentic-rag-research-assistant/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_analyzer.py      # Breaks queries into sub-questions
│   │   ├── web_searcher.py        # Tavily API integration
│   │   ├── document_processor.py  # Text chunking & cleaning
│   │   ├── retrieval_agent.py     # FAISS vector search
│   │   └── synthesis_agent.py     # Answer generation with citations
│   ├── orchestrator.py            # 🧠 Intelligent coordinator
│   ├── model_router.py            # Smart model selection
│   ├── model_registry.py          # Model definitions & pricing
│   ├── vector_store.py            # FAISS HNSW wrapper
│   └── config.py                  # Configuration management
├── data/
│   ├── vectordb/                  # Persistent FAISS indices
│   ├── reports/                   # Generated reports
│   └── cache/                     # Query cache
├── logs/                          # Structured logging
│   ├── main.log
│   ├── agents.log
│   ├── models.log
│   └── vector_store.log
├── models.yaml                    # Model registry
├── requirements.txt
└── README.md
```

## 🛠️ Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Routing** | Custom Router | Intelligent model selection |
| **Models** | GPT-4o-mini, GPT-4.1, GPT-5 | Query analysis, synthesis |
| **Embeddings** | text-embedding-3-small | Vector representations |
| **Vector Search** | FAISS (HNSW) | Sub-second similarity search |
| **Web Search** | Tavily API | Real-time source discovery |
| **Document Processing** | Custom | Text extraction & chunking |
| **Orchestration** | Custom async framework | Agent coordination |
| **Config** | YAML | System configuration |

## 💡 Technical Highlights

### 1. Intelligent Planning
The orchestrator uses an LLM to analyze each query and create an optimal execution plan:
```python
# LLM analyzes query complexity and selects strategy
plan = {
    "complexity": "simple|medium|complex",  # Dynamically determined
    "strategy": "cost_optimized|balanced|quality_optimized",
    "parameters": {
        "use_sub_queries": True/False,
        "num_search_results": 5-20,  # Varies by complexity
        "num_retrieval_chunks": 5-20
    }
}
```

### 2. Cost-Aware Model Selection
```python
# Automatically routes to optimal model based on task and complexity
if task == "query_analysis":
    model = "gpt-4o-mini"  # Always cheap for planning
elif task == "synthesis" and strategy == "cost_optimized":
    model = "gpt-4o-mini"  # $0.15/$0.60 per 1M tokens
elif task == "synthesis" and strategy == "balanced":
    model = "gpt-4.1"      # $3/$12 per 1M tokens
elif task == "synthesis" and strategy == "quality_optimized":
    model = "gpt-5"        # $15/$60 per 1M tokens
```

### 3. Production-Grade Vector Search
```python
# HNSW parameters optimized for speed/accuracy balance
HNSW(
    M=32,                   # Connections per layer
    efConstruction=200,     # Build-time accuracy
    efSearch=128            # Search-time accuracy
)
# Result: <1s retrieval from 10,000+ documents
```

### 4. Smart Chunking
```python
# Sentence-aware chunking with overlap
DocumentProcessor(
    chunk_size=800,         # Optimal for embeddings
    chunk_overlap=200       # Maintains context between chunks
)
```

## 📈 Scalability

| Metric | Current | Tested | Production-Ready |
|--------|---------|--------|------------------|
| Documents | 1,553 | 10,000+ | ✅ Yes |
| Queries/day | N/A | 1,000+ | ✅ Yes |
| Concurrent users | 1 | 50+ | ⚠️ Needs load balancer |
| Index size | 4.92 MB | 100+ MB | ✅ Yes |
| Retrieval time | 0.65s | <1s @ 10K docs | ✅ Yes |

## 🎓 Learning Value

This project demonstrates:

1. **System Design**: Multi-agent architecture with clear separation of concerns
2. **Cost Engineering**: 85-95% cost reduction through intelligent routing
3. **Production Patterns**: Logging, error handling, metrics, persistence
4. **ML Engineering**: Vector search, embeddings, semantic retrieval
5. **LLM Engineering**: Prompt engineering, model selection, cost optimization
6. **Adaptive Systems**: Dynamic parameter tuning based on query complexity
7. **Performance**: HNSW indexing, caching, efficient chunking

## 🔮 Roadmap

- [ ] FastAPI REST API server
- [ ] React frontend with real-time streaming
- [ ] User authentication & query history
- [ ] Advanced analytics dashboard
- [ ] Multi-modal support (images, PDFs)
- [ ] Distributed vector store (Qdrant/Weaviate)
- [ ] A/B testing framework for strategies
- [ ] Containerization (Docker)
- [ ] Kubernetes deployment configs

## 🤝 Contributing

This is a portfolio project, but feel free to:
- ⭐ Star the repo
- 🐛 Report issues
- 💡 Suggest features
- 🔧 Submit PRs

## 📄 License

MIT License - feel free to use this for learning or in your own projects!

## 👤 Author

**Rohit Pulipaka**
- 💼 [LinkedIn](https://www.linkedin.com/in/rohit-pulipaka/)
- 🐙 [GitHub](https://github.com/ropulipaka)

---

### ⭐ If this helped you, please star the repo!