"""
Configuration management for the Agentic RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORDB_DIR = DATA_DIR / "vectordb"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
for directory in [DOCUMENTS_DIR, VECTORDB_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create .gitkeep files
for directory in [DOCUMENTS_DIR, VECTORDB_DIR, REPORTS_DIR]:
    gitkeep = directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Vector DB Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Agent Settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

# LLM Settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

print("✅ Configuration loaded successfully")
print(f"   Base directory: {BASE_DIR}")
print(f"   Vector DB directory: {VECTORDB_DIR}")
print(f"   Embedding model: {EMBEDDING_MODEL}")
print(f"   Embedding dimensions: {EMBEDDING_DIM}")