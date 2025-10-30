"""
Configuration management for the Agentic RAG system.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORDB_DIR = DATA_DIR / "vectordb"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DOCUMENTS_DIR, VECTORDB_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create .gitkeep files
for directory in [DOCUMENTS_DIR, VECTORDB_DIR, REPORTS_DIR]:
    gitkeep = directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "cost_optimized")
MONTHLY_BUDGET_USD = float(os.getenv("MONTHLY_BUDGET_USD", "100.0"))
COMPLEXITY_DETECTION_METHOD = os.getenv("COMPLEXITY_DETECTION_METHOD", "rule_based")

# Check available providers
PROVIDERS_AVAILABLE = {
    "openai": bool(OPENAI_API_KEY),
    "anthropic": bool(ANTHROPIC_API_KEY),
    "google": bool(GOOGLE_API_KEY),
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Log file paths
MAIN_LOG = LOGS_DIR / "main.log"
AGENTS_LOG = LOGS_DIR / "agents.log"
MODELS_LOG = LOGS_DIR / "models.log"
VECTOR_STORE_LOG = LOGS_DIR / "vector_store.log"
ERRORS_LOG = LOGS_DIR / "errors.log"

# Shared formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler (all logs to console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Main log handler (everything)
main_handler = RotatingFileHandler(
    MAIN_LOG,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
main_handler.setFormatter(formatter)

# Agent log handler (only agent logs)
agents_handler = RotatingFileHandler(
    AGENTS_LOG,
    maxBytes=10*1024*1024,
    backupCount=5
)
agents_handler.setFormatter(formatter)
agents_handler.addFilter(lambda record: record.name.startswith('agents'))

# Model log handler (model registry and router)
models_handler = RotatingFileHandler(
    MODELS_LOG,
    maxBytes=10*1024*1024,
    backupCount=5
)
models_handler.setFormatter(formatter)
models_handler.addFilter(lambda record: 'model' in record.name.lower())

# Vector store log handler
vector_handler = RotatingFileHandler(
    VECTOR_STORE_LOG,
    maxBytes=10*1024*1024,
    backupCount=5
)
vector_handler.setFormatter(formatter)
vector_handler.addFilter(lambda record: 'vector' in record.name.lower())

# Error log handler (only ERROR and CRITICAL)
error_handler = RotatingFileHandler(
    ERRORS_LOG,
    maxBytes=10*1024*1024,
    backupCount=5
)
error_handler.setFormatter(formatter)
error_handler.setLevel(logging.ERROR)

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    handlers=[
        console_handler,
        main_handler,
        agents_handler,
        models_handler,
        vector_handler,
        error_handler
    ]
)

logger = logging.getLogger(__name__)
logger.info("Configuration loaded successfully")
logger.info(f"Available providers: {[k for k, v in PROVIDERS_AVAILABLE.items() if v]}")
logger.info(f"Logs directory: {LOGS_DIR}")
logger.info(f"Log files: main.log, agents.log, models.log, vector_store.log, errors.log")

# Vector DB Settings
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Agent Settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

# LLM Settings (used by router)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Note: Model selection is now handled by model_router.py
# EMBEDDING_MODEL and LLM_MODEL are no longer needed here