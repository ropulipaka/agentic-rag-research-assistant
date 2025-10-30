"""
Agentic RAG Research Assistant
A multi-agent system for autonomous research with RAG.
"""

__version__ = "0.1.0"

# Core components
from src.config import *
from src.vector_store import VectorStore
from src.model_router import route_request, route_embedding_request
from src.model_registry import MODEL_REGISTRY

__all__ = [
    "VectorStore",
    "route_request",
    "route_embedding_request",
    "MODEL_REGISTRY"
]