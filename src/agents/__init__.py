"""
Agent modules for the research assistant.
"""

from .query_analyzer import QueryAnalyzerAgent

# Will be added as we build them:
from .web_searcher import WebSearcherAgent
from .document_processor import DocumentProcessorAgent
from .retrieval_agent import RetrievalAgent
from .synthesis_agent import SynthesisAgent

__all__ = [
    "QueryAnalyzerAgent",
    "WebSearcherAgent",
    "DocumentProcessorAgent",
    "RetrievalAgent",
    "SynthesisAgent",
]