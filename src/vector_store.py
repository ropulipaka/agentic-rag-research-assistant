"""
FAISS Vector Store Wrapper
Handles document embedding, storage, and retrieval using FAISS.
Shared resource that can be accessed by multiple agents.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss

from src.config import EMBEDDING_DIM, VECTORDB_DIR
from src.model_router import route_embedding_request

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search with HNSW indexing."""

    def __init__(
        self,
        index_name: str = "main",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 32
    ):
        """
        Initialize vector store.

        Args:
            index_name: Name for the FAISS index (for multiple indices)
            hnsw_m: Number of connections per layer in HNSW graph
            hnsw_ef_construction: Size of dynamic candidate list during construction
            hnsw_ef_search: Size of dynamic candidate list during search
        """
        self.index_name = index_name
        self.embedding_dim = EMBEDDING_DIM

        # HNSW parameters
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # FAISS index (HNSW for fast approximate search)
        self.index: Optional[faiss.Index] = None

        # Metadata storage (FAISS only stores vectors, not text)
        self.documents = []
        self.metadata = []

        # Paths
        self.index_path = VECTORDB_DIR / f"{index_name}.faiss"
        self.metadata_path = VECTORDB_DIR / f"{index_name}_metadata.pkl"

        # Try to load existing index
        if self.index_path.exists():
            self.load()
        else:
            self._initialize_index()
            logger.info(f"Created new index: {self.index_name}")

    def _initialize_index(self):
        """Initialize a new FAISS index with HNSW."""
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        self.index.hnsw.efSearch = self.hnsw_ef_search
        logger.info(f"Initialized HNSW index with M={self.hnsw_m}, "
                   f"efConstruction={self.hnsw_ef_construction}, "
                   f"efSearch={self.hnsw_ef_search}")

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using model router.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = route_embedding_request(texts)
        return np.array(embeddings, dtype=np.float32)

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts

        Returns:
            List of document IDs (indices in the store)
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return []

        logger.info(f"Adding {len(texts)} documents to vector store")

        # Generate embeddings
        embeddings = self._generate_embeddings(texts)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store documents and metadata
        start_id = len(self.documents)
        self.documents.extend(texts)

        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.metadata.extend(metadatas)

        doc_ids = list(range(start_id, start_id + len(texts)))
        logger.info(f"Added documents with IDs: {start_id}-{start_id + len(texts) - 1}")
        logger.info(f"Total documents in index: {len(self.documents)}")

        return doc_ids

    def search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            return_scores: Whether to include similarity scores

        Returns:
            List of dicts with 'text', 'metadata', and optionally 'score'
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, cannot perform search")
            return []

        logger.debug(f"Searching for: '{query[:50]}...' (k={k})")

        # Generate query embedding
        query_embedding = self._generate_embeddings([query])

        # Search FAISS index (returns L2 distances)
        distances, indices = self.index.search(query_embedding, k)

        # Convert L2 distances to similarity scores (0-1, higher is better)
        # Using formula: similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances[0])

        # Build results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.documents):
                continue

            result = {
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
            }
            if return_scores:
                result['score'] = float(similarity)
            results.append(result)

        logger.info(f"Found {len(results)} results")
        return results

    def save(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # Create directory if needed
        os.makedirs(VECTORDB_DIR, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

        logger.info(f"Saved index to {self.index_path} ({len(self.documents)} documents)")

    def load(self):
        """Load FAISS index and metadata from disk."""
        if not self.index_path.exists():
            logger.warning(f"Index not found: {self.index_path}")
            self._initialize_index()
            return

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Restore HNSW search parameters (not saved in index file)
            self.index.hnsw.efSearch = self.hnsw_ef_search

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']

            logger.info(f"Loaded index from {self.index_path} ({len(self.documents)} documents)")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._initialize_index()

    def clear(self):
        """Clear the index (remove all documents)."""
        self._initialize_index()
        self.documents = []
        self.metadata = []
        logger.info("Cleared index")

    def delete(self):
        """Delete the index files from disk."""
        if self.index_path.exists():
            os.remove(self.index_path)
            logger.info(f"Deleted index file: {self.index_path}")

        if self.metadata_path.exists():
            os.remove(self.metadata_path)
            logger.info(f"Deleted metadata file: {self.metadata_path}")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        num_docs = len(self.documents)
        index_size_mb = 0

        if self.index and self.index.ntotal > 0:
            # Approximate size: ntotal vectors * dimension * 4 bytes (float32)
            index_size_mb = self.index.ntotal * self.embedding_dim * 4 / (1024 * 1024)

        return {
            'index_name': self.index_name,
            'num_documents': num_docs,
            'embedding_dim': self.embedding_dim,
            'index_size_mb': round(index_size_mb, 2),
            'has_index': self.index is not None and self.index.ntotal > 0,
            'index_path': str(self.index_path),
            'hnsw_params': {
                'M': self.hnsw_m,
                'efConstruction': self.hnsw_ef_construction,
                'efSearch': self.hnsw_ef_search
            }
        }


def test_vector_store():
    """Test the vector store with sample documents."""
    print("\n" + "="*70)
    print("TESTING VECTOR STORE")
    print("="*70 + "\n")

    vs = VectorStore(index_name="test")
    vs.clear()

    print("📝 Adding test documents...\n")

    documents = [
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Vector databases store embeddings for semantic search and retrieval.",
        "HNSW is a graph-based algorithm for approximate nearest neighbor search.",
        "Collaborative filtering is a recommendation technique that analyzes user behavior patterns.",
        "Neural networks learn hierarchical patterns and representations from data.",
    ]

    metadatas = [
        {"topic": "vector_search", "source": "faiss_docs"},
        {"topic": "vector_search", "source": "db_guide"},
        {"topic": "algorithms", "source": "hnsw_paper"},
        {"topic": "recommendations", "source": "recsys_book"},
        {"topic": "machine_learning", "source": "ml_textbook"}
    ]

    doc_ids = vs.add_documents(documents, metadatas)
    print(f"✅ Added {len(doc_ids)} documents (IDs: {doc_ids[0]}-{doc_ids[-1]})\n")

    # Test searches
    print("="*70)
    print("TEST SEARCHES")
    print("="*70 + "\n")

    test_queries = [
        "How do vector databases work?",
        "What is collaborative filtering?",
        "Tell me about HNSW algorithm"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)

        results = vs.search(query, k=3)

        for j, result in enumerate(results, 1):
            print(f"\n  Result {j}:")
            print(f"  └─ Score: {result['score']:.4f}")
            print(f"  └─ Topic: {result['metadata'].get('topic', 'N/A')}")
            print(f"  └─ Text: {result['text'][:80]}...")
        print()

    # Save and test persistence
    print("="*70)
    print("TESTING PERSISTENCE")
    print("="*70 + "\n")

    vs.save()
    print("✅ Saved index to disk\n")

    # Create new instance (should load saved index)
    vs2 = VectorStore(index_name="test")
    stats = vs2.get_stats()

    print("📊 Loaded Index Stats:")
    for key, value in stats.items():
        if key == 'hnsw_params':
            print(f"   {key}:")
            for param, val in value.items():
                print(f"      {param}: {val}")
        else:
            print(f"   {key}: {value}")

    # Test search on loaded index
    print(f"\n🔍 Test search on loaded index...")
    results = vs2.search("vector search", k=2)
    print(f"✅ Search successful! Found {len(results)} results")

    print("\n" + "="*70)
    print("Vector Store test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_vector_store()