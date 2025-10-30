"""
FAISS Vector Store Wrapper
Handles document embedding, storage, and retrieval using FAISS.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, cast
import numpy as np
import faiss
from openai import OpenAI

from src.config import (OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM,
                        VECTORDB_DIR)


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, index_name: str = "main"):
        """
        Initialize vector store.

        Args:
            index_name: Name for the FAISS index (for multiple indices)
        """
        self.index_name = index_name
        self.embedding_dim = EMBEDDING_DIM
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=None  # Disable proxy
        )

        # FAISS index (HNSW for fast approximate search)
        self.index: Optional[faiss.Index] = None

        # Metadata storage (FAISS only stores vectors, not text)
        self.documents = []  # List of document texts
        self.metadata = []  # List of metadata dicts

        # Paths
        self.index_path = VECTORDB_DIR / f"{index_name}.faiss"
        self.metadata_path = VECTORDB_DIR / f"{index_name}_metadata.pkl"

        # Try to load existing index
        if self.index_path.exists():
            self.load()
            print(f"✅ Loaded existing index: {self.index_name}")
            print(f"   Documents: {len(self.documents)}")
        else:
            self._initialize_index()
            print(f"✅ Created new index: {self.index_name}")

    def _initialize_index(self):
        """
        Initialize a new FAISS index with HNSW.

        HNSW parameters:
        - M=32: Number of connections per layer
        - efConstruction=64: Quality of index construction
        """
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)

        # Set construction parameter
        self.index.hnsw.efConstruction = 64

        # Set search parameter
        self.index.hnsw.efSearch = 32

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Batch embedding for efficiency
        response = self.client.embeddings.create(model=EMBEDDING_MODEL,
                                                 input=texts)

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]

        # Convert to numpy array
        return np.array(embeddings, dtype=np.float32)

    def add_documents(self,
                      texts: List[str],
                      metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts

        Returns:
            List of document IDs (indices in the store)
        """
        if not texts:
            return []

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self._generate_embeddings(texts)

        # Add to FAISS index
        self.index.add(embeddings)  # type: ignore

        # Store documents and metadata
        start_id = len(self.documents)
        self.documents.extend(texts)

        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.metadata.extend(metadatas)

        # Return document IDs
        doc_ids = list(range(start_id, start_id + len(texts)))

        print(
            f"✅ Added {len(texts)} documents (IDs: {start_id}-{start_id + len(texts) - 1})"
        )

        return doc_ids

    def search(self,
               query: str,
               k: int = 5,
               return_scores: bool = True) -> List[Dict]:
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
            print("⚠️  Index is empty!")
            return []

        # Generate query embedding
        query_embedding = self._generate_embeddings([query])

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)  # type: ignore

        # Convert distances to similarity scores
        similarities = 1 / (1 + distances[0])

        # Build results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.documents):  # Valid index
                result = {
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                }
                if return_scores:
                    result['score'] = float(similarity)
                results.append(result)

        return results

    def save(self):
        """
        Save FAISS index and metadata to disk.
        """
        if self.index is None:
            print("⚠️  No index to save!")
            return

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata and documents
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(
                {
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)

        print(f"✅ Saved index to {self.index_path}")
        print(f"   Documents: {len(self.documents)}")

    def load(self):
        """
        Load FAISS index and metadata from disk.
        """
        if not self.index_path.exists():
            print(f"⚠️  Index not found: {self.index_path}")
            self._initialize_index()
            return

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata and documents
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']

        print(f"✅ Loaded index from {self.index_path}")
        print(f"   Documents: {len(self.documents)}")

    def clear(self):
        """
        Clear the index (remove all documents).
        """
        self._initialize_index()
        self.documents = []
        self.metadata = []
        print("✅ Cleared index")

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dict with stats
        """
        return {
            'index_name': self.index_name,
            'num_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'index_size_mb': self.index.ntotal * self.embedding_dim * 4 / (1024 * 1024) if self.index else 0,
            'has_index': self.index is not None,
            'index_path': str(self.index_path)
        }


# Test function
def test_vector_store():
    """
    Test the vector store with sample documents.
    """
    print("\n" + "=" * 50)
    print("Testing FAISS Vector Store")
    print("=" * 50 + "\n")

    # Create vector store
    vs = VectorStore(index_name="test")

    # Sample documents
    documents = [
        "FAISS is a library for efficient similarity search",
        "Vector databases store embeddings for semantic search",
        "HNSW is an algorithm for approximate nearest neighbor search",
        "Recommendation systems use collaborative filtering",
        "Neural networks learn patterns from data",
    ]

    # Add documents
    print("\n1. Adding documents...")
    vs.add_documents(documents)

    # Search
    print("\n2. Searching...")
    query = "How do vector databases work?"
    results = vs.search(query, k=3)

    print(f"\nQuery: {query}")
    print("\nTop 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text']}")

    # Save
    print("\n3. Saving index...")
    vs.save()

    # Stats
    print("\n4. Index stats:")
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 50)
    print("Test complete! ✅")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_vector_store()
