"""
Retrieval Agent
Manages document storage and retrieval using the shared FAISS vector store.
Provides agent interface for multi-agent orchestration.
"""

import logging
from typing import List, Dict, Any, Tuple

from src.vector_store import VectorStore

logger = logging.getLogger("agents.retrieval_agent")


class RetrievalAgent:
    """
    Agent that handles document storage and retrieval using vector search.
    Uses the shared VectorStore for persistence and cross-agent access.
    """

    def __init__(self, index_name: str = "main"):
        """
        Initialize the Retrieval Agent.

        Args:
            index_name: Name for the vector store index (allows multiple indices)
        """
        self.index_name = index_name

        # Initialize shared vector store
        self.vector_store = VectorStore(
            index_name=index_name,
            hnsw_m=32,              # Good balance of speed/accuracy
            hnsw_ef_construction=200,  # Higher quality index
            hnsw_ef_search=128      # Fast search with good recall
        )

        stats = self.vector_store.get_stats()
        logger.info(f"Retrieval Agent initialized")
        logger.info(f"  Index: {index_name}")
        logger.info(f"  Current documents: {stats['num_documents']}")
        logger.info(f"  Index size: {stats['index_size_mb']} MB")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add processed chunks to the vector store.

        Args:
            chunks: List of chunks from Document Processor (with text and metadata)

        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        # Add to vector store
        doc_ids = self.vector_store.add_documents(texts, metadatas)

        # Auto-save after adding
        self.vector_store.save()

        logger.info(f"Successfully added {len(chunks)} chunks")
        logger.info(f"Total documents in index: {self.vector_store.get_stats()['num_documents']}")

        return len(doc_ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for relevant chunks given a query.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1, higher is more similar)

        Returns:
            List of (chunk_data, score) tuples, sorted by relevance (highest first)
        """
        logger.info(f"Searching for: '{query[:50]}...' (top_k={top_k})")

        # Search vector store
        results = self.vector_store.search(query, k=top_k, return_scores=True)

        # Convert to agent format: (chunk_data, score) tuples
        formatted_results = []
        for result in results:
            # Apply minimum score filter
            if result["score"] < min_score:
                continue

            chunk_data = {
                "text": result["text"],
                "metadata": result["metadata"]
            }
            score = result["score"]
            formatted_results.append((chunk_data, score))

        logger.info(f"Found {len(formatted_results)} relevant chunks (min_score={min_score})")

        return formatted_results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all chunks in the vector store.
        Useful for synthesis or fact-checking agents.

        Returns:
            List of all chunk data with metadata
        """
        all_chunks = []
        for i, (text, metadata) in enumerate(zip(self.vector_store.documents, self.vector_store.metadata)):
            all_chunks.append({
                "id": i,
                "text": text,
                "metadata": metadata
            })

        logger.info(f"Retrieved all {len(all_chunks)} chunks from vector store")
        return all_chunks

    def clear_index(self):
        """Clear the entire index and metadata."""
        self.vector_store.clear()
        self.vector_store.save()
        logger.info("Cleared index and metadata")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self.vector_store.get_stats()


def test_retrieval_agent():
    """Test the Retrieval Agent."""
    print("\n" + "="*70)
    print("TESTING RETRIEVAL AGENT")
    print("="*70 + "\n")

    # Clear any existing index for clean test
    agent = RetrievalAgent(index_name="test_agent")
    agent.clear_index()

    print("📝 Creating test chunks...\n")

    # Simulate chunks from Document Processor (Agent #3)
    test_chunks = [
        {
            "text": "Collaborative filtering is a recommendation technique that analyzes user behavior patterns. It identifies users with similar preferences and recommends items based on what similar users liked.",
            "metadata": {
                "source_url": "https://example.com/collab-filtering",
                "question": "What is collaborative filtering?",
                "chunk_index": 0,
                "total_chunks": 3
            }
        },
        {
            "text": "Content-based filtering recommends items similar to those a user has liked in the past. It analyzes item features and user preferences to make recommendations.",
            "metadata": {
                "source_url": "https://example.com/content-based",
                "question": "What is content-based filtering?",
                "chunk_index": 1,
                "total_chunks": 3
            }
        },
        {
            "text": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports both exact and approximate nearest neighbor search algorithms like HNSW.",
            "metadata": {
                "source_url": "https://example.com/faiss",
                "question": "What is FAISS?",
                "chunk_index": 0,
                "total_chunks": 2
            }
        },
        {
            "text": "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It provides excellent recall with fast search times, making it ideal for large-scale vector databases.",
            "metadata": {
                "source_url": "https://example.com/hnsw",
                "question": "What is HNSW?",
                "chunk_index": 1,
                "total_chunks": 2
            }
        },
        {
            "text": "Two-tower models are neural network architectures used in recommendation systems. They have separate towers for encoding users and items, which are then compared using dot product or cosine similarity.",
            "metadata": {
                "source_url": "https://example.com/two-tower",
                "question": "What are two-tower models?",
                "chunk_index": 0,
                "total_chunks": 1
            }
        }
    ]

    print(f"Test chunks: {len(test_chunks)}\n")

    # Test 1: Add chunks to vector store
    print("="*70)
    print("TEST 1: Adding chunks")
    print("="*70 + "\n")

    num_added = agent.add_chunks(test_chunks)
    stats = agent.get_stats()

    print(f"✅ Added {num_added} chunks")
    print(f"   Total documents: {stats['num_documents']}")
    print(f"   Index size: {stats['index_size_mb']} MB\n")

    # Test 2: Search with various queries
    print("="*70)
    print("TEST 2: Searching for relevant chunks")
    print("="*70 + "\n")

    test_queries = [
        "How does collaborative filtering work?",
        "Tell me about vector search with HNSW",
        "What are neural recommendation models?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)

        results = agent.search(query, top_k=3)

        if results:
            print(f"Found {len(results)} results:\n")
            for j, (chunk, score) in enumerate(results, 1):
                print(f"  Result {j}:")
                print(f"  └─ Score: {score:.4f}")
                print(f"  └─ Source: {chunk['metadata']['source_url']}")
                print(f"  └─ Text: {chunk['text'][:100]}...")
                print()
        else:
            print("No results found\n")

    # Test 3: Test persistence (reload agent)
    print("="*70)
    print("TEST 3: Testing persistence")
    print("="*70 + "\n")

    print("Creating new agent instance (should load saved index)...")
    agent2 = RetrievalAgent(index_name="test_agent")
    stats2 = agent2.get_stats()

    print(f"✅ Loaded index")
    print(f"   Documents: {stats2['num_documents']}")
    print(f"   Index size: {stats2['index_size_mb']} MB\n")

    # Verify by searching
    query = "What is FAISS?"
    print(f"Test search on loaded index: '{query}'")
    results = agent2.search(query, top_k=2)

    if results:
        print(f"✅ Search successful! Found {len(results)} results")
        for j, (chunk, score) in enumerate(results, 1):
            print(f"  {j}. Score: {score:.4f} - {chunk['text'][:80]}...")
    else:
        print("❌ No results found")

    # Test 4: Get all chunks (for other agents)
    print("\n" + "="*70)
    print("TEST 4: Get all chunks (for other agents)")
    print("="*70 + "\n")

    all_chunks = agent2.get_all_chunks()
    print(f"✅ Retrieved {len(all_chunks)} total chunks")
    print(f"   First chunk ID: {all_chunks[0]['id']}")
    print(f"   Last chunk ID: {all_chunks[-1]['id']}")

    print("\n" + "="*70)
    print("Retrieval Agent test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_retrieval_agent()