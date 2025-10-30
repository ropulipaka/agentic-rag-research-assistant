"""
Synthesis Agent
Generates comprehensive answers from retrieved context using LLM.
"""

import logging
from typing import List, Dict, Any, Optional

from src.model_router import route_request

logger = logging.getLogger("agents.synthesis_agent")


class SynthesisAgent:
    """
    Agent that synthesizes comprehensive answers from retrieved context.
    Uses intelligent model routing for optimal quality and cost.
    """

    def __init__(self, strategy: str = "balanced"):
        """
        Initialize the Synthesis Agent.

        Args:
            strategy: Routing strategy (cost_optimized, balanced, quality_optimized)
        """
        self.strategy = strategy
        logger.info(f"Synthesis Agent initialized with strategy: {strategy}")

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[tuple],
        search_results: Optional[List[Dict]] = None,
        max_chunks: int = 10,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesize a comprehensive answer from retrieved context.

        Args:
            query: Original user query
            retrieved_chunks: List of (chunk_data, score) tuples from Retrieval Agent
            search_results: Optional search results from Web Searcher
            max_chunks: Maximum number of chunks to include in context
            include_sources: Whether to include source citations

        Returns:
            Dict with 'answer', 'sources', and metadata
        """
        logger.info(f"Synthesizing answer for query: '{query[:50]}...'")
        logger.info(f"Using {len(retrieved_chunks)} retrieved chunks")

        # Step 1: Organize and rank context
        context = self._prepare_context(retrieved_chunks, search_results, max_chunks)

        # Step 2: Build synthesis prompt
        messages = self._build_synthesis_prompt(query, context, include_sources)

        # Step 3: Route to appropriate model
        logger.info(f"Routing synthesis request (strategy={self.strategy})")

        try:
            response = route_request(
                task_type="synthesis",
                messages=messages,
                strategy=self.strategy,
                auto_detect_complexity=True
            )

            answer = response.choices[0].message.content

            logger.info(f"Synthesis complete. Answer length: {len(answer)} chars")

            # Step 4: Extract sources
            sources = self._extract_sources(retrieved_chunks, search_results) if include_sources else []

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "num_chunks_used": len(context["chunks"]),
                    "num_sources": len(sources),
                    "model_used": response.model,
                    "strategy": self.strategy
                }
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def _prepare_context(
        self,
        retrieved_chunks: List[tuple],
        search_results: Optional[List[Dict]],
        max_chunks: int
    ) -> Dict[str, Any]:
        """
        Prepare and organize context for synthesis.

        Args:
            retrieved_chunks: Retrieved chunks with scores
            search_results: Search results from web
            max_chunks: Maximum chunks to include

        Returns:
            Organized context dict
        """
        # Sort chunks by score (descending) and take top max_chunks
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)[:max_chunks]

        # Build context sections
        chunks_text = []
        for i, (chunk_data, score) in enumerate(sorted_chunks, 1):
            text = chunk_data["text"]
            metadata = chunk_data.get("metadata", {})
            source = metadata.get("source_url", "Unknown")

            # Format chunk with metadata
            chunk_text = f"[Source {i}] (Relevance: {score:.3f})\n{text}\nURL: {source}"
            chunks_text.append(chunk_text)

        context = {
            "chunks": chunks_text,
            "num_chunks": len(chunks_text)
        }

        # Add search results if available
        if search_results:
            context["search_results"] = search_results[:5]  # Top 5 search results

        return context

    def _build_synthesis_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        include_sources: bool
    ) -> List[Dict[str, str]]:
        """
        Build the synthesis prompt for the LLM.

        Args:
            query: User query
            context: Organized context
            include_sources: Whether to include citations

        Returns:
            Messages list for LLM
        """
        # System prompt
        system_prompt = """
            You are a research assistant that synthesizes comprehensive, accurate answers from provided context.
            
            Your task:
            1. Read all provided sources carefully
            2. Synthesize information into a clear, well-structured answer
            3. Use specific details and examples from the sources
            4. Cite sources using [Source N] format when referencing specific information
            5. If sources conflict, acknowledge different perspectives
            6. If information is insufficient, state what's missing
            
            Structure your answer:
            - Start with a brief summary/overview (2-3 sentences)
            - Provide detailed explanation in clear sections
            - Use bullet points for lists or multiple items
            - End with key takeaways if appropriate
            
            Be accurate, comprehensive, and helpful.
        """

        # User prompt with context
        chunks_context = "\n\n".join(context["chunks"])

        user_prompt = f"""
            Query: {query}
            
            Here are the relevant sources I found:
            
            {chunks_context}
            
            Based on these sources, provide a comprehensive answer to the query.
        """

        if include_sources:
            user_prompt += "\n\nImportant: Cite sources using [Source N] format when referencing specific information."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def _extract_sources(
        self,
        retrieved_chunks: List[tuple],
        search_results: Optional[List[Dict]]
    ) -> List[Dict[str, str]]:
        """
        Extract unique sources from retrieved chunks and search results.

        Args:
            retrieved_chunks: Retrieved chunks with scores
            search_results: Search results

        Returns:
            List of source dicts with URL, title, relevance
        """
        sources = []
        seen_urls = set()

        # Extract from retrieved chunks
        for i, (chunk_data, score) in enumerate(retrieved_chunks, 1):
            metadata = chunk_data.get("metadata", {})
            url = metadata.get("source_url")

            if url and url not in seen_urls:
                sources.append({
                    "id": i,
                    "url": url,
                    "title": metadata.get("title", "Source"),
                    "relevance": round(score, 3),
                    "type": "retrieved"
                })
                seen_urls.add(url)

        # Add search results if available
        if search_results:
            for result in search_results:
                url = result.get("url")
                if url and url not in seen_urls:
                    sources.append({
                        "id": len(sources) + 1,
                        "url": url,
                        "title": result.get("title", "Search Result"),
                        "relevance": result.get("score", 0.0),
                        "type": "search"
                    })
                    seen_urls.add(url)

        return sources

    def summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Generate a summary of given text.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            style: Summary style (concise, detailed, bullet_points)

        Returns:
            Summary text
        """
        logger.info(f"Generating {style} summary (max {max_length} words)")

        system_prompt = f"""
            You are a summarization expert. Create a {style} summary of the provided text.
            
            Requirements:
            - Maximum {max_length} words
            - Capture key points and main ideas
            - Maintain accuracy
            - Use clear, direct language
        """

        if style == "bullet_points":
            system_prompt += "\n- Format as bullet points (3-5 points)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize this text:\n\n{text}"}
        ]

        try:
            response = route_request(
                task_type="synthesis",
                messages=messages,
                strategy="cost_optimized",  # Use cheap model for summaries
                auto_detect_complexity=False,
                max_completion_tokens=max_length * 2  # Words to tokens ratio ~1:1.5
            )

            summary = response.choices[0].message.content
            logger.info(f"Summary generated: {len(summary)} chars")

            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise


def test_synthesis_agent():
    """Test the Synthesis Agent."""
    print("\n" + "="*70)
    print("TESTING SYNTHESIS AGENT")
    print("="*70 + "\n")

    agent = SynthesisAgent(strategy="balanced")

    # Simulate retrieved chunks from Retrieval Agent
    test_chunks = [
        (
            {
                "text": "Collaborative filtering is a method of making automatic predictions about the interests of a user by collecting preferences from many users. The underlying assumption is that if person A has the same opinion as person B on an issue, A is more likely to have B's opinion on a different issue.",
                "metadata": {
                    "source_url": "https://example.com/collab-filtering",
                    "title": "Collaborative Filtering Explained",
                    "question": "What is collaborative filtering?"
                }
            },
            0.85  # High relevance score
        ),
        (
            {
                "text": "Two-tower models are a popular architecture for recommendation systems. They consist of two neural networks (towers): one encodes user features and one encodes item features. These embeddings are then compared using dot product or cosine similarity to predict user-item interactions.",
                "metadata": {
                    "source_url": "https://example.com/two-tower",
                    "title": "Two-Tower Models in RecSys",
                    "question": "What are two-tower models?"
                }
            },
            0.72  # Medium-high relevance
        ),
        (
            {
                "text": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.",
                "metadata": {
                    "source_url": "https://example.com/faiss",
                    "title": "FAISS Documentation",
                    "question": "What is FAISS?"
                }
            },
            0.45  # Lower relevance
        )
    ]

    # Simulate search results
    search_results = [
        {
            "title": "Introduction to Recommendation Systems",
            "url": "https://example.com/recsys-intro",
            "snippet": "Recommendation systems are a subclass of information filtering systems...",
            "score": 0.92
        }
    ]

    print("📝 Test Query: 'Explain collaborative filtering and two-tower models'\n")
    print(f"Retrieved chunks: {len(test_chunks)}")
    print(f"Search results: {len(search_results)}\n")

    # Test 1: Full synthesis
    print("="*70)
    print("TEST 1: Full Synthesis with Sources")
    print("="*70 + "\n")

    try:
        result = agent.synthesize(
            query="Explain collaborative filtering and two-tower models for recommendation systems",
            retrieved_chunks=test_chunks,
            search_results=search_results,
            max_chunks=10,
            include_sources=True
        )

        print("✅ Synthesis Complete!\n")
        print("📄 Answer:")
        print("-" * 70)
        print(result["answer"])
        print("-" * 70)
        print()

        print(f"📚 Sources ({len(result['sources'])}):")
        for source in result["sources"]:
            print(f"  [{source['id']}] {source['title']}")
            print(f"      URL: {source['url']}")
            print(f"      Relevance: {source['relevance']}")
            print(f"      Type: {source['type']}")
            print()

        print("📊 Metadata:")
        for key, value in result["metadata"].items():
            print(f"   {key}: {value}")
        print()

    except Exception as e:
        print(f"❌ Synthesis failed: {e}\n")

    # Test 2: Summarization
    print("="*70)
    print("TEST 2: Text Summarization")
    print("="*70 + "\n")

    long_text = """
    Recommendation systems are algorithms designed to suggest relevant items to users. 
    They are used in various applications from e-commerce to streaming services. 
    The main approaches include collaborative filtering, which analyzes user behavior patterns, 
    content-based filtering, which examines item characteristics, and hybrid methods that 
    combine both approaches. Modern recommendation systems often use deep learning techniques 
    like neural collaborative filtering and two-tower architectures. These systems face 
    challenges such as cold start problems, scalability issues, and the need to balance 
    exploration and exploitation. Success metrics include precision, recall, and user engagement.
    """

    try:
        summary = agent.summarize(long_text, max_length=50, style="concise")

        print("Original text length:", len(long_text.split()), "words")
        print("\n📝 Summary (concise, max 50 words):")
        print("-" * 70)
        print(summary)
        print("-" * 70)
        print("\nSummary length:", len(summary.split()), "words")
        print()

    except Exception as e:
        print(f"❌ Summarization failed: {e}\n")

    # Test 3: Different strategies
    print("="*70)
    print("TEST 3: Different Routing Strategies")
    print("="*70 + "\n")

    strategies = ["cost_optimized", "balanced", "quality_optimized"]

    for strategy in strategies:
        print(f"Testing strategy: {strategy}")
        agent_test = SynthesisAgent(strategy=strategy)

        try:
            result = agent_test.synthesize(
                query="What is collaborative filtering?",
                retrieved_chunks=test_chunks[:1],  # Just one chunk
                include_sources=False
            )

            model_used = result["metadata"]["model_used"]
            answer_length = len(result["answer"])

            print(f"   Model: {model_used}")
            print(f"   Answer length: {answer_length} chars")
            print()

        except Exception as e:
            print(f"   ❌ Failed: {e}\n")

    print("="*70)
    print("Synthesis Agent test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_synthesis_agent()