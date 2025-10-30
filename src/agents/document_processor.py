"""
Document Processor Agent
Processes and chunks web content for vector storage.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger("agents.document_processor")


class DocumentProcessorAgent:
    """
    Agent that processes raw web content into clean, chunked documents.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the Document Processor Agent.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document Processor Agent initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def process(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process search results into chunked documents.

        Args:
            search_results: Results from Web Searcher Agent (with full_content)

        Returns:
            List of processed document chunks with metadata
        """
        logger.info(f"Processing {len(search_results)} search result sets")

        all_chunks = []

        for result_set in search_results:
            question = result_set["question"]

            # Process full content if available
            if "full_content" in result_set and result_set["full_content"]:
                for content in result_set["full_content"]:
                    chunks = self._process_content(
                        content["raw_content"],
                        source_url=content["url"],
                        question=question
                    )
                    all_chunks.extend(chunks)

            # Fallback: process snippets if no full content
            else:
                logger.warning(f"No full content for '{question}', using snippets")
                for result in result_set.get("results", []):
                    chunks = self._process_content(
                        result["snippet"],
                        source_url=result["url"],
                        question=question
                    )
                    all_chunks.extend(chunks)

        logger.info(f"Processing complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _process_content(self, raw_content: str, source_url: str, question: str) -> List[Dict[str, Any]]:
        """
        Process a single piece of raw content.

        Args:
            raw_content: Raw text content
            source_url: Source URL for metadata
            question: Original question for context

        Returns:
            List of chunks with metadata
        """
        # Step 1: Clean the text
        cleaned_text = self._clean_text(raw_content)

        # Step 2: Chunk the text
        chunks = self._chunk_text(cleaned_text)

        # Step 3: Add metadata to each chunk
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            processed_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_url": source_url,
                    "question": question,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        return processed_chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean raw text by removing unwanted characters and formatting.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove HTML tags (in case any leaked through)
        text = re.sub(r'<[^>]+>', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?()\-\'\"]+', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping pieces with sentence-aware splitting.

        Args:
            text: Cleaned text

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            # Calculate the end position
            end = start + self.chunk_size

            # If we exceed text length, just take what's left
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a good sentence boundary
            # Look backwards from end position to find sentence break
            chunk_text = text[start:end]

            # Find the last occurrence of sentence-ending punctuation
            last_period = chunk_text.rfind('. ')
            last_question = chunk_text.rfind('? ')
            last_exclamation = chunk_text.rfind('! ')

            # Get the maximum position (closest to end)
            sentence_end = max(last_period, last_question, last_exclamation)

            # Only use sentence boundary if it's in the latter half of the chunk
            # This prevents tiny chunks if there's a period early on
            if sentence_end > len(chunk_text) // 2:
                # Add 2 to include the punctuation and space
                end = start + sentence_end + 2

            # Extract the chunk
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Calculate next start with overlap
            next_start = end - self.chunk_overlap

            # Make sure next_start doesn't go before current start
            if next_start <= start:
                next_start = start + (self.chunk_size // 2)
            else:
                # Find sentence boundary in the overlap region for clean start
                overlap_text = text[next_start:end]

                # Look for the first sentence start in the overlap
                first_period = overlap_text.find('. ')
                first_question = overlap_text.find('? ')
                first_exclamation = overlap_text.find('! ')

                # Get valid boundaries (exclude -1)
                boundaries = [b for b in [first_period, first_question, first_exclamation] if b != -1]

                if boundaries:
                    # Use the first sentence boundary found
                    first_boundary = min(boundaries)
                    # Move start to after the punctuation and space
                    next_start = next_start + first_boundary + 2

            start = next_start

        return chunks


def test_document_processor():
    """Test the Document Processor Agent."""
    print("\n" + "="*70)
    print("TESTING DOCUMENT PROCESSOR AGENT")
    print("="*70 + "\n")

    agent = DocumentProcessorAgent(chunk_size=800, chunk_overlap=200)

    # Simulate web search results (from Agent #2)
    test_search_results = [
        {
            "question": "What are recommendation systems?",
            "results": [
                {
                    "title": "Recommendation Systems Overview",
                    "url": "https://example.com/recsys",
                    "snippet": "Recommendation systems are algorithms that suggest relevant items to users based on their preferences and behavior.",
                    "score": 0.95
                }
            ],
            "full_content": [
                {
                    "url": "https://example.com/recsys-full",
                    "raw_content": """
                    Recommendation systems are a subclass of information filtering systems that seek to predict 
                    the rating or preference a user would give to an item. They are primarily used in commercial 
                    applications. Recommendation systems are utilized in a variety of areas, with commonly 
                    recognized examples taking the form of playlist generators for video and music services, 
                    product recommenders for online stores, or content recommenders for social media platforms 
                    and open web content recommenders.

                    These systems can operate using a number of different technologies. The most popular 
                    approaches are collaborative filtering, content-based filtering, and hybrid approaches. 
                    Collaborative filtering builds a model from a user's past behavior as well as similar 
                    decisions made by other users. This model is then used to predict items that the user 
                    may have an interest in.

                    Content-based filtering approaches utilize a series of discrete characteristics of an item 
                    in order to recommend additional items with similar properties. Hybrid approaches combine 
                    collaborative filtering and content-based filtering. Most businesses use hybrid systems to 
                    leverage the strengths of both approaches.
                    """ * 3  # Repeat to simulate longer content
                }
            ]
        },
        {
            "question": "How does collaborative filtering work?",
            "results": [
                {
                    "title": "Collaborative Filtering Guide",
                    "url": "https://example.com/collab",
                    "snippet": "Collaborative filtering is a method of making automatic predictions about user interests.",
                    "score": 0.92
                }
            ],
            "full_content": []  # Test fallback to snippets
        }
    ]

    print("📝 Test Input:")
    print(f"   Search Results: {len(test_search_results)} question sets")
    print(f"   Question 1: {test_search_results[0]['question']}")
    print(f"      - Full content available: Yes ({len(test_search_results[0]['full_content'][0]['raw_content'])} chars)")
    print(f"   Question 2: {test_search_results[1]['question']}")
    print(f"      - Full content available: No (will use snippets)\n")

    print("="*70)
    print("Processing documents...")
    print("="*70 + "\n")

    chunks = agent.process(test_search_results)

    print(f"✅ Processing complete: {len(chunks)} chunks created\n")

    # Show chunk statistics
    print("📊 Chunk Statistics:")
    chunks_by_question = {}
    for chunk in chunks:
        question = chunk["metadata"]["question"]
        if question not in chunks_by_question:
            chunks_by_question[question] = []
        chunks_by_question[question].append(chunk)

    for question, q_chunks in chunks_by_question.items():
        print(f"\n   Question: {question}")
        print(f"   └─ Chunks created: {len(q_chunks)}")
        print(f"   └─ Avg chunk size: {sum(len(c['text']) for c in q_chunks) // len(q_chunks)} chars")

    # Show sample chunks
    print("\n" + "="*70)
    print("Sample Chunks (first 2):")
    print("="*70 + "\n")

    for i, chunk in enumerate(chunks[:2], 1):
        print(f"Chunk {i}:")
        print(f"└─ Text length: {len(chunk['text'])} chars")
        print(f"└─ Source: {chunk['metadata']['source_url']}")
        print(f"└─ Question: {chunk['metadata']['question']}")
        print(f"└─ Chunk {chunk['metadata']['chunk_index'] + 1} of {chunk['metadata']['total_chunks']}")
        print(f"└─ Preview: {chunk['text'][:200]}...\n")

    # Test text cleaning
    print("="*70)
    print("Testing text cleaning:")
    print("="*70 + "\n")

    dirty_text = """
    <html>This   has    excessive    whitespace!
    And HTML tags <div>here</div>.
    Plus a URL: https://example.com/test
    And special chars: @#$%^&*
    """

    cleaned = agent._clean_text(dirty_text)
    print(f"Original: {repr(dirty_text)}")
    print(f"Cleaned:  {repr(cleaned)}\n")

    # Test chunking with VARIED sentences
    print("="*70)
    print("Testing chunking logic with varied content:")
    print("="*70 + "\n")

    # Create realistic varied text
    long_text = """
    Machine learning is a subset of artificial intelligence that focuses on building systems 
    that can learn from data. These systems improve their performance over time without being 
    explicitly programmed. Deep learning is a specialized branch of machine learning that uses 
    neural networks with multiple layers. These deep neural networks can automatically learn 
    hierarchical representations of data. Natural language processing combines machine learning 
    with linguistic knowledge to understand human language. Modern NLP systems can perform tasks 
    like translation, sentiment analysis, and question answering. Computer vision is another 
    important application area that enables machines to interpret visual information. Autonomous 
    vehicles rely heavily on computer vision algorithms to navigate safely. Reinforcement learning 
    is a paradigm where agents learn by interacting with their environment. AlphaGo used reinforcement 
    learning to defeat world champions in the game of Go. Transfer learning allows models trained on 
    one task to be adapted for related tasks. This approach has become increasingly popular in recent 
    years. Large language models like GPT have revolutionized natural language processing. These models 
    are trained on vast amounts of text data from the internet. Recommendation systems help users 
    discover relevant content in information-rich environments. Netflix and Spotify use sophisticated 
    recommendation algorithms to personalize user experiences. Collaborative filtering is one popular 
    approach that leverages user behavior patterns. Content-based filtering considers item characteristics 
    to make recommendations. Hybrid systems combine multiple recommendation strategies for better results.
    """

    chunks_test = agent._chunk_text(long_text)

    print(f"Input text length: {len(long_text)} chars")
    print(f"Chunks created: {len(chunks_test)}")
    print(f"Chunk sizes: {[len(c) for c in chunks_test]}\n")

    # Show first two chunks with overlap detection
    if len(chunks_test) >= 2:
        print("Chunk 1 (first 300 chars):")
        print(f"{chunks_test[0][:300]}...\n")

        print("Chunk 1 (last 150 chars):")
        print(f"...{chunks_test[0][-150:]}\n")

        print("Chunk 2 (first 150 chars):")
        print(f"{chunks_test[1][:150]}...\n")

        # Check for actual overlap content
        chunk1_end = chunks_test[0][-200:]  # Last 200 chars of chunk 1
        chunk2_start = chunks_test[1][:200]  # First 200 chars of chunk 2

        # Find common sentences
        overlap_found = False
        for sentence in chunk1_end.split('. '):
            if sentence and sentence in chunk2_start:
                print(f"✅ OVERLAP DETECTED!")
                print(f"   Overlapping content: '{sentence[:100]}...'")
                overlap_found = True
                break

        if not overlap_found:
            print("❌ No clear overlap detected in this test")

    print("\n" + "="*70)
    print("Document Processor Agent test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_document_processor()