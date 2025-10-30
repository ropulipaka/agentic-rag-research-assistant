"""
Web Searcher Agent
Searches the web for information to answer sub-questions.
"""

import logging
import os
from typing import List, Dict, Any
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agents.web_searcher")


class WebSearcherAgent:
    """
    Agent that searches the web and extracts relevant information.
    """

    def __init__(self):
        """Initialize the Web Searcher Agent."""
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        logger.info("Web Searcher Agent initialized")

    def search(self, sub_questions: List[str], extract_full: bool = True, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for each sub-question.

        Args:
            sub_questions: List of questions to search for
            extract_full: Whether to extract full content from top URLs
            max_results: Number of results per question

        Returns:
            List of search results with snippets and optionally full content
        """
        logger.info(f"Searching web for {len(sub_questions)} questions")

        # Step 1: Search for snippets
        search_results = self._search_web(sub_questions, max_results)

        # Step 2: Extract full content from top URLs (optional)
        if extract_full:
            final_results = self._extract_full_content(search_results)
        else:
            final_results = search_results

        logger.info(f"Search completed: {len(final_results)} question sets processed")

        return final_results

    def _search_web(self, sub_questions: List[str], max_results: int) -> List[Dict[str, Any]]:
        """
        Search the web for each sub-question using Tavily.

        Args:
            sub_questions: List of questions to search for
            max_results: Number of results per question

        Returns:
            List of search results with snippets
        """
        all_results = []

        for question in sub_questions:
            try:
                # Search using Tavily
                response = self.tavily_client.search(
                    query=question,
                    max_results=max_results,
                    search_depth="advanced"  # Gets more detailed results
                )

                logger.debug(f"Found {len(response.get('results', []))} results for: {question}")

                # Structure the results
                search_result = {
                    "question": question,
                    "results": []
                }

                for result in response.get("results", []):
                    search_result["results"].append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", ""),  # Tavily calls it 'content'
                        "score": result.get("score", 0.0)  # Relevance score
                    })

                all_results.append(search_result)

            except Exception as e:
                logger.error(f"Error searching '{question}': {e}")
                # Continue with other questions even if one fails
                continue

        return all_results

    def _extract_full_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract full content from top URLs for deeper analysis.

        Args:
            search_results: Results from _search_web()

        Returns:
            Enriched results with full content
        """
        logger.info("Extracting full content from top URLs")

        enriched_results = []

        for result_set in search_results:
            question = result_set["question"]
            results = result_set["results"]

            # Get top 3 most relevant URLs
            top_urls = [r["url"] for r in sorted(results, key=lambda x: x["score"], reverse=True)[:3]]

            try:
                # Extract full content from these URLs
                extracted = self.tavily_client.extract(urls=top_urls)

                logger.debug(f"Extracted {len(extracted.get('results', []))} pages for: {question}")

                # Merge extracted content with original results
                enriched_set = {
                    "question": question,
                    "results": results,  # Keep original snippets
                    "full_content": []
                }

                for content in extracted.get("results", []):
                    enriched_set["full_content"].append({
                        "url": content.get("url", ""),
                        "raw_content": content.get("raw_content", "")
                    })

                enriched_results.append(enriched_set)

            except Exception as e:
                logger.error(f"Error extracting content: {e}")
                # Fall back to just using snippets
                enriched_results.append({
                    "question": question,
                    "results": results,
                    "full_content": []
                })

        return enriched_results


def test_web_searcher():
    """Test the Web Searcher Agent."""
    print("\n" + "="*70)
    print("TESTING WEB SEARCHER AGENT")
    print("="*70 + "\n")

    agent = WebSearcherAgent()

    # Test sub-questions (from Query Analyzer output)
    test_sub_questions = [
        "What are recommendation systems?",
        "How does collaborative filtering work?"
    ]

    print(f"📝 Test Sub-Questions ({len(test_sub_questions)}):")
    for i, q in enumerate(test_sub_questions, 1):
        print(f"   {i}. {q}")

    print("\n" + "="*70)
    print("TEST 1: Search with snippets only")
    print("="*70 + "\n")

    results_snippets = agent.search(test_sub_questions, extract_full=False)

    print(f"✅ Got results for {len(results_snippets)} questions\n")

    for result_set in results_snippets:
        print(f"📊 Question: {result_set['question']}")
        print(f"   Results Found: {len(result_set['results'])}\n")

        for i, result in enumerate(result_set['results'][:3], 1):  # Show top 3
            print(f"   Result {i}:")
            print(f"   └─ Title: {result['title']}")
            print(f"   └─ URL: {result['url']}")
            print(f"   └─ Score: {result['score']:.3f}")
            print(f"   └─ Snippet: {result['snippet'][:200]}...\n")

    print("\n" + "="*70)
    print("TEST 2: Search with full content extraction")
    print("="*70 + "\n")

    # Use just one question for full extraction test (to save API calls)
    single_question = ["What is FAISS vector database?"]

    results_full = agent.search(single_question, extract_full=True)

    print(f"✅ Got enriched results for {len(results_full)} question(s)\n")

    for result_set in results_full:
        print(f"📊 Question: {result_set['question']}")
        print(f"   Snippets: {len(result_set['results'])} results")
        print(f"   Full Content: {len(result_set['full_content'])} pages extracted\n")

        # Show snippet results
        print("   Top Results (Snippets):")
        for i, result in enumerate(result_set['results'][:2], 1):
            print(f"   {i}. {result['title']}")
            print(f"      Score: {result['score']:.3f}")
            print(f"      Snippet: {result['snippet'][:150]}...\n")

        # Show full content
        if result_set['full_content']:
            print("   Full Content Extracted:")
            for i, content in enumerate(result_set['full_content'], 1):
                print(f"   {i}. URL: {content['url']}")
                print(f"      Content Length: {len(content['raw_content'])} chars")
                print(f"      Preview: {content['raw_content'][:300]}...\n")

    print("="*70)
    print("Web Searcher Agent test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_web_searcher()