"""
Query Analyzer Agent
Breaks down complex queries into structured sub-questions for research.
"""

import logging
from typing import List, Dict
from src.model_router import route_request

logger = logging.getLogger("agents.query_analyzer")


class QueryAnalyzerAgent:
    """
    Agent that analyzes user queries and breaks them into research sub-questions.
    """

    def __init__(self):
        """Initialize the Query Analyzer Agent."""
        logger.info("Query Analyzer Agent initialized")

    def analyze(self, query: str) -> Dict:
        """
        Analyze a user query and break it into sub-questions.

        Args:
            query: User's research question

        Returns:
            Dict with:
                - original_query: The original query
                - sub_questions: List of sub-questions to research
                - query_type: Type of query (factual, comparative, analytical, etc.)
                - complexity: Estimated complexity (simple, medium, complex)
        """
        logger.info(f"Analyzing query: {query}")

        # Build prompt for query analysis
        system_prompt = """You are a research assistant that breaks down complex questions into smaller, focused sub-questions.
            
            Your task:
            1. Analyze the user's question
            2. Identify the main topic and key aspects to research
            3. Break it down into 3-7 specific sub-questions that would help answer the main question
            4. Classify the query type
            5. Estimate complexity
            
            Respond in this EXACT format:
            
            QUERY_TYPE: [factual|comparative|analytical|opinion|how-to]
            COMPLEXITY: [simple|medium|complex]
            SUB_QUESTIONS:
            1. [First sub-question]
            2. [Second sub-question]
            3. [Third sub-question]
            ...
            
            Guidelines:
            - Make sub-questions specific and searchable
            - Each sub-question should address one aspect
            - Order them logically (background → details → implications)
            - Avoid yes/no questions
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]

        # Route request through model router
        try:
            response = route_request(
                task_type="query_analysis",
                messages=messages,
                temperature=0.3  # Lower temperature for structured output
            )

            result = response.choices[0].message.content
            logger.debug(f"Query analysis response: {result}")

            # Parse the response
            parsed = self._parse_response(result)
            parsed["original_query"] = query

            logger.info(f"Query analyzed: {parsed['query_type']} query with "
                       f"{len(parsed['sub_questions'])} sub-questions")

            return parsed

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return fallback structure
            return {
                "original_query": query,
                "sub_questions": [query],  # Just use original as fallback
                "query_type": "unknown",
                "complexity": "medium"
            }

    def _parse_response(self, response: str) -> Dict:
        """
        Parse the structured response from the LLM.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary
        """
        lines = response.strip().split('\n')

        result = {
            "query_type": "unknown",
            "complexity": "medium",
            "sub_questions": []
        }

        in_sub_questions = False

        for line in lines:
            line = line.strip()

            if line.startswith("QUERY_TYPE:"):
                result["query_type"] = line.split(":", 1)[1].strip().lower()

            elif line.startswith("COMPLEXITY:"):
                result["complexity"] = line.split(":", 1)[1].strip().lower()

            elif line.startswith("SUB_QUESTIONS:"):
                in_sub_questions = True

            elif in_sub_questions and line:
                # Remove numbering (e.g., "1. ", "2. ")
                if line[0].isdigit() and '. ' in line:
                    question = line.split('. ', 1)[1].strip()
                    result["sub_questions"].append(question)

        return result


def test_query_analyzer():
    """Test the Query Analyzer Agent."""
    print("\n" + "="*70)
    print("TESTING QUERY ANALYZER AGENT")
    print("="*70 + "\n")

    agent = QueryAnalyzerAgent()

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do recommendation systems work at Netflix and Spotify?",
        "Compare the recommendation algorithms used by Meta's Facebook Feed versus TikTok's For You page"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test Query {i}")
        print(f"{'='*70}")
        print(f"Query: {query}\n")

        result = agent.analyze(query)

        print(f"Query Type: {result['query_type']}")
        print(f"Complexity: {result['complexity']}")
        print(f"\nSub-Questions ({len(result['sub_questions'])}):")
        for j, sq in enumerate(result['sub_questions'], 1):
            print(f"  {j}. {sq}")
        print()

    print("="*70)
    print("Query Analyzer test complete!")
    print("="*70)


if __name__ == "__main__":
    test_query_analyzer()