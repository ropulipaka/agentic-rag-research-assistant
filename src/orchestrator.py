"""
Intelligent Orchestrator Agent
LLM-powered orchestration that plans and executes research workflows dynamically.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents import (
    QueryAnalyzerAgent,
    WebSearcherAgent,
    DocumentProcessorAgent,
    RetrievalAgent,
    SynthesisAgent
)
from src.model_router import route_request
from src.config import ROUTING_STRATEGY

logger = logging.getLogger("orchestrator")


class OrchestratorAgent:
    """
    Intelligent orchestrator that uses LLM to plan and execute research workflows.
    """

    def __init__(self, global_strategy: str = ROUTING_STRATEGY):
        """
        Initialize the Orchestrator Agent.

        Args:
            global_strategy: Default routing strategy for all agents
        """
        self.global_strategy = global_strategy

        # Initialize all agents
        logger.info("Initializing orchestrator agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.web_searcher = WebSearcherAgent()
        self.doc_processor = DocumentProcessorAgent()
        self.retrieval_agent = RetrievalAgent(index_name="orchestrator_test")
        self.synthesis_agent = SynthesisAgent(strategy=global_strategy)

        # Metrics tracking
        self.metrics = {
            "total_cost": 0.0,
            "total_time": 0.0,
            "agent_costs": {},
            "agent_times": {}
        }

        logger.info(f"Orchestrator initialized with strategy: {global_strategy}")
        logger.info("All agents loaded and ready")

    async def research(
        self,
        query: str,
        user_strategy: Optional[str] = None,
        user_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute intelligent research workflow.

        Args:
            query: User's research query
            user_strategy: Optional strategy override
            user_preferences: Optional user preferences (e.g., max_cost, min_quality)

        Returns:
            Research result with answer, sources, and metrics
        """
        start_time = time.time()
        self._reset_metrics()

        logger.info("="*70)
        logger.info(f"STARTING RESEARCH: '{query[:50]}...'")
        logger.info("="*70)

        try:
            # Step 1: Create intelligent research plan
            plan = await self._create_research_plan(query, user_strategy, user_preferences)

            logger.info(f"\n📋 Research Plan Created:")
            logger.info(f"   Strategy: {plan['strategy']}")
            logger.info(f"   Estimated complexity: {plan['complexity']}")
            logger.info(f"   Steps: {len(plan['steps'])}")

            # Step 2: Execute the plan
            result = await self._execute_research_plan(query, plan)

            # Step 3: Calculate final metrics
            total_time = time.time() - start_time
            self.metrics["total_time"] = total_time

            result["metrics"] = {
                "total_time_seconds": round(total_time, 2),
                "total_cost_usd": round(self.metrics["total_cost"], 4),
                "agent_breakdown": {
                    "costs": self.metrics["agent_costs"],
                    "times": self.metrics["agent_times"]
                },
                "strategy_used": plan['strategy']
            }

            logger.info(f"\n✅ RESEARCH COMPLETE!")
            logger.info(f"   Total Time: {total_time:.2f}s")
            logger.info(f"   Total Cost: ${self.metrics['total_cost']:.4f}")
            logger.info("="*70 + "\n")

            return result

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            raise

    async def _create_research_plan(
        self,
        query: str,
        user_strategy: Optional[str],
        user_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Use LLM to create intelligent research plan.

        Args:
            query: Research query
            user_strategy: Optional strategy override
            user_preferences: Optional preferences

        Returns:
            Research plan dict
        """
        logger.info("\n🧠 Creating intelligent research plan...")

        planning_prompt = f"""
            You are a research planning expert. You MUST respond with ONLY valid JSON.
        
            Query: "{query}"
            
            User Preferences: {user_preferences or "None"}
            
            Analyze the query and determine:
            1. Complexity: simple (single fact), medium (multiple concepts), or complex (multi-faceted)
            2. Strategy: cost_optimized, balanced, quality_optimized, or latency_optimized
            3. Parameters: use_sub_queries (yes/no), num_search_results (5-20), num_retrieval_chunks (5-20), synthesis_depth (brief/detailed/comprehensive)
            
            Respond with ONLY this JSON structure (no other text):
            
            {{
              "complexity": "simple|medium|complex",
              "strategy": "cost_optimized|balanced|quality_optimized|latency_optimized",
              "reasoning": "brief explanation of why these choices",
              "parameters": {{
                "use_sub_queries": true,
                "num_search_results": 10,
                "num_retrieval_chunks": 10,
                "synthesis_depth": "detailed"
              }},
              "estimated_time_seconds": 30,
              "estimated_cost_usd": 0.05
            }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are a research planning expert. Respond with ONLY valid JSON, no additional text."
            },
            {
                "role": "user",
                "content": planning_prompt
            }
        ]

        try:
            # Use cheap model for planning
            response = route_request(
                task_type="query_analysis",
                messages=messages,
                strategy="cost_optimized",
                auto_detect_complexity=False,
                max_completion_tokens=500
            )

            # Track planning cost (estimated)
            planning_cost = 0.0002
            self.metrics["agent_costs"]["planning"] = planning_cost
            self.metrics["total_cost"] += planning_cost

            content = response.choices[0].message.content.strip()

            # Try to extract JSON if there's extra text
            import json
            import re

            # If content has markdown code blocks, extract JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # Parse JSON
            plan = json.loads(content)

            # Apply user strategy override if provided
            if user_strategy:
                plan["strategy"] = user_strategy
                logger.info(f"   Strategy overridden by user: {user_strategy}")

            # Create execution steps
            plan["steps"] = self._create_execution_steps(plan)

            logger.info(f"   Plan created: {plan['complexity']} complexity, {plan['strategy']} strategy")
            logger.info(f"   Reasoning: {plan.get('reasoning', 'N/A')}")

            return plan

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse plan JSON, using defaults: {e}")
            logger.debug(f"LLM response was: {content[:200]}...")
            return self._get_default_plan(query, user_strategy)
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            return self._get_default_plan(query, user_strategy)

    def _create_execution_steps(self, plan: Dict) -> List[Dict]:
        """
        Create ordered execution steps from plan.

        Args:
            plan: Research plan

        Returns:
            List of execution step dicts
        """
        params = plan.get("parameters", {})

        steps = [
            {
                "name": "query_analysis",
                "agent": "QueryAnalyzer",
                "enabled": params.get("use_sub_queries", True),
                "params": {}
            },
            {
                "name": "web_search",
                "agent": "WebSearcher",
                "enabled": True,
                "params": {
                    "max_results": params.get("num_search_results", 10)
                }
            },
            {
                "name": "document_processing",
                "agent": "DocumentProcessor",
                "enabled": True,
                "params": {}
            },
            {
                "name": "retrieval",
                "agent": "RetrievalAgent",
                "enabled": True,
                "params": {
                    "top_k": params.get("num_retrieval_chunks", 10)
                }
            },
            {
                "name": "synthesis",
                "agent": "SynthesisAgent",
                "enabled": True,
                "params": {
                    "max_chunks": params.get("num_retrieval_chunks", 10)
                }
            }
        ]

        return [step for step in steps if step["enabled"]]

    def _get_default_plan(self, query: str, user_strategy: Optional[str]) -> Dict:
        """
        Get default research plan (fallback).

        Args:
            query: Research query
            user_strategy: Optional strategy

        Returns:
            Default plan dict
        """
        query_length = len(query.split())

        # Simple heuristics
        if query_length < 10:
            complexity = "simple"
            num_chunks = 5
        elif query_length < 20:
            complexity = "medium"
            num_chunks = 10
        else:
            complexity = "complex"
            num_chunks = 15

        strategy = user_strategy or self.global_strategy

        plan = {
            "complexity": complexity,
            "strategy": strategy,
            "reasoning": "Default plan based on query length heuristic",
            "parameters": {
                "use_sub_queries": query_length > 15,
                "num_search_results": 10,
                "num_retrieval_chunks": num_chunks,
                "synthesis_depth": "detailed"
            },
            "estimated_time_seconds": 30,
            "estimated_cost_usd": 0.05
        }

        plan["steps"] = self._create_execution_steps(plan)

        return plan

    async def _execute_research_plan(self, query: str, plan: Dict) -> Dict[str, Any]:
        """
        Execute the research plan step by step.

        Args:
            query: Original query
            plan: Research plan

        Returns:
            Research results
        """
        logger.info("\n🚀 Executing research plan...\n")

        results = {
            "query": query,
            "plan": plan
        }

        # Step 1: Query Analysis
        if self._is_step_enabled(plan, "query_analysis"):
            analysis_result = await self._run_agent_step(
                "Query Analysis",
                lambda: self.query_analyzer.analyze(query)
            )
            # analysis_result is a Dict with "sub_questions" key
            results["sub_queries"] = analysis_result.get("sub_questions", [query])
            results["query_analysis"] = analysis_result
            logger.info(f"   Generated {len(results['sub_queries'])} sub-queries")
        else:
            results["sub_queries"] = [query]
            logger.info(f"   Skipping query analysis (using original query)")

        # Step 2: Web Search
        if self._is_step_enabled(plan, "web_search"):
            step_params = self._get_step_params(plan, "web_search")
            max_results = step_params.get("max_results", 10)

            search_results_raw = await self._run_agent_step(
                "Web Search",
                lambda: self.web_searcher.search(
                    results["sub_queries"],
                    max_results=max_results
                )
            )

            # Flatten nested structure for synthesis agent
            # web_searcher returns: [{"question": "...", "results": [...], "full_content": [...]}]
            # synthesis_agent expects: [{"title": "...", "url": "...", "score": ...}]
            flattened_results = []
            for result_set in search_results_raw:
                for result in result_set.get("results", []):
                    flattened_results.append(result)

            results["search_results"] = flattened_results
            results["search_results_raw"] = search_results_raw  # Keep raw for doc processor
            logger.info(f"   Found {len(flattened_results)} total search results")

        # Step 3: Document Processing
        if self._is_step_enabled(plan, "document_processing"):
            # Document processor needs the raw nested structure
            chunks = await self._run_agent_step(
                "Document Processing",
                lambda: self.doc_processor.process(results["search_results_raw"])
            )
            results["chunks"] = chunks
            logger.info(f"   Processed {len(chunks)} chunks")

        # Step 4: Add to Vector Store & Retrieve
        if self._is_step_enabled(plan, "retrieval"):
            step_params = self._get_step_params(plan, "retrieval")
            top_k = step_params.get("top_k", 10)

            # Add chunks to vector store
            await self._run_agent_step(
                "Vector Store Indexing",
                lambda: self.retrieval_agent.add_chunks(results["chunks"])
            )

            # Retrieve relevant chunks
            retrieved_chunks = await self._run_agent_step(
                "Context Retrieval",
                lambda: self.retrieval_agent.search(query, top_k=top_k)
            )
            results["retrieved_chunks"] = retrieved_chunks
            logger.info(f"   Retrieved {len(retrieved_chunks)} relevant chunks")

        # Step 5: Synthesis
        if self._is_step_enabled(plan, "synthesis"):
            step_params = self._get_step_params(plan, "synthesis")

            # SynthesisAgent does NOT accept strategy parameter in synthesize()
            # It uses self.strategy set during __init__
            synthesis_result = await self._run_agent_step(
                "Answer Synthesis",
                lambda: self.synthesis_agent.synthesize(
                    query=query,
                    retrieved_chunks=results["retrieved_chunks"],
                    search_results=results.get("search_results", []),
                    max_chunks=step_params.get("max_chunks", 10),
                    include_sources=True
                )
            )

            results["answer"] = synthesis_result["answer"]
            results["sources"] = synthesis_result["sources"]
            results["synthesis_metadata"] = synthesis_result["metadata"]

            logger.info(f"   Generated answer ({len(synthesis_result['answer'])} chars)")
            logger.info(f"   Cited {len(synthesis_result['sources'])} sources")

        return results

    async def _run_agent_step(self, step_name: str, agent_func) -> Any:
        """
        Run an agent step with timing and cost tracking.

        Args:
            step_name: Name of the step
            agent_func: Function to execute

        Returns:
            Result from agent
        """
        logger.info(f"   ⏳ {step_name}...")
        start_time = time.time()

        try:
            # All agents are sync
            result = agent_func()

            elapsed = time.time() - start_time

            # Track time
            agent_key = step_name.lower().replace(" ", "_")
            self.metrics["agent_times"][agent_key] = round(elapsed, 3)

            # Estimate costs (TODO: Get actual costs from model router)
            estimated_cost = 0.0
            if "query_analysis" in agent_key:
                estimated_cost = 0.0002  # Small LLM call
            elif "synthesis" in agent_key or "answer_synthesis" in agent_key:
                estimated_cost = 0.01    # Large LLM call with context
            elif "vector_store_indexing" in agent_key:
                estimated_cost = 0.0005  # Embeddings cost

            if estimated_cost > 0:
                self.metrics["agent_costs"][agent_key] = estimated_cost
                self.metrics["total_cost"] += estimated_cost

            logger.info(f"   ✅ {step_name} complete ({elapsed:.2f}s, ~${estimated_cost:.4f})")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"   ❌ {step_name} failed after {elapsed:.2f}s: {e}")
            raise

    def _is_step_enabled(self, plan: Dict, step_name: str) -> bool:
        """Check if a step is enabled in the plan."""
        for step in plan.get("steps", []):
            if step["name"] == step_name:
                return step.get("enabled", True)
        return True

    def _get_step_params(self, plan: Dict, step_name: str) -> Dict:
        """Get parameters for a specific step."""
        for step in plan.get("steps", []):
            if step["name"] == step_name:
                return step.get("params", {})
        return {}

    def _reset_metrics(self):
        """Reset metrics tracking."""
        self.metrics = {
            "total_cost": 0.0,
            "total_time": 0.0,
            "agent_costs": {},
            "agent_times": {}
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution metrics.

        Returns:
            Metrics summary dict
        """
        return {
            "total_cost_usd": round(self.metrics["total_cost"], 4),
            "total_time_seconds": round(self.metrics["total_time"], 2),
            "agent_breakdown": {
                "costs": self.metrics["agent_costs"],
                "times": self.metrics["agent_times"]
            }
        }


async def test_orchestrator():
    """Test the Intelligent Orchestrator with a single comprehensive query."""
    print("\n" + "="*70)
    print("TESTING INTELLIGENT ORCHESTRATOR")
    print("="*70 + "\n")

    orchestrator = OrchestratorAgent(global_strategy="balanced")

    # Single comprehensive test query
    test_query = "Explain how two-tower models work in recommendation systems and compare them to collaborative filtering approaches"

    print(f"Query: {test_query}\n")
    print(f"Expected: Medium complexity, balanced strategy\n")
    print("="*70 + "\n")

    try:
        result = await orchestrator.research(
            query=test_query,
            user_strategy=None  # Let orchestrator decide
        )

        print(f"\n{'='*70}")
        print("RESEARCH RESULTS")
        print(f"{'='*70}\n")

        print(f"📋 Research Plan:")
        print(f"   Complexity: {result['plan']['complexity']}")
        print(f"   Strategy Used: {result['plan']['strategy']}")
        print(f"   Reasoning: {result['plan']['reasoning']}")
        print(f"   Parameters:")
        for key, value in result['plan']['parameters'].items():
            print(f"      {key}: {value}")
        print()

        print(f"🔍 Query Analysis:")
        if 'query_analysis' in result:
            qa = result['query_analysis']
            print(f"   Type: {qa.get('query_type', 'N/A')}")
            print(f"   Complexity: {qa.get('complexity', 'N/A')}")
            print(f"   Sub-queries: {len(result['sub_queries'])}")
            for i, sq in enumerate(result['sub_queries'], 1):
                print(f"      {i}. {sq}")
        print()

        print(f"🌐 Search Results: {len(result.get('search_results', []))} found")
        print()

        print(f"📄 Document Processing:")
        print(f"   Chunks created: {len(result.get('chunks', []))}")
        print(f"   Chunks retrieved: {len(result.get('retrieved_chunks', []))}")
        print()

        print(f"📝 Answer Preview (first 500 chars):")
        print("-" * 70)
        print(result['answer'][:500] + "...")
        print("-" * 70)
        print()

        print(f"📚 Sources Cited: {len(result['sources'])}")
        for i, source in enumerate(result['sources'][:5], 1):
            print(f"   [{i}] {source['title']}")
            print(f"       URL: {source['url']}")
            print(f"       Relevance: {source['relevance']}")
        if len(result['sources']) > 5:
            print(f"   ... and {len(result['sources']) - 5} more sources")
        print()

        print(f"📊 Performance Metrics:")
        metrics = result['metrics']
        print(f"   Total Time: {metrics['total_time_seconds']}s")
        print(f"   Total Cost: ${metrics['total_cost_usd']}")
        print(f"\n   Agent Breakdown:")
        for agent, time_val in metrics['agent_breakdown']['times'].items():
            cost_val = metrics['agent_breakdown']['costs'].get(agent, 0.0)
            print(f"      {agent:.<30} {time_val:>6.2f}s  ${cost_val:>8.4f}")
        print()

        print(f"✅ Full research pipeline executed successfully!")
        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"❌ Research failed: {e}\n")
        import traceback
        traceback.print_exc()

    print("="*70)
    print("Orchestrator test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())