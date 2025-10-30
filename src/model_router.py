"""
Model Router - Intelligently routes requests to the best LLM model.
Handles complexity detection, model selection, and API calls.
"""

import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    ROUTING_STRATEGY,
    PROVIDERS_AVAILABLE,
    LLM_TEMPERATURE,
    MAX_TOKENS
)
from src.model_registry import (
    MODEL_REGISTRY,
    get_cheapest_model,
    get_fastest_model,
    get_best_quality_model,
    get_embedding_model,
    get_model_info
)

logger = logging.getLogger(__name__)

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================================
# Complexity Detection
# ============================================================================

def detect_complexity(
    task_type: str,
    messages: List[Dict],
    method: str = "rule_based"
) -> str:
    """
    Automatically detect task complexity.

    Args:
        task_type: Type of task (query_analysis, synthesis, etc.)
        messages: Message list for the task
        method: Detection method (rule_based, llm_based, hybrid)

    Returns:
        Complexity level: "simple" | "medium" | "complex"
    """
    if method == "rule_based":
        return _detect_complexity_rule_based(task_type, messages)
    elif method == "llm_based":
        return _detect_complexity_llm_based(task_type, messages)
    elif method == "hybrid":
        # Try rule-based first, use LLM for borderline cases
        rule_result = _detect_complexity_rule_based(task_type, messages)
        if rule_result == "medium":  # Borderline case
            return _detect_complexity_llm_based(task_type, messages)
        return rule_result
    else:
        logger.warning(f"Unknown complexity detection method: {method}, using rule_based")
        return _detect_complexity_rule_based(task_type, messages)


def _detect_complexity_rule_based(task_type: str, messages: List[Dict]) -> str:
    """
    Rule-based complexity detection using heuristics.

    Args:
        task_type: Type of task
        messages: Message list

    Returns:
        Complexity level
    """
    # Extract user message content
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content += msg.get("content", "")

    # Calculate metrics
    length = len(user_content)
    word_count = len(user_content.split())
    sentence_count = user_content.count('.') + user_content.count('?') + user_content.count('!')

    # Technical keywords that indicate complexity
    technical_keywords = [
        'algorithm', 'architecture', 'system', 'implementation', 'optimization',
        'performance', 'scalability', 'database', 'framework', 'analyze',
        'compare', 'evaluate', 'explain', 'design', 'reasoning'
    ]
    technical_count = sum(1 for keyword in technical_keywords if keyword in user_content.lower())

    logger.debug(f"Complexity metrics - Length: {length}, Words: {word_count}, "
                f"Sentences: {sentence_count}, Technical: {technical_count}")

    # Task-specific rules
    if task_type == "query_analysis":
        if length < 50 and word_count < 10:
            return "simple"
        elif length > 200 or technical_count > 3:
            return "complex"
        else:
            return "medium"

    elif task_type == "synthesis":
        # Synthesis is usually complex
        return "complex"

    elif task_type == "fact_checking":
        # Fact checking needs accuracy
        return "complex"

    elif task_type in ["web_search", "document_processing"]:
        if length < 100:
            return "simple"
        elif length > 300:
            return "complex"
        else:
            return "medium"

    # Default rules
    if length < 50 and sentence_count <= 1:
        return "simple"
    elif length > 200 or sentence_count > 5 or technical_count > 2:
        return "complex"
    else:
        return "medium"


def _detect_complexity_llm_based(task_type: str, messages: List[Dict]) -> str:
    """
    LLM-based complexity detection (uses cheap model).

    Args:
        task_type: Type of task
        messages: Message list

    Returns:
        Complexity level
    """
    if not openai_client:
        logger.warning("OpenAI client not available, falling back to rule-based")
        return _detect_complexity_rule_based(task_type, messages)

    try:
        # Use cheapest model for classification
        cheap_model = get_cheapest_model(provider="openai")

        prompt = f"""Analyze this {task_type} task and classify its complexity.

Task content: {messages[-1].get('content', '')}

Classify as:
- "simple": Straightforward, single concept, basic question
- "medium": Multi-part or requires some reasoning
- "complex": Technical, multi-faceted, needs deep analysis

Respond with ONLY ONE WORD: simple, medium, or complex"""

        response = openai_client.chat.completions.create(
            model=cheap_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        complexity = response.choices[0].message.content.strip().lower() #type:ignore

        if complexity in ["simple", "medium", "complex"]:
            logger.debug(f"LLM detected complexity: {complexity}")
            return complexity
        else:
            logger.warning(f"Invalid LLM complexity response: {complexity}")
            return "medium"

    except Exception as e:
        logger.error(f"LLM complexity detection failed: {e}")
        return _detect_complexity_rule_based(task_type, messages)


# ============================================================================
# Model Selection
# ============================================================================

def select_model(
    task_type: str,
    complexity: str,
    strategy: str = ROUTING_STRATEGY,
    provider: Optional[str] = None
) -> str:
    """
    Select the best model based on task, complexity, and strategy.

    Args:
        task_type: Type of task
        complexity: Complexity level (simple, medium, complex)
        strategy: Routing strategy (cost_optimized, quality_optimized, balanced, latency_optimized)
        provider: Optional provider preference

    Returns:
        Model name
    """
    logger.debug(f"Selecting model - Task: {task_type}, Complexity: {complexity}, Strategy: {strategy}")

    # Strategy-based selection
    if strategy == "cost_optimized":
        # Always use cheapest unless complex task requires quality
        if complexity == "complex" and task_type in ["synthesis", "fact_checking"]:
            model = get_best_quality_model(provider=provider)
        else:
            model = get_cheapest_model(provider=provider)

    elif strategy == "quality_optimized":
        # Always use best quality
        model = get_best_quality_model(provider=provider)

    elif strategy == "latency_optimized":
        # Always use fastest
        model = get_fastest_model(provider=provider)

    elif strategy == "balanced":
        # Smart routing based on task and complexity
        if complexity == "simple":
            model = get_cheapest_model(provider=provider)
        elif complexity == "complex":
            model = get_best_quality_model(provider=provider)
        else:  # medium
            # Use mid-tier model (gpt-5-mini)
            available = [m for m in MODEL_REGISTRY.keys() 
                        if MODEL_REGISTRY[m].get("enabled", True)
                        and MODEL_REGISTRY[m].get("model_type") == "text_generation"]
            if provider:
                available = [m for m in available if MODEL_REGISTRY[m]["provider"] == provider]

            # Try to find gpt-5-mini or equivalent
            if "gpt-5-mini" in available:
                model = "gpt-5-mini"
            else:
                model = get_cheapest_model(provider=provider)

    else:
        logger.warning(f"Unknown strategy: {strategy}, using balanced")
        model = get_cheapest_model(provider=provider)

    logger.info(f"Selected model: {model} (task={task_type}, complexity={complexity}, strategy={strategy})")
    return model


# ============================================================================
# API Calling
# ============================================================================

def call_openai(
    model: str,
    messages: List[Dict],
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    **kwargs
) -> Any:
    """
    Call OpenAI API.

    Args:
        model: Model name
        messages: Message list
        temperature: Temperature setting
        max_tokens: Max tokens to generate
        **kwargs: Additional parameters

    Returns:
        API response
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized (missing API key)")

    logger.debug(f"Calling OpenAI model: {model}")

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages, #type:ignore
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ) #type:ignore

        # Log token usage and cost
        usage = response.usage
        model_info = get_model_info(model)

        if model_info and usage:
            input_cost = (usage.prompt_tokens / 1_000_000) * model_info["cost_per_1m_input"]
            output_cost = (usage.completion_tokens / 1_000_000) * model_info["cost_per_1m_output"]
            total_cost = input_cost + output_cost

            logger.info(f"OpenAI API call - Model: {model}, "
                       f"Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out, "
                       f"Cost: ${total_cost:.6f}")

        return response

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def call_anthropic(model: str, messages: List[Dict], **kwargs) -> Any:
    """
    Call Anthropic (Claude) API.

    Args:
        model: Model name
        messages: Message list
        **kwargs: Additional parameters

    Returns:
        API response
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("Anthropic API key not configured")

    logger.warning("Anthropic API not yet implemented")
    raise NotImplementedError("Anthropic support coming soon")


def call_google(model: str, messages: List[Dict], **kwargs) -> Any:
    """
    Call Google (Gemini) API.

    Args:
        model: Model name
        messages: Message list
        **kwargs: Additional parameters

    Returns:
        API response
    """
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not configured")

    logger.warning("Google API not yet implemented")
    raise NotImplementedError("Google support coming soon")


# ============================================================================
# Main Routing Function
# ============================================================================

def route_request(
    task_type: str,
    messages: List[Dict],
    complexity: Optional[str] = None,
    auto_detect_complexity: bool = True,
    strategy: str = ROUTING_STRATEGY,
    provider: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Route a request to the best model.

    Args:
        task_type: Type of task (query_analysis, synthesis, etc.)
        messages: Message list
        complexity: Optional manual complexity override
        auto_detect_complexity: Whether to auto-detect complexity
        strategy: Routing strategy
        provider: Optional provider preference
        **kwargs: Additional parameters for API call

    Returns:
        API response
    """
    logger.info(f"Routing request - Task: {task_type}")

    # Detect complexity if not provided
    if complexity is None and auto_detect_complexity:
        complexity = detect_complexity(task_type, messages)
    elif complexity is None:
        complexity = "medium"  # Default

    # Select model
    model = select_model(task_type, complexity, strategy, provider)

    # Get model info
    model_info = get_model_info(model)
    if not model_info:
        raise ValueError(f"Model not found in registry: {model}")

    # Route to appropriate provider
    provider_name = model_info["provider"]

    if provider_name == "openai":
        return call_openai(model, messages, **kwargs)
    elif provider_name == "anthropic":
        return call_anthropic(model, messages, **kwargs)
    elif provider_name == "google":
        return call_google(model, messages, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


# ============================================================================
# Embedding Routing
# ============================================================================

def route_embedding_request(
    texts: List[str],
    strategy: str = ROUTING_STRATEGY
) -> List[List[float]]:
    """
    Route embedding request to the best embedding model.

    Args:
        texts: List of texts to embed
        strategy: Routing strategy

    Returns:
        List of embeddings
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized (missing API key)")

    # Select embedding model
    model = get_embedding_model(strategy)

    logger.info(f"Generating embeddings with {model} for {len(texts)} texts")

    try:
        response = openai_client.embeddings.create(
            model=model,
            input=texts
        )

        # Log usage and cost
        model_info = get_model_info(model)
        if model_info:
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1_000_000) * model_info["cost_per_1m_tokens"]
            logger.info(f"Embedding generated - Model: {model}, Tokens: {total_tokens}, Cost: ${cost:.6f}")

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


# ============================================================================
# Testing
# ============================================================================

def test_router():
    """Test the model router with sample requests."""
    print("\n" + "="*70)
    print("TESTING MODEL ROUTER")
    print("="*70 + "\n")

    # Test complexity detection
    print("1. Testing Complexity Detection\n")

    test_cases = [
        {
            "task": "query_analysis",
            "messages": [{"role": "user", "content": "What is AI?"}]
        },
        {
            "task": "query_analysis",
            "messages": [{"role": "user", "content": "Explain the differences between supervised and unsupervised learning, including their use cases, advantages, and limitations."}]
        },
        {
            "task": "synthesis",
            "messages": [{"role": "user", "content": "Write a comprehensive report."}]
        }
    ]

    for case in test_cases:
        complexity = detect_complexity(case["task"], case["messages"])
        print(f"Task: {case['task']}")
        print(f"Query: {case['messages'][0]['content'][:60]}...")
        print(f"Detected Complexity: {complexity}\n")

    # Test model selection
    print("\n2. Testing Model Selection\n")

    strategies = ["cost_optimized", "balanced", "quality_optimized"]
    complexities = ["simple", "medium", "complex"]

    for strategy in strategies:
        print(f"Strategy: {strategy}")
        for complexity in complexities:
            model = select_model("query_analysis", complexity, strategy)
            print(f"   {complexity}: {model}")
        print()

    print("="*70)
    print("Router test complete!")
    print("="*70)


if __name__ == "__main__":
    test_router()