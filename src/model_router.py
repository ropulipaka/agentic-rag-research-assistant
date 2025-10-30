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
    get_model_info,
    list_available_models
)

logger = logging.getLogger(__name__)

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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

        complexity = response.choices[0].message.content.strip().lower()

        if complexity in ["simple", "medium", "complex"]:
            logger.debug(f"LLM detected complexity: {complexity}")
            return complexity
        else:
            logger.warning(f"Invalid LLM complexity response: {complexity}")
            return "medium"

    except Exception as e:
        logger.error(f"LLM complexity detection failed: {e}")
        return _detect_complexity_rule_based(task_type, messages)


def select_model(
    task_type: str,
    complexity: str,
    strategy: str = ROUTING_STRATEGY,
    provider: Optional[str] = None
) -> str:
    """
    Intelligently select the best model based on task, complexity, and strategy.
    Uses a scoring system that balances cost, quality, speed, and task fit.

    Args:
        task_type: Type of task (query_analysis, synthesis, fact_checking, etc.)
        complexity: Complexity level (simple, medium, complex)
        strategy: Routing strategy (cost_optimized, quality_optimized, balanced, latency_optimized)
        provider: Optional provider preference

    Returns:
        Model name
    """
    logger.debug(f"Selecting model - Task: {task_type}, Complexity: {complexity}, Strategy: {strategy}")

    # Get available text generation and reasoning models
    available = list_available_models(provider=provider)
    available = [
        m for m in available 
        if MODEL_REGISTRY[m].get("model_type") in ["text_generation", "reasoning"]
    ]

    if not available:
        raise ValueError(f"No available models for provider: {provider}")

    # Score each model
    scored_models = []

    for model_name in available:
        model_info = MODEL_REGISTRY[model_name]

        # Base metrics
        total_cost = model_info["cost_per_1m_input"] + model_info["cost_per_1m_output"]
        speed_map = {"fast": 3, "medium": 2, "slow": 1}
        speed_score = speed_map.get(model_info["speed_category"], 2)
        strengths = model_info.get("strengths", [])
        model_type = model_info.get("model_type", "text_generation")

        # Initialize scores
        cost_efficiency = 1 / (total_cost + 0.01)  # Lower cost = higher score
        quality_score = total_cost  # Higher cost = proxy for quality
        task_fit = 0
        complexity_fit = 0

        if task_type == "fact_checking":
            # Fact-checking needs accuracy and reasoning
            if "reasoning" in strengths or model_type == "reasoning":
                task_fit += 5  # Strong preference for reasoning models
            if "accuracy" in strengths or "fact_checking" in strengths:
                task_fit += 3
            # Penalize cheap models for fact-checking
            if total_cost < 1.0:
                task_fit -= 2

        elif task_type == "synthesis":
            # Synthesis needs quality and reasoning
            if "reasoning" in strengths or "complex_tasks" in strengths:
                task_fit += 4
            if "accuracy" in strengths:
                task_fit += 2
            # Prefer mid-to-high quality models
            if total_cost > 5.0:
                task_fit += 2
            elif total_cost < 1.0:
                task_fit -= 1

        elif task_type == "query_analysis":
            # Query analysis needs speed and balance
            if "speed" in strengths or "balanced" in strengths:
                task_fit += 3
            if speed_score >= 2:
                task_fit += 2
            # Prefer fast, cheap models
            if total_cost < 2.0 and speed_score == 3:
                task_fit += 2

        elif task_type == "web_search":
            # Web search needs speed
            if "speed" in strengths:
                task_fit += 3
            if speed_score == 3:
                task_fit += 2
            # Cheap and fast is ideal
            if total_cost < 1.0:
                task_fit += 2

        elif task_type == "document_processing":
            # Document processing needs balance
            if "balanced" in strengths:
                task_fit += 3
            if speed_score >= 2:
                task_fit += 1
            # Mid-tier models ideal
            if 1.0 <= total_cost <= 5.0:
                task_fit += 2

        if complexity == "simple":
            # Simple tasks: prefer cheap, fast models
            if "simple_tasks" in strengths:
                complexity_fit += 4
            if total_cost < 1.0:
                complexity_fit += 3
            if speed_score == 3:
                complexity_fit += 2
            # Penalize expensive models for simple tasks
            if total_cost > 5.0:
                complexity_fit -= 3

        elif complexity == "medium":
            # Medium tasks: balanced models
            if "balanced" in strengths:
                complexity_fit += 4
            # Mid-tier cost is ideal
            if 1.0 <= total_cost <= 5.0:
                complexity_fit += 3
            if speed_score >= 2:
                complexity_fit += 1
            # Avoid extremes
            if total_cost < 0.5 or total_cost > 10.0:
                complexity_fit -= 2

        elif complexity == "complex":
            # Complex tasks: quality matters
            if "reasoning" in strengths or "complex_tasks" in strengths:
                complexity_fit += 5
            if "accuracy" in strengths:
                complexity_fit += 3
            # Prefer higher-quality models
            if total_cost > 5.0:
                complexity_fit += 2
            # Don't use cheap models for complex tasks
            if total_cost < 1.0:
                complexity_fit -= 3

        feature_bonus = 0

        # Caching support is valuable for repeated queries
        if "caching" in strengths:
            feature_bonus += 1

        # Multimodal support adds flexibility
        if "multimodal" in strengths:
            feature_bonus += 0.5

        if strategy == "cost_optimized":
            # Minimize cost while maintaining task fit
            final_score = (
                (cost_efficiency * 10) +  # Cost is king
                (task_fit * 2) +          # But task fit matters
                (complexity_fit * 2) +    # And complexity fit
                feature_bonus
            )

        elif strategy == "quality_optimized":
            # Maximize quality regardless of cost
            final_score = (
                (quality_score * 2) +     # Quality first
                (task_fit * 3) +          # Strong task alignment
                (complexity_fit * 3) +    # Strong complexity alignment
                (feature_bonus * 2)
            )

        elif strategy == "latency_optimized":
            # Minimize latency while maintaining quality
            final_score = (
                (speed_score * 10) +      # Speed is critical
                (task_fit * 2) +          # Task fit matters
                (complexity_fit * 1) +    # Complexity fit less important
                feature_bonus
            )

        elif strategy == "balanced":
            # Balance all factors intelligently
            final_score = (
                (cost_efficiency * 3) +   # Cost matters
                (quality_score * 0.5) +   # Quality matters some
                (speed_score * 2) +       # Speed matters
                (task_fit * 4) +          # Task fit very important
                (complexity_fit * 4) +    # Complexity fit very important
                (feature_bonus * 2)
            )

        else:
            # Default to balanced
            final_score = (
                (cost_efficiency * 3) +
                (quality_score * 0.5) +
                (speed_score * 2) +
                (task_fit * 4) +
                (complexity_fit * 4) +
                (feature_bonus * 2)
            )

        scored_models.append({
            "model": model_name,
            "score": final_score,
            "cost": total_cost,
            "speed": model_info["speed_category"],
            "task_fit": task_fit,
            "complexity_fit": complexity_fit
        })

    # Sort by score (highest first)
    scored_models.sort(key=lambda x: x["score"], reverse=True)

    # Get best model
    best_model = scored_models[0]["model"]

    logger.info(f"Selected model: {best_model} (task={task_type}, complexity={complexity}, strategy={strategy})")
    
    # Log top 3 candidates
    top_3 = scored_models[:3]
    for i, m in enumerate(top_3, 1):
        logger.debug(
            f"  Candidate #{i}: {m['model']} - "
            f"score={round(m['score'], 2)}, "
            f"task_fit={m['task_fit']}, "
            f"complexity_fit={m['complexity_fit']}, "
            f"cost=${m['cost']:.2f}"
        )

    return best_model


def call_openai(
    model: str,
    messages: List[Dict],
    temperature: float = LLM_TEMPERATURE,
    max_completion_tokens: int = MAX_TOKENS,
    **kwargs
) -> Any:
    """
    Call OpenAI API.

    Args:
        model: Model name
        messages: Message list
        temperature: Temperature setting
        max_completion_tokens: Max tokens to generate
        **kwargs: Additional parameters

    Returns:
        API response
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized (missing API key)")

    logger.debug(f"Calling OpenAI model: {model}")

    try:
        # GPT-5 models don't support custom temperature
        # They only support temperature=1 (default)
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            **kwargs
        }

        # Only add temperature if NOT a GPT-5 model
        if not model.startswith("gpt-5"):
            params["temperature"] = temperature

        response = openai_client.chat.completions.create(**params)

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