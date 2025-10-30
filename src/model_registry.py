"""
Model Registry - Loads from models.yaml
Manages available LLM models and their metadata.

Last updated: October 2025
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_registry() -> Dict[str, Dict]:
    """
    Load model registry from YAML file.

    Returns:
        Model registry dictionary

    Raises:
        FileNotFoundError: If models.yaml doesn't exist
    """
    yaml_path = Path(__file__).parent.parent / "models.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {yaml_path}\n"
            "Create models.yaml in the project root with model definitions."
        )

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded {len(data['models'])} models from {yaml_path}")
        return data['models']
    except Exception as e:
        logger.error(f"Failed to load models.yaml: {e}")
        raise


# Load registry on import
MODEL_REGISTRY = load_registry()


def get_model_info(model_name: str) -> Optional[Dict]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model info dict or None if not found
    """
    return MODEL_REGISTRY.get(model_name)


def list_available_models(
    provider: Optional[str] = None,
    model_type: Optional[str] = None
) -> List[str]:
    """
    List all available models (enabled ones only).

    Args:
        provider: Optional provider filter (openai, anthropic, google)
        model_type: Optional type filter (text_generation, embedding)

    Returns:
        List of model names
    """
    models = []
    for name, info in MODEL_REGISTRY.items():
        # Skip disabled models
        if not info.get("enabled", True):
            continue
        # Filter by provider if specified
        if provider and info["provider"] != provider:
            continue
        # Filter by type if specified
        if model_type and info.get("model_type") != model_type:
            continue
        models.append(name)

    return models


def get_cheapest_model(
    provider: Optional[str] = None,
    model_type: Optional[str] = None  # ← Change to Optional, default None
) -> str:
    """
    Get the cheapest available model.

    Args:
        provider: Optional provider filter
        model_type: Optional type filter (text_generation, reasoning, embedding)
                   If None, considers all generative models

    Returns:
        Model name
    """
    # If no model_type specified, get all generative models (text_generation + reasoning)
    if model_type is None:
        available = list_available_models(provider=provider)
        # Filter to only generative models (exclude embeddings)
        available = [
            m for m in available 
            if MODEL_REGISTRY[m].get("model_type") in ["text_generation", "reasoning"]
        ]
    else:
        available = list_available_models(provider=provider, model_type=model_type)

    if not available:
        raise ValueError(f"No available models for provider={provider}, type={model_type}")

    if model_type == "embedding":
        # For embeddings, just compare single cost
        cheapest = min(available, key=lambda m: MODEL_REGISTRY[m]["cost_per_1m_tokens"])
    else:
        # For text generation/reasoning, sum input + output
        cheapest = min(
            available,
            key=lambda m: (
                MODEL_REGISTRY[m]["cost_per_1m_input"] + 
                MODEL_REGISTRY[m]["cost_per_1m_output"]
            )
        )

    logger.debug(f"Cheapest {model_type or 'generative'} model: {cheapest}")
    return cheapest


def get_fastest_model(
    provider: Optional[str] = None,
    model_type: Optional[str] = None
) -> str:
    """
    Get the fastest available model (cheapest among fast models).

    Args:
        provider: Optional provider filter
        model_type: Optional type filter

    Returns:
        Model name
    """
    # If no model_type specified, get all generative models
    if model_type is None:
        available = list_available_models(provider=provider)
        available = [
            m for m in available 
            if MODEL_REGISTRY[m].get("model_type") in ["text_generation", "reasoning"]
        ]
    else:
        available = list_available_models(provider=provider, model_type=model_type)

    if not available:
        raise ValueError(f"No available models for provider={provider}, type={model_type}")

    # Filter for fast models
    fast = [m for m in available if MODEL_REGISTRY[m]["speed_category"] == "fast"]

    if not fast:
        # Fallback to medium
        fast = [m for m in available if MODEL_REGISTRY[m]["speed_category"] == "medium"]

    if not fast:
        fast = available

    # Among fast models, pick the cheapest
    if model_type == "embedding":
        result = min(fast, key=lambda m: MODEL_REGISTRY[m]["cost_per_1m_tokens"])
    else:
        result = min(
            fast,
            key=lambda m: (
                MODEL_REGISTRY[m]["cost_per_1m_input"] + 
                MODEL_REGISTRY[m]["cost_per_1m_output"]
            )
        )

    logger.debug(f"Fastest {model_type or 'generative'} model: {result}")
    return result


def get_best_quality_model(
    provider: Optional[str] = None,
    model_type: Optional[str] = None  # ← Change to Optional
) -> str:
    """
    Get the highest quality model (based on cost as proxy).

    Args:
        provider: Optional provider filter
        model_type: Optional type filter

    Returns:
        Model name
    """
    # If no model_type specified, get all generative models
    if model_type is None:
        available = list_available_models(provider=provider)
        available = [
            m for m in available 
            if MODEL_REGISTRY[m].get("model_type") in ["text_generation", "reasoning"]
        ]
    else:
        available = list_available_models(provider=provider, model_type=model_type)

    if not available:
        raise ValueError(f"No available models for provider={provider}, type={model_type}")

    if model_type == "embedding":
        # For embeddings, higher cost = better quality
        best = max(available, key=lambda m: MODEL_REGISTRY[m]["cost_per_1m_tokens"])
    else:
        # For text generation/reasoning, sum input + output
        best = max(
            available,
            key=lambda m: (
                MODEL_REGISTRY[m]["cost_per_1m_input"] + 
                MODEL_REGISTRY[m]["cost_per_1m_output"]
            )
        )

    logger.debug(f"Best quality {model_type or 'generative'} model: {best}")
    return best


def get_embedding_model(strategy: str = "cost_optimized") -> str:
    """
    Get the best embedding model based on strategy.

    Args:
        strategy: Routing strategy (cost_optimized, quality_optimized, balanced)

    Returns:
        Embedding model name
    """
    if strategy == "cost_optimized":
        return get_cheapest_model(model_type="embedding")
    elif strategy == "quality_optimized":
        return get_best_quality_model(model_type="embedding")
    else:  # balanced
        return "text-embedding-3-small"


def print_registry():
    """Print the model registry in a readable format."""
    print("\n" + "="*80)
    print("MODEL REGISTRY")
    print("="*80 + "\n")

    # Group by provider
    providers = {}
    for name, info in MODEL_REGISTRY.items():
        provider = info["provider"]
        if provider not in providers:
            providers[provider] = {
                "text_generation": [],
                "reasoning": [],
                "embedding": []
            }

        model_type = info.get("model_type", "text_generation")

        # Ensure the model_type key exists
        if model_type not in providers[provider]:
            providers[provider][model_type] = []

        providers[provider][model_type].append((name, info))

    for provider, types in providers.items():
        print(f"🔹 {provider.upper()}")

        # Text generation models
        if types.get("text_generation"):
            print("   Text Generation:")
            for name, info in types["text_generation"]:
                enabled = "✅" if info.get("enabled", True) else "❌"
                cost = info["cost_per_1m_input"] + info["cost_per_1m_output"]
                print(f"      {enabled} {name}")
                print(f"         ${cost:.2f}/1M tokens | {info['speed_category']} | {info['max_tokens']:,} ctx")
                print(f"         {info['description']}")

        # Reasoning models
        if types.get("reasoning"):
            print("   Reasoning:")
            for name, info in types["reasoning"]:
                enabled = "✅" if info.get("enabled", True) else "❌"
                cost = info["cost_per_1m_input"] + info["cost_per_1m_output"]
                print(f"      {enabled} {name}")
                print(f"         ${cost:.2f}/1M tokens | {info['speed_category']} | {info['max_tokens']:,} ctx")
                print(f"         {info['description']}")

        # Embedding models
        if types.get("embedding"):
            print("   Embeddings:")
            for name, info in types["embedding"]:
                enabled = "✅" if info.get("enabled", True) else "❌"
                print(f"      {enabled} {name}")
                print(f"         ${info['cost_per_1m_tokens']:.2f}/1M tokens | {info['embedding_dim']}d")
                print(f"         {info['description']}")

        print()


if __name__ == "__main__":
    print_registry()

    print("="*80)
    print("AVAILABLE MODELS")
    print("="*80 + "\n")

    text_models = list_available_models(model_type="text_generation")
    reasoning_models = list_available_models(model_type="reasoning")
    embedding_models = list_available_models(model_type="embedding")

    print(f"Text Generation ({len(text_models)}): {', '.join(text_models)}")
    print(f"Reasoning ({len(reasoning_models)}): {', '.join(reasoning_models)}")
    print(f"Embeddings ({len(embedding_models)}): {', '.join(embedding_models)}\n")

    print("="*80)
    print("MODEL SELECTION")
    print("="*80 + "\n")

    print("All Generative Models:")
    print(f"   💰 Cheapest: {get_cheapest_model()}")
    print(f"   ⚡ Fastest: {get_fastest_model()}")
    print(f"   🏆 Best quality: {get_best_quality_model()}\n")

    print("Text Generation Only:")
    print(f"   💰 Cheapest: {get_cheapest_model(model_type='text_generation')}")
    print(f"   ⚡ Fastest: {get_fastest_model(model_type='text_generation')}")
    print(f"   🏆 Best quality: {get_best_quality_model(model_type='text_generation')}\n")

    print("Embeddings:")
    print(f"   💰 Cheapest: {get_embedding_model('cost_optimized')}")
    print(f"   🏆 Best quality: {get_embedding_model('quality_optimized')}")