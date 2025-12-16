"""Embedding generation for RAG benchmarking."""
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding generation."""
    num_texts: int
    total_time: float
    avg_time_per_text: float
    model_name: str
    dimension: int
    tokens_processed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_texts': self.num_texts,
            'total_time': self.total_time,
            'avg_time_per_text': self.avg_time_per_text,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'tokens_processed': self.tokens_processed
        }


class EmbeddingGenerator(ABC):
    """Base class for embedding generators."""

    def __init__(self, model_name: str):
        """Initialize embedding generator."""
        self.model_name = model_name
        self.dimension = None

    @abstractmethod
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Array of embeddings
        """
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Text string

        Returns:
            Embedding vector
        """
        pass

    def benchmark_generation(self, texts: List[str], batch_size: int = 32) -> tuple[np.ndarray, EmbeddingMetrics]:
        """
        Generate embeddings and return metrics.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Tuple of (embeddings, metrics)
        """
        start_time = time.time()
        embeddings = self.generate_embeddings(texts, batch_size)
        total_time = time.time() - start_time

        metrics = EmbeddingMetrics(
            num_texts=len(texts),
            total_time=total_time,
            avg_time_per_text=total_time / len(texts),
            model_name=self.model_name,
            dimension=embeddings.shape[1] if len(embeddings) > 0 else 0
        )

        return embeddings, metrics


class OpenAIEmbedding(EmbeddingGenerator):
    """OpenAI embedding generator."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedding generator.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (if None, reads from environment)
        """
        super().__init__(model_name)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

            # Set dimensions based on model
            if "text-embedding-3-small" in model_name:
                self.dimension = 1536
            elif "text-embedding-3-large" in model_name:
                self.dimension = 3072
            elif "text-embedding-ada-002" in model_name:
                self.dimension = 1536
            else:
                self.dimension = 1536  # default

        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)


class SentenceTransformerEmbedding(EmbeddingGenerator):
    """Sentence Transformers embedding generator (local)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers embedding generator.

        Args:
            model_name: Hugging Face model name
        """
        super().__init__(model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype(np.float32)


class RandomEmbedding(EmbeddingGenerator):
    """Random embedding generator for testing."""

    def __init__(self, dimension: int = 384, seed: int = 42):
        """
        Initialize random embedding generator.

        Args:
            dimension: Embedding dimension
            seed: Random seed
        """
        super().__init__(f"random-{dimension}d")
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate random embedding for single text."""
        # Use hash of text as seed for reproducibility
        text_seed = hash(text) % (2**32)
        rng = np.random.RandomState(text_seed)
        embedding = rng.randn(self.dimension).astype(np.float32)
        # Normalize
        return embedding / np.linalg.norm(embedding)

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate random embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))
        return np.array(embeddings, dtype=np.float32)


def get_embedding_generator(
    provider: str,
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator.

    Args:
        provider: Provider name ('openai', 'sentence-transformers', 'random')
        model_name: Model name (provider-specific)
        **kwargs: Additional arguments for the generator

    Returns:
        EmbeddingGenerator instance
    """
    provider = provider.lower()

    if provider == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedding(model_name=model, **kwargs)

    elif provider in ["sentence-transformers", "sentence_transformers", "local"]:
        model = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedding(model_name=model)

    elif provider == "random":
        dimension = kwargs.get('dimension', 384)
        return RandomEmbedding(dimension=dimension)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Commonly used embedding configurations
EMBEDDING_CONFIGS = {
    "openai-small": {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536
    },
    "openai-large": {
        "provider": "openai",
        "model_name": "text-embedding-3-large",
        "dimension": 3072
    },
    "sentence-transformers-small": {
        "provider": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384
    },
    "sentence-transformers-medium": {
        "provider": "sentence-transformers",
        "model_name": "all-mpnet-base-v2",
        "dimension": 768
    },
    "sentence-transformers-large": {
        "provider": "sentence-transformers",
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dimension": 384
    },
    "random-small": {
        "provider": "random",
        "dimension": 384
    },
    "random-medium": {
        "provider": "random",
        "dimension": 768
    }
}
