"""Base benchmark class for vector databases."""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    database: str
    vector_dimension: int
    num_vectors: int
    index_time: float
    query_time_avg: float
    query_time_p50: float
    query_time_p95: float
    query_time_p99: float
    recall_at_10: float
    queries_per_second: float
    memory_usage_mb: float = 0.0
    index_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseBenchmark(ABC):
    """Abstract base class for vector database benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark.

        Args:
            config: Configuration dictionary with database-specific settings
        """
        self.config = config
        self.vectors = None
        self.query_vectors = None
        self.ground_truth = None

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def create_index(self, index_type: str = "ivfflat") -> None:
        """
        Create vector index.

        Args:
            index_type: Type of index to create (e.g., 'ivfflat', 'hnsw')
        """
        pass

    @abstractmethod
    def insert_vectors(self, vectors: np.ndarray, batch_size: int = 1000) -> float:
        """
        Insert vectors into the database.

        Args:
            vectors: Array of vectors to insert
            batch_size: Number of vectors per batch

        Returns:
            Time taken to insert all vectors (seconds)
        """
        pass

    @abstractmethod
    def query_vectors(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[List[List[int]], List[float]]:
        """
        Query the database for nearest neighbors.

        Args:
            query_vectors: Vectors to query
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (results, query_times) where results is list of neighbor IDs
            and query_times is list of query execution times
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (drop tables, etc.)."""
        pass

    def generate_random_vectors(self, num_vectors: int, dimension: int, seed: int = 42) -> np.ndarray:
        """
        Generate random normalized vectors.

        Args:
            num_vectors: Number of vectors to generate
            dimension: Dimension of each vector
            seed: Random seed for reproducibility

        Returns:
            Array of normalized random vectors
        """
        np.random.seed(seed)
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors

    def compute_ground_truth(self, vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> List[List[int]]:
        """
        Compute ground truth nearest neighbors using brute force.

        Args:
            vectors: Database vectors
            query_vectors: Query vectors
            k: Number of neighbors

        Returns:
            List of lists containing indices of k nearest neighbors for each query
        """
        ground_truth = []
        for query in query_vectors:
            # Compute cosine similarity
            similarities = np.dot(vectors, query)
            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1].tolist()
            ground_truth.append(top_k_indices)
        return ground_truth

    def calculate_recall(self, results: List[List[int]], ground_truth: List[List[int]], k: int = 10) -> float:
        """
        Calculate recall@k.

        Args:
            results: Retrieved nearest neighbors
            ground_truth: True nearest neighbors
            k: Number of neighbors to consider

        Returns:
            Average recall across all queries
        """
        recalls = []
        for result, truth in zip(results, ground_truth):
            result_set = set(result[:k])
            truth_set = set(truth[:k])
            recall = len(result_set & truth_set) / k
            recalls.append(recall)
        return np.mean(recalls)

    def run_benchmark(self, num_vectors: int, dimension: int, num_queries: int = 100,
                     k: int = 10, index_type: str = "ivfflat") -> BenchmarkResults:
        """
        Run complete benchmark workflow.

        Args:
            num_vectors: Number of vectors to index
            dimension: Vector dimension
            num_queries: Number of queries to execute
            k: Number of nearest neighbors to retrieve
            index_type: Type of index to create

        Returns:
            BenchmarkResults object
        """
        print(f"Generating {num_vectors} vectors of dimension {dimension}...")
        self.vectors = self.generate_random_vectors(num_vectors, dimension, seed=42)
        self.query_vectors = self.generate_random_vectors(num_queries, dimension, seed=123)

        print("Computing ground truth...")
        self.ground_truth = self.compute_ground_truth(self.vectors, self.query_vectors, k)

        print("Connecting to database...")
        self.connect()

        try:
            print("Creating index...")
            self.create_index(index_type)

            print("Inserting vectors...")
            index_time = self.insert_vectors(self.vectors)

            print(f"Querying {num_queries} vectors...")
            results, query_times = self.query_vectors(self.query_vectors, k)

            print("Calculating metrics...")
            recall = self.calculate_recall(results, self.ground_truth, k)

            query_times_array = np.array(query_times)
            results_obj = BenchmarkResults(
                database=self.__class__.__name__.replace("Benchmark", "").lower(),
                vector_dimension=dimension,
                num_vectors=num_vectors,
                index_time=index_time,
                query_time_avg=np.mean(query_times_array),
                query_time_p50=np.percentile(query_times_array, 50),
                query_time_p95=np.percentile(query_times_array, 95),
                query_time_p99=np.percentile(query_times_array, 99),
                recall_at_10=recall,
                queries_per_second=len(query_times) / np.sum(query_times_array),
                index_type=index_type
            )

            return results_obj

        finally:
            print("Cleaning up...")
            self.cleanup()
            self.disconnect()
