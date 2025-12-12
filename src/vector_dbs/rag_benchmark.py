"""RAG-specific benchmark base class."""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import numpy as np

from src.parsers.document_parser import Document, IngestionMetrics
from src.utils.chunking import Chunk, get_chunking_strategy
from src.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingMetrics
from src.monitoring.resource_monitor import ResourceMonitor, ResourceMetrics


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: int
    query_text: str
    latency: float
    num_results: int
    top_k: int


@dataclass
class RAGBenchmarkResults:
    """Complete results for RAG benchmark."""
    database: str
    embedding_model: str

    # Dataset info
    num_documents: int
    num_chunks: int
    chunk_size: int
    chunk_strategy: str

    # Ingestion metrics
    parsing_time: float
    embedding_time: float
    insertion_time: float
    total_ingestion_time: float

    # Query metrics
    num_queries: int
    avg_query_latency: float
    p50_query_latency: float
    p95_query_latency: float
    p99_query_latency: float
    queries_per_second: float

    # Accuracy metrics (document-level)
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    mrr: float  # Mean Reciprocal Rank

    # Resource metrics
    ingestion_resources: Optional[Dict[str, Any]] = None
    query_resources: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: str = ""
    config: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class RAGBenchmark(ABC):
    """Base class for RAG benchmarks."""

    def __init__(
        self,
        db_config: Dict[str, Any],
        embedding_generator: EmbeddingGenerator,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunk_strategy: str = "sentence"
    ):
        """
        Initialize RAG benchmark.

        Args:
            db_config: Database configuration
            embedding_generator: Embedding generator instance
            chunk_size: Target chunk size (characters)
            chunk_overlap: Overlap between chunks (characters)
            chunk_strategy: Chunking strategy ('fixed', 'sentence', 'paragraph')
        """
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy_name = chunk_strategy

        # Initialize chunking strategy
        self.chunking_strategy = get_chunking_strategy(
            chunk_strategy,
            chunk_size,
            chunk_overlap
        )

        # State
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.ground_truth: Optional[List[List[int]]] = None
        self.chunk_to_doc_map: Dict[int, str] = {}  # Maps chunk_id -> doc_id

    @abstractmethod
    def connect(self) -> None:
        """Connect to vector database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from vector database."""
        pass

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """
        Create collection/table for vectors.

        Args:
            dimension: Vector dimension
        """
        pass

    @abstractmethod
    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """
        Insert chunks with embeddings into database.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings
            batch_size: Batch size for insertion

        Returns:
            Time taken for insertion (seconds)
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """
        Query database for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            Tuple of (chunk_ids, query_time)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (drop collections, etc.)."""
        pass

    def ingest_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        monitor_resources: bool = True
    ) -> IngestionMetrics:
        """
        Ingest documents: parse, chunk, embed, and insert.

        Args:
            documents: List of documents
            batch_size: Batch size for operations
            monitor_resources: Whether to monitor system resources

        Returns:
            Ingestion metrics
        """
        resource_monitor = ResourceMonitor() if monitor_resources else None

        if resource_monitor:
            resource_monitor.start()

        # Step 1: Chunk documents
        print(f"Chunking {len(documents)} documents...")
        parsing_start = time.time()

        all_chunks = []
        chunk_id = 0
        for doc in documents:
            doc_chunks = self.chunking_strategy.chunk(doc.content, doc.id)
            # Add document metadata to chunks and build mapping
            for chunk in doc_chunks:
                chunk.metadata.update({
                    'doc_id': doc.id,
                    'source': doc.source,
                    **doc.metadata
                })
                # Store chunk_id -> doc_id mapping
                self.chunk_to_doc_map[chunk_id] = doc.id
                chunk_id += 1
            all_chunks.extend(doc_chunks)

        parsing_time = time.time() - parsing_start
        self.chunks = all_chunks

        print(f"Created {len(all_chunks)} chunks in {parsing_time:.2f}s")

        # Step 2: Generate embeddings
        print(f"Generating embeddings...")
        embedding_start = time.time()

        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings, embed_metrics = self.embedding_generator.benchmark_generation(
            chunk_texts,
            batch_size=batch_size
        )
        self.embeddings = embeddings

        embedding_time = time.time() - embedding_start
        print(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")

        # Step 3: Insert into database
        print(f"Inserting into {self.db_config.get('database', 'database')}...")
        insertion_time = self.insert_chunks(all_chunks, embeddings, batch_size)

        if resource_monitor:
            resource_metrics = resource_monitor.stop()
        else:
            resource_metrics = None

        # Calculate chunk sizes
        chunk_sizes = [len(chunk.text) for chunk in all_chunks]

        return IngestionMetrics(
            num_documents=len(documents),
            num_chunks=len(all_chunks),
            total_parsing_time=parsing_time,
            total_embedding_time=embedding_time,
            total_insertion_time=insertion_time,
            avg_parsing_time_per_doc=parsing_time / len(documents),
            avg_embedding_time_per_chunk=embedding_time / len(all_chunks),
            avg_insertion_time_per_chunk=insertion_time / len(all_chunks),
            total_size_bytes=sum(len(doc.content.encode('utf-8')) for doc in documents),
            chunk_sizes=chunk_sizes,
            ingestion_resource_metrics=resource_metrics
        )

    def run_queries(
        self,
        query_texts: List[str],
        top_k: int = 10,
        monitor_resources: bool = True
    ) -> Tuple[List[QueryMetrics], ResourceMetrics]:
        """
        Run queries and measure performance.

        Args:
            query_texts: List of query strings
            top_k: Number of results to retrieve
            monitor_resources: Whether to monitor resources

        Returns:
            Tuple of (query_metrics_list, resource_metrics)
        """
        resource_monitor = ResourceMonitor() if monitor_resources else None

        if resource_monitor:
            resource_monitor.start()

        query_metrics_list = []

        print(f"Running {len(query_texts)} queries...")
        for i, query_text in enumerate(query_texts):
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query_text)

            # Execute query
            result_ids, query_time, similarity_scores = self.query(query_embedding, top_k)

            metrics = QueryMetrics(
                query_id=i,
                query_text=query_text,
                latency=query_time,
                num_results=len(result_ids),
                top_k=top_k
            )
            query_metrics_list.append(metrics)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(query_texts)} queries")

        if resource_monitor:
            resource_metrics = resource_monitor.stop()
        else:
            resource_metrics = None

        return query_metrics_list, resource_metrics

    def calculate_recall(
        self,
        query_results: List[List[int]],
        ground_truth: List[List[int]],
        k: int
    ) -> float:
        """
        Calculate recall@k.

        Args:
            query_results: Retrieved chunk IDs for each query
            ground_truth: True relevant chunk IDs for each query
            k: Number of top results to consider

        Returns:
            Average recall@k
        """
        recalls = []
        for results, truth in zip(query_results, ground_truth):
            result_set = set(results[:k])
            truth_set = set(truth[:k])
            if len(truth_set) > 0:
                recall = len(result_set & truth_set) / len(truth_set)
            else:
                recall = 0.0
            recalls.append(recall)

        return np.mean(recalls)

    def calculate_precision(
        self,
        query_results: List[List[int]],
        ground_truth: List[List[int]],
        k: int
    ) -> float:
        """
        Calculate precision@k.

        Args:
            query_results: Retrieved chunk IDs for each query
            ground_truth: True relevant chunk IDs for each query
            k: Number of top results to consider

        Returns:
            Average precision@k
        """
        precisions = []
        for results, truth in zip(query_results, ground_truth):
            result_set = set(results[:k])
            truth_set = set(truth)
            if len(result_set) > 0:
                precision = len(result_set & truth_set) / len(result_set)
            else:
                precision = 0.0
            precisions.append(precision)

        return np.mean(precisions)

    def calculate_mrr(
        self,
        query_results: List[List[int]],
        ground_truth: List[List[int]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR is the average of reciprocal ranks of the first relevant result.
        If the first relevant item appears at position 3, the reciprocal rank is 1/3.

        Args:
            query_results: Retrieved chunk IDs for each query
            ground_truth: True relevant chunk IDs for each query

        Returns:
            Mean Reciprocal Rank (MRR)
        """
        reciprocal_ranks = []
        for results, truth in zip(query_results, ground_truth):
            truth_set = set(truth)
            rank = 0
            for i, result_id in enumerate(results, 1):
                if result_id in truth_set:
                    rank = 1.0 / i
                    break
            reciprocal_ranks.append(rank)

        return np.mean(reciprocal_ranks)

    def calculate_document_level_recall(
        self,
        query_results_chunks: List[List[int]],
        ground_truth_docs: List[List[str]],
        k: int
    ) -> float:
        """
        Calculate recall@k at document level.

        Converts chunk-level results to document-level by using chunk_to_doc_map,
        then compares against ground truth document IDs.

        Args:
            query_results_chunks: Retrieved chunk IDs for each query
            ground_truth_docs: True relevant document IDs for each query
            k: Number of top results to consider

        Returns:
            Average recall@k at document level
        """
        recalls = []
        for chunk_results, doc_truth in zip(query_results_chunks, ground_truth_docs):
            # Convert chunk IDs to document IDs (deduplicate)
            retrieved_docs = []
            seen_docs = set()
            for chunk_id in chunk_results[:k]:
                doc_id = self.chunk_to_doc_map.get(chunk_id)
                if doc_id and doc_id not in seen_docs:
                    retrieved_docs.append(doc_id)
                    seen_docs.add(doc_id)

            # Calculate recall
            retrieved_set = set(retrieved_docs)
            truth_set = set(doc_truth)

            if len(truth_set) > 0:
                recall = len(retrieved_set & truth_set) / len(truth_set)
            else:
                recall = 0.0
            recalls.append(recall)

        return np.mean(recalls)

    def calculate_document_level_precision(
        self,
        query_results_chunks: List[List[int]],
        ground_truth_docs: List[List[str]],
        k: int
    ) -> float:
        """
        Calculate precision@k at document level.

        Converts chunk-level results to document-level by using chunk_to_doc_map,
        then compares against ground truth document IDs.

        Args:
            query_results_chunks: Retrieved chunk IDs for each query
            ground_truth_docs: True relevant document IDs for each query
            k: Number of top results to consider

        Returns:
            Average precision@k at document level
        """
        precisions = []
        for chunk_results, doc_truth in zip(query_results_chunks, ground_truth_docs):
            # Convert chunk IDs to document IDs (deduplicate, preserve order)
            retrieved_docs = []
            seen_docs = set()
            for chunk_id in chunk_results[:k]:
                doc_id = self.chunk_to_doc_map.get(chunk_id)
                if doc_id and doc_id not in seen_docs:
                    retrieved_docs.append(doc_id)
                    seen_docs.add(doc_id)

            # Calculate precision
            retrieved_set = set(retrieved_docs)
            truth_set = set(doc_truth)

            if len(retrieved_set) > 0:
                precision = len(retrieved_set & truth_set) / len(retrieved_set)
            else:
                precision = 0.0
            precisions.append(precision)

        return np.mean(precisions)

    def calculate_document_level_mrr(
        self,
        query_results_chunks: List[List[int]],
        ground_truth_docs: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) at document level.

        Converts chunk-level results to document-level, then finds the rank
        of the first relevant document.

        Args:
            query_results_chunks: Retrieved chunk IDs for each query
            ground_truth_docs: True relevant document IDs for each query

        Returns:
            Mean Reciprocal Rank at document level
        """
        reciprocal_ranks = []
        for chunk_results, doc_truth in zip(query_results_chunks, ground_truth_docs):
            # Convert chunk IDs to document IDs (preserve order, deduplicate)
            retrieved_docs = []
            seen_docs = set()
            for chunk_id in chunk_results:
                doc_id = self.chunk_to_doc_map.get(chunk_id)
                if doc_id and doc_id not in seen_docs:
                    retrieved_docs.append(doc_id)
                    seen_docs.add(doc_id)

            # Find rank of first relevant document
            truth_set = set(doc_truth)
            rank = 0.0
            for i, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in truth_set:
                    rank = 1.0 / i
                    break
            reciprocal_ranks.append(rank)

        return np.mean(reciprocal_ranks)

    def run_full_benchmark(
        self,
        documents: List[Document],
        query_texts: List[str],
        ground_truth_doc_ids: Optional[List[List[str]]] = None,
        top_k: int = 10,
        batch_size: int = 100
    ) -> RAGBenchmarkResults:
        """
        Run complete RAG benchmark.

        Args:
            documents: Documents to ingest
            query_texts: Query strings
            ground_truth_doc_ids: Ground truth document IDs for queries (optional)
            top_k: Number of results to retrieve
            batch_size: Batch size for operations

        Returns:
            Complete benchmark results
        """
        from datetime import datetime

        print(f"\n{'='*60}")
        print(f"Starting RAG Benchmark: {self.__class__.__name__}")
        print(f"{'='*60}")

        # Connect
        self.connect()

        try:
            # Create collection
            print(f"\nCreating collection with dimension {self.embedding_generator.dimension}")
            self.create_collection(self.embedding_generator.dimension)

            # Ingest documents
            print(f"\n--- Document Ingestion ---")
            ingestion_metrics = self.ingest_documents(
                documents,
                batch_size=batch_size,
                monitor_resources=True
            )

            # Run queries
            print(f"\n--- Query Execution ---")
            query_metrics_list, query_resources = self.run_queries(
                query_texts,
                top_k=top_k,
                monitor_resources=True
            )

            # Calculate latency statistics
            latencies = [qm.latency for qm in query_metrics_list]

            # Calculate accuracy metrics if ground truth provided
            if ground_truth_doc_ids:
                # Get query results (chunk IDs)
                query_results = []
                for query_text in query_texts:
                    query_embedding = self.embedding_generator.generate_embedding(query_text)
                    result_ids, _ = self.query(query_embedding, top_k)
                    query_results.append(result_ids)

                # Calculate document-level metrics
                recall_at_1 = self.calculate_document_level_recall(query_results, ground_truth_doc_ids, 1)
                recall_at_5 = self.calculate_document_level_recall(query_results, ground_truth_doc_ids, 5)
                recall_at_10 = self.calculate_document_level_recall(query_results, ground_truth_doc_ids, 10)
                precision_at_1 = self.calculate_document_level_precision(query_results, ground_truth_doc_ids, 1)
                precision_at_5 = self.calculate_document_level_precision(query_results, ground_truth_doc_ids, 5)
                precision_at_10 = self.calculate_document_level_precision(query_results, ground_truth_doc_ids, 10)
                mrr = self.calculate_document_level_mrr(query_results, ground_truth_doc_ids)
            else:
                recall_at_1 = recall_at_5 = recall_at_10 = 0.0
                precision_at_1 = precision_at_5 = precision_at_10 = 0.0
                mrr = 0.0

            # Build results
            results = RAGBenchmarkResults(
                database=self.__class__.__name__.replace('Benchmark', '').replace('RAG', '').lower(),
                embedding_model=self.embedding_generator.model_name,
                num_documents=ingestion_metrics.num_documents,
                num_chunks=ingestion_metrics.num_chunks,
                chunk_size=self.chunk_size,
                chunk_strategy=self.chunk_strategy_name,
                parsing_time=ingestion_metrics.total_parsing_time,
                embedding_time=ingestion_metrics.total_embedding_time,
                insertion_time=ingestion_metrics.total_insertion_time,
                total_ingestion_time=(
                    ingestion_metrics.total_parsing_time +
                    ingestion_metrics.total_embedding_time +
                    ingestion_metrics.total_insertion_time
                ),
                num_queries=len(query_texts),
                avg_query_latency=np.mean(latencies),
                p50_query_latency=np.percentile(latencies, 50),
                p95_query_latency=np.percentile(latencies, 95),
                p99_query_latency=np.percentile(latencies, 99),
                queries_per_second=len(latencies) / sum(latencies) if sum(latencies) > 0 else 0,
                recall_at_1=recall_at_1,
                recall_at_5=recall_at_5,
                recall_at_10=recall_at_10,
                precision_at_1=precision_at_1,
                precision_at_5=precision_at_5,
                precision_at_10=precision_at_10,
                mrr=mrr,
                ingestion_resources=(
                    ingestion_metrics.ingestion_resource_metrics.to_dict()
                    if getattr(ingestion_metrics, 'ingestion_resource_metrics', None) is not None and hasattr(ingestion_metrics.ingestion_resource_metrics, 'to_dict')
                    else None
                ),
                query_resources=query_resources.to_dict() if query_resources else None,
                timestamp=datetime.now().isoformat(),
                config=self.db_config
            )

            return results

        finally:
            # Cleanup
            print(f"\n--- Cleanup ---")
            self.cleanup()
            self.disconnect()
