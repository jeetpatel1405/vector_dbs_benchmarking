#!/usr/bin/env python3
"""Comprehensive benchmark orchestrator for RAG systems."""
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from src.embeddings.embedding_generator import get_embedding_generator, EMBEDDING_CONFIGS
from src.parsers.document_parser import DocumentParser, Document

# Import all database adapters
from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark
from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark
from src.vector_dbs.weaviate_adapter import WeaviateRAGBenchmark
from src.vector_dbs.milvus_adapter import MilvusRAGBenchmark
from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark
from src.vector_dbs.faiss_adapter import FAISSRAGBenchmark
from src.vector_dbs.opensearch_adapter import OpenSearchRAGBenchmark


class BenchmarkRunner:
    """Orchestrates RAG benchmarks across multiple vector databases."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results = []
        self.benchmarks = {}  # Will be populated with database-specific benchmarks

        # Auto-register all database adapters
        self._register_adapters()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _register_adapters(self):
        """Register all available database adapters."""
        self.benchmarks['pgvector'] = PgvectorRAGBenchmark
        self.benchmarks['qdrant'] = QdrantRAGBenchmark
        self.benchmarks['weaviate'] = WeaviateRAGBenchmark
        self.benchmarks['milvus'] = MilvusRAGBenchmark
        self.benchmarks['chroma'] = ChromaRAGBenchmark
        self.benchmarks['faiss'] = FAISSRAGBenchmark
        self.benchmarks['opensearch'] = OpenSearchRAGBenchmark

    def register_benchmark(self, name: str, benchmark_class):
        """
        Register a database benchmark class.

        Args:
            name: Database name (e.g., 'pgvector', 'qdrant')
            benchmark_class: Benchmark class for this database
        """
        self.benchmarks[name] = benchmark_class

    def prepare_documents(self, data_dir: str = None) -> List[Document]:
        """
        Prepare documents for benchmarking.

        Args:
            data_dir: Directory containing documents (if None, uses config)

        Returns:
            List of parsed documents
        """
        data_dir = data_dir or self.config.get('data_dir', 'data/sample_docs')

        print(f"\nLoading documents from: {data_dir}")

        parser = DocumentParser()
        documents = parser.parse_directory(data_dir, recursive=True)

        print(f"Loaded {len(documents)} documents")

        # Limit documents if specified
        max_docs = self.config.get('max_documents')
        if max_docs and len(documents) > max_docs:
            documents = documents[:max_docs]
            print(f"Limited to {max_docs} documents")

        return documents

    def run_benchmark(
        self,
        database: str,
        documents: List[Document],
        queries: List[str],
        embedding_config: str,
        chunk_size: int,
        chunk_strategy: str
    ) -> Dict[str, Any]:
        """
        Run benchmark for a specific configuration.

        Args:
            database: Database name ('pgvector', 'qdrant', 'weaviate', etc.)
            documents: Documents to ingest
            queries: Query strings
            embedding_config: Embedding configuration name
            chunk_size: Chunk size in characters
            chunk_strategy: Chunking strategy

        Returns:
            Benchmark results as dictionary
        """
        print(f"\n{'='*80}")
        print(f"Benchmarking {database.upper()}")
        print(f"Embedding: {embedding_config}, Chunk Size: {chunk_size}, Strategy: {chunk_strategy}")
        print(f"{'='*80}")

        # Get embedding generator
        embed_config = EMBEDDING_CONFIGS.get(embedding_config)
        if not embed_config:
            raise ValueError(f"Unknown embedding config: {embedding_config}")

        embedding_gen = get_embedding_generator(**embed_config)

        # Get database configuration
        db_config = self.config['databases'].get(database)
        if not db_config:
            raise ValueError(f"Unknown database: {database}")

        # Get benchmark class
        benchmark_class = self.benchmarks.get(database)
        if not benchmark_class:
            raise ValueError(f"No benchmark registered for: {database}")

        # Create benchmark instance
        benchmark = benchmark_class(
            db_config=db_config,
            embedding_generator=embedding_gen,
            chunk_size=chunk_size,
            chunk_overlap=self.config.get('chunk_overlap', 50),
            chunk_strategy=chunk_strategy
        )

        # Run benchmark
        results = benchmark.run_full_benchmark(
            documents=documents,
            query_texts=queries,
            top_k=self.config.get('top_k', 10),
            batch_size=self.config.get('batch_size', 100)
        )

        return results.to_dict()

    def run_all_scenarios(self):
        """Run all benchmark scenarios defined in config."""
        # Load documents
        documents = self.prepare_documents()

        # Load queries
        queries = self.config.get('queries', [
            'What is machine learning?',
            'How do vector databases work?',
            'Explain RAG systems',
            'What is semantic search?',
            'How to optimize embeddings?'
        ])

        print(f"\nRunning {len(queries)} queries per benchmark")

        # Get benchmark scenarios
        scenarios = self.config.get('scenarios', [])

        if not scenarios:
            # Default scenario
            scenarios = [{
                'database': 'pgvector',
                'embedding': 'sentence-transformers-small',
                'chunk_size': 512,
                'chunk_strategy': 'sentence'
            }]

        # Run each scenario
        for i, scenario in enumerate(scenarios):
            print(f"\n\n{'#'*80}")
            print(f"# Scenario {i+1}/{len(scenarios)}")
            print(f"{'#'*80}")

            try:
                result = self.run_benchmark(
                    database=scenario['database'],
                    documents=documents,
                    queries=queries,
                    embedding_config=scenario['embedding'],
                    chunk_size=scenario['chunk_size'],
                    chunk_strategy=scenario['chunk_strategy']
                )

                # Add scenario metadata
                result['scenario'] = scenario
                result['scenario_id'] = i + 1

                self.results.append(result)

            except Exception as e:
                print(f"\nERROR in scenario {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # Save results
        self.save_results()

        # Generate comparison
        self.generate_comparison()

    def save_results(self, output_dir: str = "results"):
        """Save benchmark results."""
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual results as JSON
        for i, result in enumerate(self.results):
            filename = f"{output_dir}/benchmark_{result['database']}_{timestamp}_{i+1}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved: {filename}")

        # Save all results as single JSON
        all_results_file = f"{output_dir}/all_results_{timestamp}.json"
        with open(all_results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved: {all_results_file}")

        # Save as CSV for easy analysis
        self._save_csv(output_dir, timestamp)

    def _save_csv(self, output_dir: str, timestamp: str):
        """Save results as CSV."""
        if not self.results:
            return

        # Flatten results for CSV
        rows = []
        for result in self.results:
            row = {
                'database': result['database'],
                'embedding_model': result['embedding_model'],
                'chunk_size': result['chunk_size'],
                'chunk_strategy': result['chunk_strategy'],
                'num_documents': result['num_documents'],
                'num_chunks': result['num_chunks'],
                'parsing_time': result['parsing_time'],
                'embedding_time': result['embedding_time'],
                'insertion_time': result['insertion_time'],
                'total_ingestion_time': result['total_ingestion_time'],
                'num_queries': result['num_queries'],
                'avg_query_latency': result['avg_query_latency'],
                'p50_query_latency': result['p50_query_latency'],
                'p95_query_latency': result['p95_query_latency'],
                'p99_query_latency': result['p99_query_latency'],
                'queries_per_second': result['queries_per_second'],
                'recall_at_1': result['recall_at_1'],
                'recall_at_5': result['recall_at_5'],
                'recall_at_10': result['recall_at_10'],
                'precision_at_10': result['precision_at_10'],
                'timestamp': result['timestamp']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_file = f"{output_dir}/results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

    def generate_comparison(self):
        """Generate comparison report."""
        if not self.results:
            print("\nNo results to compare")
            return

        print(f"\n\n{'='*80}")
        print("BENCHMARK COMPARISON")
        print(f"{'='*80}\n")

        # Create comparison table
        rows = []
        for result in self.results:
            rows.append({
                'Database': result['database'],
                'Embedding': result['embedding_model'][:30],
                'Chunks': result['num_chunks'],
                'Chunk Size': result['chunk_size'],
                'Ingestion (s)': f"{result['total_ingestion_time']:.2f}",
                'Avg Query (ms)': f"{result['avg_query_latency']*1000:.2f}",
                'P95 Query (ms)': f"{result['p95_query_latency']*1000:.2f}",
                'QPS': f"{result['queries_per_second']:.2f}",
                'Recall@10': f"{result['recall_at_10']*100:.1f}%"
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(f"\n{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RAG Vector Database Benchmark Suite'
    )
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--database',
        choices=['pgvector', 'qdrant', 'weaviate', 'chroma', 'milvus', 'faiss', 'opensearch', 'all'],
        help='Specific database to benchmark (default: all from config)'
    )
    parser.add_argument(
        '--data-dir',
        help='Directory containing documents to benchmark'
    )

    args = parser.parse_args()

    # Create runner
    runner = BenchmarkRunner(args.config)

    # Note: Benchmarks need to be registered before running
    # This will be done in the main application setup

    # Run benchmarks
    if args.database and args.database != 'all':
        # Run single database (override config scenarios)
        documents = runner.prepare_documents(args.data_dir)
        queries = runner.config.get('queries', ['test query'])

        result = runner.run_benchmark(
            database=args.database,
            documents=documents,
            queries=queries,
            embedding_config='sentence-transformers-small',
            chunk_size=512,
            chunk_strategy='sentence'
        )
        runner.results.append(result)
        runner.save_results()
    else:
        # Run all scenarios from config
        runner.run_all_scenarios()


if __name__ == "__main__":
    main()
