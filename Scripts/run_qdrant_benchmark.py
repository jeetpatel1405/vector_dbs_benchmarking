#!/usr/bin/env python3
"""
Complete end-to-end benchmark for Qdrant.
Tests query latency across different top-k values.

This is the reference implementation for the vector DB benchmarking project.
See CONTRIBUTOR_GUIDE.md for how to adapt this for other databases.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import DocumentParser, Document

# Configuration
CONFIG = {
    'corpus_path': 'Data/test_corpus/documents',
    'test_cases_path': 'Data/test_corpus/test_cases.json',
    'output_dir': 'results/qdrant_experiment_001',
    'qdrant_config': {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'benchmark_test'
    },
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_type': 'sentence-transformers',  # or 'random' for fast testing
    'chunk_size': 512,
    'chunk_overlap': 50,
    'chunk_strategy': 'fixed',
    'top_k_values': [1, 3, 5, 10, 20],
    'batch_size': 100
}


def load_documents(corpus_path: str) -> List[Document]:
    """Load all documents from corpus directory."""
    print(f"\nLoading documents from {corpus_path}...")
    parser = DocumentParser()
    documents = []

    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    txt_files = list(corpus_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {corpus_path}")

    for txt_file in sorted(txt_files):
        try:
            doc = parser.parse_txt(str(txt_file))
            documents.append(doc)
            print(f"  Loaded: {txt_file.name} ({len(doc.content)} chars)")
        except Exception as e:
            print(f"  Error loading {txt_file.name}: {e}")

    print(f"Loaded {len(documents)} documents")
    return documents


def main():
    """Run complete Qdrant benchmark."""

    print("="*70)
    print("Qdrant Vector Database Benchmark")
    print("="*70)

    # 1. Setup output directory
    print("\n[1/7] Setting up output directory...")
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # 2. Load test data
    print("\n[2/7] Loading test data...")
    documents = load_documents(CONFIG['corpus_path'])

    test_cases_path = Path(CONFIG['test_cases_path'])
    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {CONFIG['test_cases_path']}")

    with open(test_cases_path) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    # 3. Initialize embedding generator
    print("\n[3/7] Initializing embedding generator...")
    print(f"Model: {CONFIG['embedding_model']}")
    print(f"Type: {CONFIG['embedding_type']}")

    embedding_gen = get_embedding_generator(
        CONFIG['embedding_type'],
        model_name=CONFIG['embedding_model'],
        dimension=384  # MiniLM dimension
    )

    # 4. Initialize Qdrant benchmark
    print("\n[4/7] Initializing Qdrant...")
    print(f"Host: {CONFIG['qdrant_config']['host']}:{CONFIG['qdrant_config']['port']}")
    print(f"Collection: {CONFIG['qdrant_config']['collection_name']}")

    benchmark = QdrantRAGBenchmark(
        db_config=CONFIG['qdrant_config'],
        embedding_generator=embedding_gen,
        chunk_size=CONFIG['chunk_size'],
        chunk_overlap=CONFIG['chunk_overlap'],
        chunk_strategy=CONFIG['chunk_strategy']
    )

    # Connect to Qdrant
    try:
        benchmark.connect()
    except Exception as e:
        print(f"\n❌ Failed to connect to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose up -d qdrant")
        print("or:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return 1

    # 5. Create collection
    print(f"\n[5/8] Creating Qdrant collection...")
    try:
        benchmark.create_collection(embedding_gen.dimension)
        print(f"✅ Collection '{CONFIG['qdrant_config']['collection_name']}' created")
    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        benchmark.disconnect()
        return 1

    # 6. Ingest documents
    print(f"\n[6/8] Ingesting {len(documents)} documents...")
    print(f"Chunk size: {CONFIG['chunk_size']}, Overlap: {CONFIG['chunk_overlap']}")
    print(f"Strategy: {CONFIG['chunk_strategy']}")

    ingest_start = time.time()
    try:
        # Ingest documents separately from querying
        ingest_result = benchmark.ingest_documents(documents, batch_size=CONFIG['batch_size'])
        ingest_time = time.time() - ingest_start

        num_docs = len(documents)
        num_chunks = ingest_result.num_chunks if hasattr(ingest_result, 'num_chunks') else 0
        parsing_time = ingest_result.parsing_time if hasattr(ingest_result, 'parsing_time') else 0
        embedding_time = ingest_result.embedding_time if hasattr(ingest_result, 'embedding_time') else 0
        insertion_time = ingest_result.insertion_time if hasattr(ingest_result, 'insertion_time') else 0

        print(f"✅ Ingestion completed in {ingest_time:.2f}s")
        print(f"   Documents: {num_docs}")
        print(f"   Chunks created: {num_chunks}")
        print(f"   Parsing time: {parsing_time:.2f}s")
        print(f"   Embedding time: {embedding_time:.2f}s")
        print(f"   Insertion time: {insertion_time:.2f}s")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        benchmark.disconnect()
        return 1

    # 7. Run queries at different top-k values
    print(f"\n[7/8] Running query latency benchmark...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Top-k values: {CONFIG['top_k_values']}")

    results = []
    for top_k in CONFIG['top_k_values']:
        print(f"\n  Testing top_k={top_k}...")
        latencies = []

        for i, tc in enumerate(test_cases, 1):
            try:
                start = time.time()
                # Generate embedding for query
                query_embedding = embedding_gen.generate_embedding(tc['query'])
                # Query the database
                result_ids, query_time = benchmark.query(query_embedding, top_k=top_k)
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)

                if i % 5 == 0 or i == len(test_cases):
                    print(f"    Query {i}/{len(test_cases)}: {latency:.2f}ms ({len(result_ids)} results)")
            except Exception as e:
                print(f"    Query {i} failed: {e}")
                import traceback
                if i == 1:  # Print full traceback for first failure
                    traceback.print_exc()
                continue

        if latencies:
            latencies_sorted = sorted(latencies)
            avg_latency = np.mean(latencies)
            p50_latency = latencies_sorted[len(latencies)//2]
            p95_latency = latencies_sorted[int(len(latencies)*0.95)]
            p99_latency = latencies_sorted[int(len(latencies)*0.99)]
            min_latency = min(latencies)
            max_latency = max(latencies)

            result = {
                'top_k': top_k,
                'num_queries': len(latencies),
                'avg_latency_ms': float(avg_latency),
                'p50_latency_ms': float(p50_latency),
                'p95_latency_ms': float(p95_latency),
                'p99_latency_ms': float(p99_latency),
                'min_latency_ms': float(min_latency),
                'max_latency_ms': float(max_latency),
                'queries_per_second': 1000.0 / avg_latency if avg_latency > 0 else 0
            }
            results.append(result)

            print(f"    Avg: {avg_latency:.2f}ms, P50: {p50_latency:.2f}ms, "
                  f"P95: {p95_latency:.2f}ms, QPS: {result['queries_per_second']:.2f}")

    # Disconnect from Qdrant
    benchmark.disconnect()

    if not results:
        print("\n❌ No successful queries! Check Qdrant connection and data.")
        return 1

    # 8. Export results and generate plots
    print(f"\n[8/8] Exporting results and generating visualizations...")

    # Export results JSON
    results_data = {
        'database': 'qdrant',
        'config': CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ingestion': {
            'total_time_sec': ingest_time,
            'num_documents': num_docs,
            'num_chunks': num_chunks,
            'parsing_time_sec': parsing_time,
            'embedding_time_sec': embedding_time,
            'insertion_time_sec': insertion_time
        },
        'query_results': results
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✅ Results saved to: {results_file}")

    # Generate latency vs top-k plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    top_k_vals = [r['top_k'] for r in results]
    avg_latencies = [r['avg_latency_ms'] for r in results]
    p95_latencies = [r['p95_latency_ms'] for r in results]
    qps_values = [r['queries_per_second'] for r in results]

    # Plot 1: Latency vs Top-K
    ax1.plot(top_k_vals, avg_latencies, marker='o', linewidth=2, markersize=8,
             label='Average', color='#2E86AB')
    ax1.plot(top_k_vals, p95_latencies, marker='s', linewidth=2, markersize=6,
             label='P95', color='#A23B72', linestyle='--')
    ax1.set_xlabel('Top-K Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Qdrant Query Latency vs Top-K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xticks(top_k_vals)

    # Plot 2: Queries Per Second vs Top-K
    ax2.plot(top_k_vals, qps_values, marker='o', linewidth=2, markersize=8,
             color='#F18F01')
    ax2.set_xlabel('Top-K Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Queries Per Second', fontsize=12, fontweight='bold')
    ax2.set_title('Qdrant Throughput vs Top-K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(top_k_vals)

    plt.tight_layout()
    plot_file = output_dir / 'latency_vs_topk.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_file}")

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nDatabase: Qdrant")
    print(f"Documents: {num_docs}")
    print(f"Chunks: {num_chunks}")
    print(f"Ingestion time: {ingest_time:.2f}s")
    print(f"\nQuery Performance:")
    print(f"{'Top-K':<8} {'Avg (ms)':<12} {'P95 (ms)':<12} {'QPS':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['top_k']:<8} {r['avg_latency_ms']:<12.2f} "
              f"{r['p95_latency_ms']:<12.2f} {r['queries_per_second']:<10.2f}")

    print(f"\n📊 Results: {results_file}")
    print(f"📈 Plot: {plot_file}")
    print("\n✅ You now have experimental data! See IMPLEMENTATION_PLAN.md for next steps.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
