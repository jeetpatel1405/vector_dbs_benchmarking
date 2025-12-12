#!/usr/bin/env python3
"""
Complete end-to-end benchmark for Weaviate.
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

from src.vector_dbs.weaviate_adapter import WeaviateRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import DocumentParser, Document
from src.monitoring.resource_monitor import ResourceMonitor

# Configuration
CONFIG = {
    'corpus_path': 'Data/test_corpus/documents',
    'test_cases_path': 'Data/test_corpus/test_cases.json',
    'output_dir': 'results/weaviate_experiment_001',
    'weaviate_config': {
        'host': 'localhost',
        'port': 8080,
        'class_name': 'BenchmarkTest'
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

    # Get all supported document files
    txt_files = list(corpus_dir.glob("*.txt"))
    xml_files = list(corpus_dir.glob("*.xml"))
    all_files = txt_files + xml_files

    if not all_files:
        raise FileNotFoundError(f"No supported files found in {corpus_path}")

    for file_path in sorted(all_files):
        try:
            doc = parser.parse_file(str(file_path))
            documents.append(doc)
            print(f"  Loaded: {file_path.name} ({len(doc.content)} chars)")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    print(f"Loaded {len(documents)} documents")
    return documents


def main():
    """Run complete Qdrant benchmark."""

    print("="*70)
    print("Weaviate Vector Database Benchmark")
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
    print("\n[4/7] Initializing Weaviate...")
    print(f"Host: {CONFIG['weaviate_config']['host']}:{CONFIG['weaviate_config']['port']}")
    print(f"Collection: {CONFIG['weaviate_config']['class_name']}")

    benchmark = WeaviateRAGBenchmark(
        db_config=CONFIG['weaviate_config'],
        embedding_generator=embedding_gen,
        chunk_size=CONFIG['chunk_size'],
        chunk_overlap=CONFIG['chunk_overlap'],
        chunk_strategy=CONFIG['chunk_strategy']
    )

    # Connect to Qdrant
    try:
        benchmark.connect()
    except Exception as e:
        print(f"\n❌ Failed to connect to Weaviate: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker-compose up -d weaviate")
        print("or:")
        print("  docker-compose up -d weaviate")
        return 1

    # 5. Create collection
    print(f"\n[5/8] Creating Qdrant collection...")
    try:
        benchmark.create_collection(embedding_gen.dimension)
        print(f"✅ Collection '{CONFIG['weaviate_config']['class_name']}' created")
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
        num_chunks = ingest_result.num_chunks
        parsing_time = ingest_result.total_parsing_time
        embedding_time = ingest_result.total_embedding_time
        insertion_time = ingest_result.total_insertion_time

        print(f"✅ Ingestion completed in {ingest_time:.2f}s")
        print(f"   Documents: {num_docs}")
        print(f"   Chunks created: {num_chunks}")
        print(f"   Parsing time: {parsing_time:.2f}s")
        print(f"   Embedding time: {embedding_time:.2f}s")
        print(f"   Insertion time: {insertion_time:.2f}s")

        # Display ingestion resource metrics if available
        if hasattr(ingest_result, 'ingestion_resource_metrics') and ingest_result.ingestion_resource_metrics:
            rm = ingest_result.ingestion_resource_metrics
            print(f"   📊 Resource Usage:")
            print(f"      Duration: {rm.duration:.2f}s")
            print(f"      CPU: avg={rm.cpu_avg:.1f}%, max={rm.cpu_max:.1f}%")
            print(f"      Memory: avg={rm.memory_avg_mb:.1f}MB, max={rm.memory_max_mb:.1f}MB")
            if rm.disk_read_total_mb > 0 or rm.disk_write_total_mb > 0:
                print(f"      Disk: read={rm.disk_read_total_mb:.2f}MB, write={rm.disk_write_total_mb:.2f}MB")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        benchmark.disconnect()
        return 1

    # 7. Run queries at different top-k values and calculate IR metrics
    print(f"\n[7/8] Running query latency and quality benchmark...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Top-k values: {CONFIG['top_k_values']}")

    # Extract ground truth document IDs from test cases
    ground_truth_doc_ids = [tc['relevant_doc_ids'] for tc in test_cases]
    queries = [tc['query'] for tc in test_cases]

    results = []
    for top_k in CONFIG['top_k_values']:
        print(f"\n  Testing top_k={top_k}...")

        # Start resource monitoring for this top_k
        resource_monitor = ResourceMonitor()
        resource_monitor.start()

        latencies = []
        all_similarities = []  # Track all similarity scores
        query_results_chunks = []  # Store chunk IDs for IR metrics

        for i, tc in enumerate(test_cases, 1):
            try:
                start = time.time()
                # Generate embedding for query
                query_embedding = embedding_gen.generate_embedding(tc['query'])
                # Query the database
                result_ids, query_time, similarity_scores = benchmark.query(query_embedding, top_k=top_k)
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
                query_results_chunks.append(result_ids)

                # Track similarity scores for quality metrics
                if similarity_scores:
                    all_similarities.append(similarity_scores)

                if i % 5 == 0 or i == len(test_cases):
                    avg_sim = np.mean(similarity_scores) if similarity_scores else 0
                    print(f"    Query {i}/{len(test_cases)}: {latency:.2f}ms, "
                          f"{len(result_ids)} results, avg_similarity={avg_sim:.3f}")
            except Exception as e:
                print(f"    Query {i} failed: {e}")
                import traceback
                if i == 1:  # Print full traceback for first failure
                    traceback.print_exc()
                continue

        # Stop resource monitoring
        resource_metrics = resource_monitor.stop()

        if latencies:
            latencies_sorted = sorted(latencies)
            avg_latency = np.mean(latencies)
            p50_latency = latencies_sorted[len(latencies)//2]
            p95_latency = latencies_sorted[int(len(latencies)*0.95)]
            p99_latency = latencies_sorted[int(len(latencies)*0.99)]
            min_latency = min(latencies)
            max_latency = max(latencies)

            # Calculate similarity-based quality metrics
            avg_similarity = float(np.mean([np.mean(sims) for sims in all_similarities])) if all_similarities else 0.0
            avg_top1_similarity = float(np.mean([sims[0] for sims in all_similarities if len(sims) > 0])) if all_similarities else 0.0
            min_similarity = float(np.mean([np.min(sims) for sims in all_similarities])) if all_similarities else 0.0

            # Calculate document-level IR metrics
            recall_at_k = float(benchmark.calculate_document_level_recall(query_results_chunks, ground_truth_doc_ids, top_k))
            precision_at_k = float(benchmark.calculate_document_level_precision(query_results_chunks, ground_truth_doc_ids, top_k))
            mrr = float(benchmark.calculate_document_level_mrr(query_results_chunks, ground_truth_doc_ids))

            result = {
                'top_k': top_k,
                'num_queries': len(latencies),
                'avg_latency_ms': float(avg_latency),
                'p50_latency_ms': float(p50_latency),
                'p95_latency_ms': float(p95_latency),
                'p99_latency_ms': float(p99_latency),
                'min_latency_ms': float(min_latency),
                'max_latency_ms': float(max_latency),
                'queries_per_second': 1000.0 / avg_latency if avg_latency > 0 else 0,
                # Similarity-based quality metrics
                'avg_similarity': avg_similarity,
                'avg_top1_similarity': avg_top1_similarity,
                'min_similarity': min_similarity,
                # IR metrics (document-level)
                f'recall_at_{top_k}': recall_at_k,
                f'precision_at_{top_k}': precision_at_k,
                'mrr': mrr,
                # Resource metrics
                'resource_metrics': resource_metrics.to_dict() if resource_metrics else None,
                'ingestion_resource_metrics': (
                    ingest_result.ingestion_resource_metrics.to_dict()
                    if getattr(ingest_result, 'ingestion_resource_metrics', None) is not None and hasattr(ingest_result.ingestion_resource_metrics, 'to_dict')
                    else None
                )
            }
            results.append(result)

            print(f"    Avg: {avg_latency:.2f}ms, P50: {p50_latency:.2f}ms, "
                  f"P95: {p95_latency:.2f}ms, QPS: {result['queries_per_second']:.2f}")
            print(f"    Similarity: avg={avg_similarity:.3f}, top1={avg_top1_similarity:.3f}")
            print(f"    IR Metrics: Recall@{top_k}={recall_at_k:.3f}, Precision@{top_k}={precision_at_k:.3f}, MRR={mrr:.3f}")

    # Disconnect from Qdrant
    benchmark.disconnect()

    if not results:
        print("\n❌ No successful queries! Check Weaviate connection and data.")
        return 1

    # 8. Export results and generate plots
    print(f"\n[8/8] Exporting results and generating visualizations...")

    # Export results JSON
    results_data = {
        'database': 'weaviate',
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

    # Generate performance and quality plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    top_k_vals = [r['top_k'] for r in results]
    avg_latencies = [r['avg_latency_ms'] for r in results]
    p95_latencies = [r['p95_latency_ms'] for r in results]
    qps_values = [r['queries_per_second'] for r in results]
    avg_similarities = [r['avg_similarity'] for r in results]
    top1_similarities = [r['avg_top1_similarity'] for r in results]

    # Plot 1: Latency vs Top-K
    ax1 = axes[0, 0]
    ax1.plot(top_k_vals, avg_latencies, marker='o', linewidth=2, markersize=8,
             label='Average', color='#2E86AB')
    ax1.plot(top_k_vals, p95_latencies, marker='s', linewidth=2, markersize=6,
             label='P95', color='#A23B72', linestyle='--')
    ax1.set_xlabel('Top-K Value', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Query Latency (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Query Latency vs Top-K', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    ax1.set_xticks(top_k_vals)

    # Plot 2: Queries Per Second vs Top-K
    ax2 = axes[0, 1]
    ax2.plot(top_k_vals, qps_values, marker='o', linewidth=2, markersize=8,
             color='#F18F01')
    ax2.set_xlabel('Top-K Value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Queries Per Second', fontsize=11, fontweight='bold')
    ax2.set_title('Throughput vs Top-K', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(top_k_vals)

    # Plot 3: Average Similarity vs Top-K (Quality)
    ax3 = axes[1, 0]
    ax3.plot(top_k_vals, avg_similarities, marker='o', linewidth=2, markersize=8,
             label='Avg All Results', color='#06A77D')
    ax3.plot(top_k_vals, top1_similarities, marker='s', linewidth=2, markersize=6,
             label='Top-1 Result', color='#D62828', linestyle='--')
    ax3.set_xlabel('Top-K Value', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cosine Similarity', fontsize=11, fontweight='bold')
    ax3.set_title('Retrieval Quality (Semantic Similarity)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)
    ax3.set_xticks(top_k_vals)
    ax3.set_ylim([0, 1])

    # Plot 4: Quality-Speed Tradeoff
    ax4 = axes[1, 1]
    scatter = ax4.scatter(avg_latencies, avg_similarities, c=top_k_vals,
                          s=200, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    for i, k in enumerate(top_k_vals):
        ax4.annotate(f'k={k}', (avg_latencies[i], avg_similarities[i]),
                     fontsize=9, ha='center', va='center', fontweight='bold')
    ax4.set_xlabel('Query Latency (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Similarity', fontsize=11, fontweight='bold')
    ax4.set_title('Quality-Speed Tradeoff', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Top-K', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_file = output_dir / 'performance_quality.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Performance & quality plot saved to: {plot_file}")

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nDatabase: Weaviate")
    print(f"Documents: {num_docs}")
    print(f"Chunks: {num_chunks}")
    print(f"Ingestion time: {ingest_time:.2f}s")
    print(f"\nQuery Performance & Quality:")
    print(f"{'Top-K':<8} {'Avg (ms)':<12} {'P95 (ms)':<12} {'QPS':<10} {'Avg Sim':<10} {'Top-1 Sim':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['top_k']:<8} {r['avg_latency_ms']:<12.2f} "
              f"{r['p95_latency_ms']:<12.2f} {r['queries_per_second']:<10.2f} "
              f"{r['avg_similarity']:<10.3f} {r['avg_top1_similarity']:<10.3f}")

    print(f"\n📊 Results: {results_file}")
    print(f"📈 Plot: {plot_file}")
    print("\n✅ You now have experimental data with quality metrics!")
    print("   - Semantic similarity measures retrieval quality (0-1 scale)")
    print("   - Higher similarity = more relevant results")
    print("   - See IMPLEMENTATION_PLAN.md for next steps.")
    
    # Display resource metrics summary if available  
    resource_summary = {}
    for r in results:
        if r.get('resource_metrics'):
            rm = r['resource_metrics']
            topk = r['top_k']
            resource_summary[topk] = {
                'cpu_avg': rm['cpu']['avg'],
                'memory_avg': rm['memory']['avg_mb'],
                'duration': rm['duration']
            }
    
    if resource_summary:
        print(f"\n📊 Resource Usage During Queries:")
        print(f"{'Top-K':<8} {'Duration(s)':<12} {'CPU Avg(%)':<12} {'Memory(MB)':<12}")
        print("-" * 48)
        for topk, metrics in resource_summary.items():
            print(f"{topk:<8} {metrics['duration']:<12.2f} {metrics['cpu_avg']:<12.1f} {metrics['memory_avg']:<12.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
