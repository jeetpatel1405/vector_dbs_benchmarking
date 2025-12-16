#!/usr/bin/env python3
"""
Ingestion performance benchmark for Milvus.
Tests how chunk size, batch size, and document count affect ingestion performance.

This complements the query latency benchmark (run_qdrant_benchmark.py) by
focusing on the document ingestion pipeline.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_dbs.milvus_adapter import MilvusRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import DocumentParser, Document

# Configuration
CONFIG = {
    'corpus_path': 'Data/test_corpus/documents',
    'output_dir': 'results/milvus_ingestion_experiment_001',
    'milvus_config': {
        'host': 'localhost',
        'port': 19530,
        'collection_name': 'ingestion_benchmark'
    },
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_type': 'sentence-transformers',

    # Experiment parameters - SIMPLIFIED FOR SPEED
    'chunk_sizes': [256, 512, 1024],  # 3 sizes instead of 5
    'batch_sizes': [50, 100],  # 2 batch sizes instead of 4
    'doc_counts': [10, 20],  # 2 doc counts instead of 4
    'chunk_overlap': 50,
    'chunk_strategy': 'fixed',
    'n_runs': 1  # Single run instead of 2 for speed
}


def load_documents(corpus_path: str, max_docs: int = None) -> List[Document]:
    """Load documents from corpus directory."""
    parser = DocumentParser()
    documents = []

    corpus_dir = Path(corpus_path)
    # Get all supported document files
    txt_files = sorted(corpus_dir.glob("*.txt"))
    xml_files = sorted(corpus_dir.glob("*.xml"))
    all_files = sorted(txt_files + xml_files)

    if max_docs:
        all_files = all_files[:max_docs]

    for file_path in all_files:
        doc = parser.parse_file(str(file_path))
        documents.append(doc)

    return documents


def run_ingestion_experiment(
    benchmark: MilvusRAGBenchmark,
    documents: List[Document],
    chunk_size: int,
    batch_size: int,
    run_num: int
) -> Dict:
    """Run a single ingestion experiment."""

    # Recreate collection for clean state
    benchmark.create_collection(benchmark.embedding_generator.dimension)

    # Configure chunk size
    benchmark.chunk_size = chunk_size

    # Run ingestion
    start_time = time.time()
    ingest_result = benchmark.ingest_documents(documents, batch_size=batch_size)
    total_time = time.time() - start_time

    # Calculate metrics
    num_docs = len(documents)
    num_chunks = ingest_result.num_chunks
    total_size = sum(len(doc.content.encode('utf-8')) for doc in documents)

    result = {
        'run': run_num,
        'chunk_size': chunk_size,
        'batch_size': batch_size,
        'num_docs': num_docs,
        'num_chunks': num_chunks,
        'total_size_bytes': total_size,
        'parsing_time': ingest_result.total_parsing_time,
        'embedding_time': ingest_result.total_embedding_time,
        'insertion_time': ingest_result.total_insertion_time,
        'total_time': total_time,
        'chunks_per_second': num_chunks / total_time if total_time > 0 else 0,
        'bytes_per_second': total_size / total_time if total_time > 0 else 0,
        'docs_per_second': num_docs / total_time if total_time > 0 else 0
    }

    return result


def main():
    """Run comprehensive ingestion benchmark."""

    print("="*70)
    print("Milvus Ingestion Performance Benchmark")
    print("="*70)

    # Setup
    print("\n[1/6] Setting up...")
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # Load all documents
    print("\n[2/6] Loading test corpus...")
    all_documents = load_documents(CONFIG['corpus_path'])
    print(f"Loaded {len(all_documents)} documents")

    # Initialize embedding generator
    print("\n[3/6] Initializing embedding generator...")
    embedding_gen = get_embedding_generator(
        CONFIG['embedding_type'],
        model_name=CONFIG['embedding_model'],
        dimension=384
    )

    # Initialize Qdrant
    print("\n[4/6] Connecting to Qdrant...")
    benchmark = MilvusRAGBenchmark(
        db_config=CONFIG['milvus_config'],
        embedding_generator=embedding_gen,
        chunk_size=CONFIG['chunk_sizes'][0],  # Will be updated per experiment
        chunk_overlap=CONFIG['chunk_overlap'],
        chunk_strategy=CONFIG['chunk_strategy']
    )
    benchmark.connect()

    # Run experiments
    print("\n[5/6] Running ingestion experiments...")
    print(f"Configurations to test: {len(CONFIG['chunk_sizes'])} chunk sizes × "
          f"{len(CONFIG['batch_sizes'])} batch sizes × "
          f"{len(CONFIG['doc_counts'])} doc counts × "
          f"{CONFIG['n_runs']} runs")

    all_results = []
    total_experiments = (len(CONFIG['chunk_sizes']) * len(CONFIG['batch_sizes']) *
                        len(CONFIG['doc_counts']) * CONFIG['n_runs'])
    experiment_num = 0

    for chunk_size in CONFIG['chunk_sizes']:
        for batch_size in CONFIG['batch_sizes']:
            for doc_count in CONFIG['doc_counts']:
                documents = all_documents[:doc_count]

                for run in range(CONFIG['n_runs']):
                    experiment_num += 1
                    print(f"\n  [{experiment_num}/{total_experiments}] "
                          f"chunk_size={chunk_size}, batch_size={batch_size}, "
                          f"docs={doc_count}, run={run+1}/{CONFIG['n_runs']}")

                    try:
                        result = run_ingestion_experiment(
                            benchmark, documents, chunk_size, batch_size, run+1
                        )
                        all_results.append(result)

                        print(f"    ✓ {result['num_chunks']} chunks in {result['total_time']:.2f}s "
                              f"({result['chunks_per_second']:.1f} chunks/s)")

                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
                        continue

    benchmark.disconnect()

    if not all_results:
        print("\n❌ No successful experiments!")
        return 1

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)

    # Aggregate results (average across runs)
    agg_df = df.groupby(['chunk_size', 'batch_size', 'num_docs']).agg({
        'num_chunks': 'mean',
        'total_size_bytes': 'mean',
        'parsing_time': 'mean',
        'embedding_time': 'mean',
        'insertion_time': 'mean',
        'total_time': 'mean',
        'chunks_per_second': 'mean',
        'bytes_per_second': 'mean',
        'docs_per_second': 'mean'
    }).reset_index()

    # Export results
    print("\n[6/6] Exporting results...")

    results_data = {
        'config': CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': len(all_results),
        'raw_results': all_results,
        'aggregated_results': agg_df.to_dict('records')
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"✅ Results saved to: {results_file}")

    # Export CSV for easy analysis
    csv_file = output_dir / 'results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV saved to: {csv_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_plots(agg_df, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("INGESTION BENCHMARK COMPLETE")
    print("="*70)

    print("\n📊 Overall Statistics:")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Chunk sizes tested: {CONFIG['chunk_sizes']}")
    print(f"  Batch sizes tested: {CONFIG['batch_sizes']}")
    print(f"  Document counts: {CONFIG['doc_counts']}")

    print("\n⚡ Performance Ranges:")
    print(f"  Throughput: {df['chunks_per_second'].min():.1f} - {df['chunks_per_second'].max():.1f} chunks/s")
    print(f"  Total time: {df['total_time'].min():.2f} - {df['total_time'].max():.2f}s")
    print(f"  Embedding time: {df['embedding_time'].min():.2f} - {df['embedding_time'].max():.2f}s")

    # Find optimal configurations
    print("\n🏆 Optimal Configurations:")
    fastest = df.loc[df['total_time'].idxmin()]
    print(f"  Fastest total time: chunk_size={fastest['chunk_size']}, "
          f"batch_size={fastest['batch_size']}, docs={fastest['num_docs']}")

    best_throughput = df.loc[df['chunks_per_second'].idxmax()]
    print(f"  Best throughput: chunk_size={best_throughput['chunk_size']}, "
          f"batch_size={best_throughput['batch_size']}, docs={best_throughput['num_docs']}")

    print(f"\n📁 Results: {output_dir}")
    print(f"📊 Plots: {output_dir}/*.png")

    return 0


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""

    # Use 20 docs for main plots (full corpus)
    df_full = df[df['num_docs'] == 20]

    if df_full.empty:
        df_full = df[df['num_docs'] == df['num_docs'].max()]

    # Plot 1: Chunk size vs Total time (fixed batch size)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Total ingestion time vs chunk size
    ax = axes[0, 0]
    for batch_size in sorted(df_full['batch_size'].unique()):
        data = df_full[df_full['batch_size'] == batch_size]
        ax.plot(data['chunk_size'], data['total_time'],
                marker='o', label=f'Batch={batch_size}', linewidth=2)
    ax.set_xlabel('Chunk Size (characters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Ingestion Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Ingestion Time vs Chunk Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Throughput vs chunk size
    ax = axes[0, 1]
    for batch_size in sorted(df_full['batch_size'].unique()):
        data = df_full[df_full['batch_size'] == batch_size]
        ax.plot(data['chunk_size'], data['chunks_per_second'],
                marker='s', label=f'Batch={batch_size}', linewidth=2)
    ax.set_xlabel('Chunk Size (characters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Throughput (chunks/s)', fontsize=11, fontweight='bold')
    ax.set_title('Throughput vs Chunk Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Time breakdown by phase
    ax = axes[1, 0]
    chunk_size = 512  # Standard size
    batch_size = 100  # Standard batch
    data = df_full[(df_full['chunk_size'] == chunk_size) &
                   (df_full['batch_size'] == batch_size)]

    if not data.empty:
        phases = ['parsing_time', 'embedding_time', 'insertion_time']
        values = [data[phase].values[0] for phase in phases]
        labels = ['Parsing', 'Embedding', 'Insertion']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(f'Ingestion Phase Breakdown\n(chunk={chunk_size}, batch={batch_size})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Batch size impact
    ax = axes[1, 1]
    chunk_size = 512
    data = df_full[df_full['chunk_size'] == chunk_size]
    ax.plot(data['batch_size'], data['total_time'],
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Ingestion Time (s)', fontsize=11, fontweight='bold')
    ax.set_title(f'Batch Size Impact\n(chunk_size={chunk_size})',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'ingestion_performance.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Main plot saved to: {plot_file}")
    plt.close()

    # Plot 2: Scaling with document count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    chunk_size = 512
    batch_size = 100
    data = df[(df['chunk_size'] == chunk_size) & (df['batch_size'] == batch_size)]

    # Time scaling
    ax1.plot(data['num_docs'], data['total_time'],
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of Documents', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Ingestion Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Scaling with Document Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Chunks created
    ax2.plot(data['num_docs'], data['num_chunks'],
             marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Number of Documents', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Chunks Created', fontsize=11, fontweight='bold')
    ax2.set_title('Chunks vs Documents', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'scaling_performance.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Scaling plot saved to: {plot_file}")
    plt.close()

    # Plot 3: Heatmap of chunk_size vs batch_size
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot = df_full.pivot_table(
        values='total_time',
        index='chunk_size',
        columns='batch_size'
    )

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Chunk Size', fontsize=11, fontweight='bold')
    ax.set_title('Ingestion Time Heatmap (seconds)', fontsize=12, fontweight='bold')

    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='Time (seconds)')
    plt.tight_layout()

    plot_file = output_dir / 'ingestion_heatmap.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {plot_file}")
    plt.close()


if __name__ == '__main__':
    sys.exit(main())
