#!/usr/bin/env python3
"""
Recreate all_databases_comparison.png matching the original style.
Uses query benchmark data from experiment_001 folders.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_query_results():
    """Load query benchmark results from all databases."""
    results_dir = project_root / "results"
    all_results = {}

    db_folders = [
        "faiss_experiment_001",
        "chroma_experiment_001",
        "qdrant_experiment_001",
        "pgvector_experiment_001",
        "weaviate_experiment_001",
        "milvus_experiment_001",
        "opensearch_experiment_001"
    ]

    for folder in db_folders:
        results_file = results_dir / folder / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    if 'query_results' in data:
                        db_name = folder.split('_')[0].upper()
                        all_results[db_name] = data['query_results']
                        print(f"✓ Loaded {db_name}: {len(data['query_results'])} top-k results")
            except Exception as e:
                print(f"✗ Error loading {folder}: {e}")

    return all_results


def create_comparison_plot(results, output_path):
    """Create 4-panel comparison plot matching original style."""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Vector Database Benchmark Comparison (7 Databases)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define colors for databases
    db_order = ['FAISS', 'CHROMA', 'QDRANT', 'PGVECTOR', 'WEAVIATE', 'MILVUS', 'OPENSEARCH']
    colors = {
        'FAISS': '#FF6B6B',
        'CHROMA': '#4ECDC4',
        'QDRANT': '#45B7D1',
        'PGVECTOR': '#96CEB4',
        'WEAVIATE': '#FFEAA7',
        'MILVUS': '#DDA15E',
        'OPENSEARCH': '#B8B8D1'
    }

    # Panel 1: Query Latency by Top K
    ax1 = axes[0, 0]
    for db in db_order:
        if db in results:
            top_k_vals = [r['top_k'] for r in results[db]]
            latencies = [r['avg_latency_ms'] for r in results[db]]
            ax1.plot(top_k_vals, latencies, marker='o', linewidth=2.5,
                    markersize=8, label=db, color=colors.get(db, 'gray'))

    ax1.set_xlabel('Top K', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Query Latency by Top K', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Query Throughput by Top K
    ax2 = axes[0, 1]
    for db in db_order:
        if db in results:
            top_k_vals = [r['top_k'] for r in results[db]]
            qps = [r['queries_per_second'] for r in results[db]]
            ax2.plot(top_k_vals, qps, marker='s', linewidth=2.5,
                    markersize=8, label=db, color=colors.get(db, 'gray'))

    ax2.set_xlabel('Top K', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Queries per Second', fontsize=12, fontweight='bold')
    ax2.set_title('Query Throughput by Top K', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel 3: Quality Metrics - Average Similarity
    ax3 = axes[1, 0]
    for db in db_order:
        if db in results:
            top_k_vals = [r['top_k'] for r in results[db]]
            similarities = [r.get('avg_similarity', 0) for r in results[db]]
            ax3.plot(top_k_vals, similarities, marker='o', linewidth=2.5,
                    markersize=8, label=db, color=colors.get(db, 'gray'))

    ax3.set_xlabel('Top K', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Similarity Score', fontsize=12, fontweight='bold')
    ax3.set_title('Quality Metrics: Average Similarity Across Top K Results',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0.3, 0.9])

    # Panel 4: Performance Summary at K=5
    ax4 = axes[1, 1]

    # Get data for K=5
    k5_data = {}
    for db in db_order:
        if db in results:
            for r in results[db]:
                if r['top_k'] == 5:
                    k5_data[db] = r
                    break

    if k5_data:
        dbs = list(k5_data.keys())

        # Normalize throughput to 0-100 scale
        throughputs = [k5_data[db]['queries_per_second'] for db in dbs]
        max_throughput = max(throughputs)
        norm_throughputs = [t / max_throughput * 100 for t in throughputs]

        # Quality scores (already 0-1, convert to 0-100)
        qualities = [k5_data[db].get('avg_top1_similarity', 0) * 100 for db in dbs]

        x = np.arange(len(dbs))
        width = 0.35

        # Plot bars
        bars1 = ax4.bar(x - width/2, norm_throughputs, width,
                       label='Throughput (normalized)', color='#5B9BD5', alpha=0.9)
        bars2 = ax4.bar(x + width/2, qualities, width,
                       label='Top-1 Quality (%)', color='#ED7D31', alpha=0.9)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
        ax4.set_title('Performance Summary at K=5', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(dbs, rotation=45, ha='right', fontsize=10)
        ax4.legend(fontsize=10, loc='upper right')
        ax4.set_ylim([0, 110])
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        # Add table overlay
        table_data = []
        for db in dbs:
            data = k5_data[db]
            table_data.append([
                db,
                f"{data['avg_latency_ms']:.2f}",
                f"{data['queries_per_second']:.1f}",
                f"{data.get('avg_top1_similarity', 0):.3f}"
            ])

        # Position table in the bottom-left area, max height at y=45
        table_ax = ax4.inset_axes([0.02, 0.02, 0.55, 0.38])
        table_ax.axis('off')

        table = table_ax.table(
            cellText=table_data,
            colLabels=['Database', 'Latency (ms)', 'QPS', 'Top-1 Quality'],
            cellLoc='center',
            loc='center',
            colWidths=[0.28, 0.24, 0.24, 0.24]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Style header
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white', fontsize=7)

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')
                cell.set_alpha(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("Query Benchmark Comparison Plot Generator")
    print("="*70)

    print("\nLoading query benchmark results...")
    results = load_query_results()

    if not results:
        print("\n❌ No query benchmark data found!")
        return 1

    print(f"\nLoaded data for {len(results)} databases")

    # Save to Nov 24 plots folder
    output_path = project_root / "results/full_suite_20251124_plots/all_databases_comparison_query.png"

    print("\nGenerating comparison plot...")
    create_comparison_plot(results, output_path)

    print(f"\nNote: Using query benchmark data from November 3, 2025")
    print(f"(No query benchmarks were run on November 24)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
