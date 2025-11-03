#!/usr/bin/env python3
"""
Create comprehensive comparison visualization for all 7 vector databases.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(database_name: str) -> dict:
    """Load benchmark results for a specific database."""
    results_path = Path(f"results/{database_name}_experiment_001/results.json")
    with open(results_path, 'r') as f:
        return json.load(f)


def create_comparison_visualization(databases: list):
    """Create comprehensive comparison charts for all databases."""

    # Load all results
    all_results = {}
    for db in databases:
        all_results[db] = load_results(db)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vector Database Benchmark Comparison (7 Databases)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define colors for each database
    colors = {
        'faiss': '#FF6B6B',
        'chroma': '#4ECDC4',
        'qdrant': '#45B7D1',
        'pgvector': '#96CEB4',
        'weaviate': '#FFEAA7',
        'milvus': '#DDA15E',
        'opensearch': '#A8DADC'
    }

    # Extract data for plotting
    top_k_values = all_results[databases[0]]['config']['top_k_values']

    # 1. Average Latency Comparison (top-left)
    ax1 = axes[0, 0]
    for db in databases:
        latencies = [r['avg_latency_ms'] for r in all_results[db]['query_results']]
        ax1.plot(top_k_values, latencies, marker='o', label=db.upper(),
                color=colors[db], linewidth=2, markersize=6)

    ax1.set_xlabel('Top K', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Latency (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Query Latency by Top K', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    ax1.set_xticks(top_k_values)
    ax1.set_xticklabels([str(k) for k in top_k_values])

    # 2. Throughput Comparison (top-right)
    ax2 = axes[0, 1]
    for db in databases:
        throughputs = [r['queries_per_second'] for r in all_results[db]['query_results']]
        ax2.plot(top_k_values, throughputs, marker='s', label=db.upper(),
                color=colors[db], linewidth=2, markersize=6)

    ax2.set_xlabel('Top K', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Queries per Second', fontsize=11, fontweight='bold')
    ax2.set_title('Query Throughput by Top K', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xticks(top_k_values)
    ax2.set_xticklabels([str(k) for k in top_k_values])

    # 3. Average Similarity Trend (bottom-left)
    ax3 = axes[1, 0]
    for db in databases:
        avg_similarities = [r['avg_similarity'] for r in all_results[db]['query_results']]
        ax3.plot(top_k_values, avg_similarities, marker='^', label=db.upper(),
                color=colors[db], linewidth=2, markersize=6)

    ax3.set_xlabel('Top K', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Similarity Score', fontsize=11, fontweight='bold')
    ax3.set_title('Quality Metrics: Average Similarity Across Top K Results',
                 fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xscale('log')
    ax3.set_xticks(top_k_values)
    ax3.set_xticklabels([str(k) for k in top_k_values])
    ax3.set_ylim(0.3, 0.9)  # Focus on the relevant range

    # 4. Performance Summary at K=5 (bottom-right)
    ax4 = axes[1, 1]

    # Extract K=5 metrics for each database
    k5_data = {}
    for db in databases:
        k5_results = next(r for r in all_results[db]['query_results'] if r['top_k'] == 5)
        k5_data[db] = {
            'latency': k5_results['avg_latency_ms'],
            'qps': k5_results['queries_per_second'],
            'avg_sim': k5_results['avg_similarity'],
            'top1_sim': k5_results['avg_top1_similarity']
        }

    # Create grouped bar chart
    x_pos = np.arange(len(databases))
    width = 0.35

    # Normalize metrics for comparison (0-100 scale)
    max_qps = max(d['qps'] for d in k5_data.values())
    normalized_qps = [k5_data[db]['qps'] / max_qps * 100 for db in databases]
    normalized_quality = [k5_data[db]['top1_sim'] * 100 for db in databases]

    bars1 = ax4.bar(x_pos - width/2, normalized_qps, width,
                    label='Throughput (normalized)', alpha=0.8, color='steelblue')
    bars2 = ax4.bar(x_pos + width/2, normalized_quality, width,
                    label='Top-1 Quality (%)', alpha=0.8, color='coral')

    ax4.set_xlabel('Database', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
    ax4.set_title('Performance Summary at K=5', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([db.upper() for db in databases], rotation=45, ha='right')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.set_ylim(0, 110)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add performance table below the chart
    table_data = []
    for db in databases:
        k5 = k5_data[db]
        table_data.append([
            db.upper(),
            f"{k5['latency']:.2f}",
            f"{k5['qps']:.1f}",
            f"{k5['top1_sim']:.3f}"
        ])

    # Create table
    table_ax = fig.add_axes([0.55, 0.08, 0.4, 0.25])
    table_ax.axis('tight')
    table_ax.axis('off')

    table = table_ax.table(cellText=table_data,
                          colLabels=['Database', 'Latency (ms)', 'QPS', 'Top-1 Quality'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.15, 0.15, 0.1, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')

    # Color code rows by database
    for i, db in enumerate(databases, 1):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(colors[db])
            cell.set_alpha(0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save the figure
    output_path = Path("results/all_databases_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comprehensive comparison saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY (K=5)")
    print("="*80)

    for db in databases:
        k5 = k5_data[db]
        print(f"\n{db.upper()}:")
        print(f"  Latency:     {k5['latency']:>8.2f} ms")
        print(f"  Throughput:  {k5['qps']:>8.1f} QPS")
        print(f"  Avg Quality: {k5['avg_sim']:>8.3f}")
        print(f"  Top-1 Quality: {k5['top1_sim']:>6.3f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)

    # Find best performers
    best_speed_db = max(databases, key=lambda db: k5_data[db]['qps'])
    best_quality_db = max(databases, key=lambda db: k5_data[db]['top1_sim'])
    worst_quality_db = min(databases, key=lambda db: k5_data[db]['top1_sim'])

    print(f"🏆 Fastest: {best_speed_db.upper()} ({k5_data[best_speed_db]['qps']:.1f} QPS)")
    print(f"🏆 Best Quality: {best_quality_db.upper()} ({k5_data[best_quality_db]['top1_sim']:.3f} top-1 similarity)")
    print(f"⚠️  Lowest Quality: {worst_quality_db.upper()} ({k5_data[worst_quality_db]['top1_sim']:.3f} top-1 similarity)")

    # Quality trend analysis
    print("\n📊 Quality Trends:")
    for db in databases:
        sims = [r['avg_similarity'] for r in all_results[db]['query_results']]
        trend = "↗️ INCREASES" if sims[-1] > sims[0] else "↘️ DECREASES"
        print(f"  {db.upper():12s}: {sims[0]:.3f} → {sims[-1]:.3f} {trend}")

    print("\n" + "="*80)


def main():
    databases = ['faiss', 'chroma', 'qdrant', 'pgvector', 'weaviate', 'milvus', 'opensearch']

    print("Creating comprehensive 7-database comparison...")
    print(f"Databases: {', '.join(db.upper() for db in databases)}\n")

    create_comparison_visualization(databases)

    print("\n✨ Comparison complete!")


if __name__ == "__main__":
    main()
