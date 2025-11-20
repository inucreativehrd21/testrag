#!/usr/bin/env python3
"""
Document Analysis Tool for Git/Python RAG
Analyzes chunk sizes, distribution, and optimal hyperparameters
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def analyze_chunks(chunks_path: str, output_dir: str = "artifacts/analysis"):
    """Comprehensive analysis of document chunks"""

    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    print(f"Total chunks: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Filter Git/Python only
    if 'domain' in df.columns:
        df_filtered = df[df['domain'].isin(['git', 'python'])].copy()
        print(f"\nFiltered to Git/Python: {len(df_filtered)} chunks")
    else:
        print("\nWarning: 'domain' column not found, using all chunks")
        df_filtered = df.copy()

    # Calculate text lengths if not present
    if 'length' not in df_filtered.columns:
        df_filtered['length'] = df_filtered['text'].str.len()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === 1. Overall Statistics ===
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    stats = {
        'total_chunks': len(df_filtered),
        'total_characters': df_filtered['length'].sum(),
        'mean_length': df_filtered['length'].mean(),
        'median_length': df_filtered['length'].median(),
        'std_length': df_filtered['length'].std(),
        'min_length': df_filtered['length'].min(),
        'max_length': df_filtered['length'].max(),
        'p25': df_filtered['length'].quantile(0.25),
        'p75': df_filtered['length'].quantile(0.75),
        'p90': df_filtered['length'].quantile(0.90),
        'p95': df_filtered['length'].quantile(0.95),
        'p99': df_filtered['length'].quantile(0.99),
    }

    for key, value in stats.items():
        if 'total' in key or 'chunks' in key:
            print(f"{key:20s}: {value:>15,.0f}")
        else:
            print(f"{key:20s}: {value:>15,.2f} chars")

    # === 2. Domain-wise Statistics ===
    if 'domain' in df_filtered.columns:
        print("\n" + "="*80)
        print("DOMAIN-WISE STATISTICS")
        print("="*80)

        domain_stats = df_filtered.groupby('domain').agg({
            'length': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)
        print(domain_stats)

        # Domain distribution
        domain_counts = df_filtered['domain'].value_counts()
        print("\nDomain Distribution:")
        for domain, count in domain_counts.items():
            pct = count / len(df_filtered) * 100
            print(f"  {domain:10s}: {count:>6,} chunks ({pct:>5.1f}%)")

    # === 3. Visualizations ===
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Git/Python Document Analysis', fontsize=16, fontweight='bold')

    # 3.1 Overall Length Distribution (Histogram)
    ax = axes[0, 0]
    ax.hist(df_filtered['length'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(stats['mean_length'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_length']:.0f}")
    ax.axvline(stats['median_length'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median_length']:.0f}")
    ax.set_xlabel('Chunk Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Chunk Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3.2 Length Distribution by Domain
    if 'domain' in df_filtered.columns:
        ax = axes[0, 1]
        domains = df_filtered['domain'].unique()
        for domain in sorted(domains):
            data = df_filtered[df_filtered['domain'] == domain]['length']
            ax.hist(data, bins=30, alpha=0.5, label=domain, edgecolor='black')
        ax.set_xlabel('Chunk Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Length Distribution by Domain')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3.3 Box Plot by Domain
    if 'domain' in df_filtered.columns:
        ax = axes[1, 0]
        df_filtered.boxplot(column='length', by='domain', ax=ax)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Chunk Length (characters)')
        ax.set_title('Chunk Length Box Plot by Domain')
        plt.sca(ax)
        plt.xticks(rotation=0)

    # 3.4 Cumulative Distribution
    ax = axes[1, 1]
    sorted_lengths = np.sort(df_filtered['length'])
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax.plot(sorted_lengths, cumulative, linewidth=2, color='steelblue')

    # Mark important percentiles
    for p in [50, 75, 90, 95, 99]:
        val = df_filtered['length'].quantile(p/100)
        ax.axvline(val, color='red', linestyle='--', alpha=0.3)
        ax.text(val, p, f'P{p}: {val:.0f}', rotation=90, va='bottom')

    ax.set_xlabel('Chunk Length (characters)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distribution of Chunk Lengths')
    ax.grid(True, alpha=0.3)

    # 3.5 Length vs Index (Temporal pattern)
    ax = axes[2, 0]
    ax.scatter(range(len(df_filtered)), df_filtered['length'], alpha=0.3, s=1, color='steelblue')
    ax.axhline(stats['mean_length'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_length']:.0f}")
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Chunk Length (characters)')
    ax.set_title('Chunk Length Distribution Across Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3.6 Top-k Recommendations
    ax = axes[2, 1]
    ax.axis('off')

    # Calculate recommendations
    total_chunks = len(df_filtered)

    # Top-k recommendations (rule of thumb: sqrt(N) to N/10)
    topk_sqrt = int(np.sqrt(total_chunks))
    topk_10pct = int(total_chunks * 0.1)
    topk_5pct = int(total_chunks * 0.05)
    topk_recommended = int(np.clip(total_chunks * 0.03, 20, 100))  # 3% with bounds

    # Chunk size recommendations
    chunk_optimal = int(stats['p75'])  # 75th percentile
    chunk_conservative = int(stats['median_length'])  # Median

    # Overlap recommendations (15-25% of chunk size)
    overlap_optimal = int(chunk_optimal * 0.20)
    overlap_conservative = int(chunk_conservative * 0.20)

    recommendations = f"""
HYPERPARAMETER RECOMMENDATIONS
{'='*40}

Dataset Size:
  Total Chunks: {total_chunks:,}
  Git/Python Only: {len(df_filtered):,}

Top-k Parameters:
  • sqrt(N):        {topk_sqrt:>4}  (conservative)
  • 10% of N:       {topk_10pct:>4}  (generous)
  • 5% of N:        {topk_5pct:>4}  (balanced)
  • Recommended:    {topk_recommended:>4}  (3% of N)

Current Config:
  • hybrid_dense_top_k:  50
  • hybrid_sparse_top_k: 50
  • rerank_top_k:        5

Recommendation:
  • hybrid_dense_top_k:  {min(topk_recommended, 50)}
  • hybrid_sparse_top_k: {min(topk_recommended, 50)}
  • rerank_top_k:        {max(5, int(topk_recommended * 0.1))}
  • rrf_k:               60 (standard)

Chunking Parameters:
  Current:
    chunk_size:    1024 chars
    chunk_overlap: 150 chars

  Recommended (P75-based):
    chunk_size:    {chunk_optimal} chars
    chunk_overlap: {overlap_optimal} chars (20%)

  Conservative (Median-based):
    chunk_size:    {chunk_conservative} chars
    chunk_overlap: {overlap_conservative} chars (20%)

Rationale:
  • P75 = {stats['p75']:.0f} chars
  • Mean = {stats['mean_length']:.0f} chars
  • Median = {stats['median_length']:.0f} chars
  • 95% chunks < {stats['p95']:.0f} chars

  → chunk_size={chunk_optimal} covers 75% of docs
  → overlap={overlap_optimal} preserves context
"""

    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plot_path = output_path / "document_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {plot_path}")

    # === 4. Save Statistics ===
    stats_output = {
        'overall': stats,
        'domain_wise': domain_counts.to_dict() if 'domain' in df_filtered.columns else {},
        'recommendations': {
            'top_k': {
                'hybrid_dense_top_k': min(topk_recommended, 50),
                'hybrid_sparse_top_k': min(topk_recommended, 50),
                'rerank_top_k': max(5, int(topk_recommended * 0.1)),
                'rrf_k': 60
            },
            'chunking': {
                'optimal': {
                    'chunk_size': chunk_optimal,
                    'chunk_overlap': overlap_optimal
                },
                'conservative': {
                    'chunk_size': chunk_conservative,
                    'chunk_overlap': overlap_conservative
                }
            }
        }
    }

    # Convert numpy/pandas types to native Python types for JSON serialization
    stats_output = convert_to_serializable(stats_output)

    stats_path = output_path / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistics saved to {stats_path}")

    # === 5. Detailed Recommendations ===
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(recommendations)

    return stats_output


def main():
    parser = argparse.ArgumentParser(description="Analyze document chunks for RAG optimization")
    parser.add_argument("--chunks", default="artifacts/chunks.parquet", help="Path to chunks.parquet")
    parser.add_argument("--output", default="artifacts/analysis", help="Output directory")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Error: chunks.parquet not found at {chunks_path}")
        print("\nPlease run data preparation first:")
        print("  python data_prep.py --config config/enhanced.yaml")
        return 1

    try:
        stats = analyze_chunks(str(chunks_path), args.output)
        print("\n" + "="*80)
        print("✓ Analysis complete!")
        print("="*80)
        print(f"\nCheck outputs:")
        print(f"  - Visualization: {args.output}/document_analysis.png")
        print(f"  - Statistics:    {args.output}/statistics.json")
        return 0

    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
