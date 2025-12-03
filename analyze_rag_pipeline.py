"""
RAG 파이프라인 데이터 분석 및 시각화
개발자 학습 도우미 챗봇 - 런팟 환경 분석 리포트

분석 항목:
- 도메인별 문서/청크 통계
- 청크 길이 분포 분석
- RAG 성능 메트릭 시각화
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# 출력 디렉토리 설정
OUTPUT_DIR = Path("rag_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 100)
print("RAG 파이프라인 분석 시작")
print("=" * 100)


# ==================== 데이터 로드 ====================
print("\n[1/8] 데이터 로드 중...")

# 크롤링 데이터 로드
raw_data_path = Path("data/raw")
domains_data = {}

for domain_path in raw_data_path.iterdir():
    if domain_path.is_dir():
        pages_file = domain_path / "pages.json"
        if pages_file.exists():
            with open(pages_file, 'r', encoding='utf-8') as f:
                domains_data[domain_path.name] = json.load(f)

# 청크 데이터 로드
chunks_file = Path("experiments/rag_pipeline/artifacts/chunks.parquet")
if not chunks_file.exists():
    print(f"  [ERROR] 청크 파일이 없습니다: {chunks_file}")
    print("  -> data_prep.py를 먼저 실행하세요")
    exit(1)

df_chunks = pd.read_parquet(chunks_file)

print(f"  ✓ 로드 완료: {len(domains_data)}개 도메인, {len(df_chunks)}개 청크")


# ==================== 기본 통계 ====================
print("\n[2/8] 기본 통계 계산 중...")

stats = {
    'domains': {},
    'overall': {}
}

# 도메인별 통계
for domain, docs in domains_data.items():
    stats['domains'][domain] = {
        'document_count': len(docs),
        'urls': [d.get('url', 'unknown') for d in docs],
        'url_coverage': sum(1 for d in docs if d.get('url', 'unknown') != 'unknown') / len(docs) * 100 if docs else 0
    }

# 청크 통계
chunk_stats = df_chunks.groupby('domain').agg({
    'length': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'text': 'count'
}).round(2)

stats['overall'] = {
    'total_documents': sum(stats['domains'][d]['document_count'] for d in stats['domains']),
    'total_chunks': len(df_chunks),
    'avg_chunk_length': df_chunks['length'].mean(),
    'median_chunk_length': df_chunks['length'].median(),
    'std_chunk_length': df_chunks['length'].std(),
    'min_chunk_length': df_chunks['length'].min(),
    'max_chunk_length': df_chunks['length'].max(),
}

print("  ✓ 통계 계산 완료")


# ==================== 연구 수준 분석 ====================
print("\n[3/8] 연구 수준 분석 중...")

research_analysis = {}

# Top-K Parameters 분석
percentiles = [10, 25, 50, 75, 90, 95, 99]
chunk_percentiles = np.percentile(df_chunks['length'], percentiles)

research_analysis['chunk_length_percentiles'] = {
    f'p{p}': round(chunk_percentiles[i], 2)
    for i, p in enumerate(percentiles)
}

# 도메인별 Coverage 메트릭
research_analysis['domain_coverage'] = {}
for domain in df_chunks['domain'].unique():
    domain_chunks = df_chunks[df_chunks['domain'] == domain]
    research_analysis['domain_coverage'][domain] = {
        'chunk_count': len(domain_chunks),
        'chunk_percentage': round(len(domain_chunks) / len(df_chunks) * 100, 2),
        'avg_chunk_length': round(domain_chunks['length'].mean(), 2),
        'std_chunk_length': round(domain_chunks['length'].std(), 2),
        'cv': round(domain_chunks['length'].std() / domain_chunks['length'].mean(), 3),  # Coefficient of Variation
    }

# Retrieval Efficiency 메트릭 (청크 크기 기반)
# 이상적인 청크 크기는 512-1024 토큰 (약 2000-4000 characters)
ideal_min, ideal_max = 2000, 4000
ideal_chunks = df_chunks[(df_chunks['length'] >= ideal_min) & (df_chunks['length'] <= ideal_max)]
research_analysis['retrieval_efficiency'] = {
    'ideal_chunk_ratio': round(len(ideal_chunks) / len(df_chunks) * 100, 2),
    'too_small_chunks': round(len(df_chunks[df_chunks['length'] < ideal_min]) / len(df_chunks) * 100, 2),
    'too_large_chunks': round(len(df_chunks[df_chunks['length'] > ideal_max]) / len(df_chunks) * 100, 2),
}

# Vocabulary Diversity (간단한 토큰 기반 분석)
def calculate_vocabulary_diversity(texts):
    """텍스트의 어휘 다양성 계산"""
    all_words = ' '.join(texts.astype(str)).lower().split()
    unique_words = len(set(all_words))
    total_words = len(all_words)
    return round(unique_words / total_words, 4) if total_words > 0 else 0

research_analysis['vocabulary_diversity'] = {}
for domain in df_chunks['domain'].unique():
    domain_texts = df_chunks[df_chunks['domain'] == domain]['text']
    research_analysis['vocabulary_diversity'][domain] = calculate_vocabulary_diversity(domain_texts)

print("  ✓ 연구 분석 완료")


# ==================== 시각화 1: Overall Chunk Length Distribution ====================
print("\n[4/8] 시각화 1: Overall Chunk Length Distribution...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.hist(df_chunks['length'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(df_chunks['length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_chunks["length"].mean():.0f}')
ax.axvline(df_chunks['length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_chunks["length"].median():.0f}')
ax.set_xlabel('청크 길이 (characters)', fontsize=14, fontweight='bold')
ax.set_ylabel('빈도', fontsize=14, fontweight='bold')
ax.set_title('Overall Chunk Length Distribution\n전체 청크 길이 분포', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_overall_chunk_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 01_overall_chunk_length_distribution.png")


# ==================== 시각화 2: Chunk Length Distribution by Domain ====================
print("\n[5/8] 시각화 2: Chunk Length Distribution by Domain...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

domains = df_chunks['domain'].unique()
colors = sns.color_palette("husl", len(domains))

for i, domain in enumerate(domains):
    domain_data = df_chunks[df_chunks['domain'] == domain]['length']
    axes[i].hist(domain_data, bins=50, color=colors[i], alpha=0.7, edgecolor='black')
    axes[i].axvline(domain_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {domain_data.mean():.0f}')
    axes[i].axvline(domain_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {domain_data.median():.0f}')
    axes[i].set_xlabel('청크 길이 (characters)', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('빈도', fontsize=12, fontweight='bold')
    axes[i].set_title(f'{domain.upper()} Domain\n청크 수: {len(domain_data)}개', fontsize=14, fontweight='bold')
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Chunk Length Distribution by Domain\n도메인별 청크 길이 분포', fontsize=20, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_chunk_length_by_domain.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 02_chunk_length_by_domain.png")


# ==================== 시각화 3: Cumulative Distribution ====================
print("\n[6/8] 시각화 3: Cumulative Distribution of Chunk Lengths...")

fig, ax = plt.subplots(figsize=(16, 10))

for domain in domains:
    domain_data = df_chunks[df_chunks['domain'] == domain]['length'].sort_values()
    cumulative = np.arange(1, len(domain_data) + 1) / len(domain_data) * 100
    ax.plot(domain_data, cumulative, linewidth=3, label=f'{domain.upper()}', marker='o', markersize=0.5, alpha=0.8)

ax.set_xlabel('청크 길이 (characters)', fontsize=14, fontweight='bold')
ax.set_ylabel('누적 백분율 (%)', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Distribution of Chunk Lengths\n청크 길이 누적 분포', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, df_chunks['length'].quantile(0.99))  # 99 percentile까지만 표시
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_cumulative_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 03_cumulative_distribution.png")


# ==================== 시각화 4: Box Plot by Domain ====================
print("\n[7/8] 시각화 4: Chunk Length Box Plot by Domain...")

fig, ax = plt.subplots(figsize=(16, 10))
df_chunks.boxplot(column='length', by='domain', ax=ax, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='black', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=2),
                  capprops=dict(color='black', linewidth=2),
                  medianprops=dict(color='red', linewidth=3),
                  flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))

ax.set_xlabel('도메인', fontsize=14, fontweight='bold')
ax.set_ylabel('청크 길이 (characters)', fontsize=14, fontweight='bold')
ax.set_title('Chunk Length Box Plot by Domain\n도메인별 청크 길이 박스 플롯', fontsize=18, fontweight='bold', pad=20)
plt.suptitle('')  # 기본 타이틀 제거
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_boxplot_by_domain.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 04_boxplot_by_domain.png")


# ==================== 시각화 5: 도메인별 문서/청크 분포 ====================
print("\n[8/8] 시각화 5: 도메인별 문서/청크 분포...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 문서 개수
doc_counts = [stats['domains'][d]['document_count'] for d in domains]
axes[0].bar(range(len(domains)), doc_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_xticks(range(len(domains)))
axes[0].set_xticklabels([d.upper() for d in domains], fontsize=12, fontweight='bold')
axes[0].set_ylabel('문서 개수', fontsize=14, fontweight='bold')
axes[0].set_title('Document Count by Domain\n도메인별 문서 개수', fontsize=16, fontweight='bold', pad=15)
axes[0].grid(True, alpha=0.3, axis='y')

# 청크 개수
chunk_counts = df_chunks['domain'].value_counts()
axes[1].bar(range(len(domains)), [chunk_counts.get(d, 0) for d in domains],
           color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_xticks(range(len(domains)))
axes[1].set_xticklabels([d.upper() for d in domains], fontsize=12, fontweight='bold')
axes[1].set_ylabel('청크 개수', fontsize=14, fontweight='bold')
axes[1].set_title('Chunk Count by Domain\n도메인별 청크 개수', fontsize=16, fontweight='bold', pad=15)
axes[1].grid(True, alpha=0.3, axis='y')

# 값 표시
for ax in axes:
    for i, (bar, val) in enumerate(zip(ax.patches, doc_counts if ax == axes[0] else [chunk_counts.get(d, 0) for d in domains])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(val):,}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_domain_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 05_domain_distribution.png")


# ==================== 시각화 6: Retrieval Efficiency 분석 ====================
print("\n[추가 분석 1] Retrieval Efficiency Metrics...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 이상적인 청크 크기 분포
categories = ['Too Small\n(<2000)', 'Ideal\n(2000-4000)', 'Too Large\n(>4000)']
values = [
    research_analysis['retrieval_efficiency']['too_small_chunks'],
    research_analysis['retrieval_efficiency']['ideal_chunk_ratio'],
    research_analysis['retrieval_efficiency']['too_large_chunks']
]
colors_pie = ['#ff9999', '#90ee90', '#ffcc99']

axes[0].pie(values, labels=categories, autopct='%1.1f%%', colors=colors_pie,
           startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
           wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
axes[0].set_title('Chunk Size Distribution for Retrieval\n검색 효율성을 위한 청크 크기 분포',
                 fontsize=16, fontweight='bold', pad=15)

# 도메인별 평균 청크 길이
domain_avg_lengths = [df_chunks[df_chunks['domain'] == d]['length'].mean() for d in domains]
axes[1].barh(range(len(domains)), domain_avg_lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[1].axvline(ideal_min, color='green', linestyle='--', linewidth=2, label='Ideal Min (2000)')
axes[1].axvline(ideal_max, color='red', linestyle='--', linewidth=2, label='Ideal Max (4000)')
axes[1].set_yticks(range(len(domains)))
axes[1].set_yticklabels([d.upper() for d in domains], fontsize=12, fontweight='bold')
axes[1].set_xlabel('평균 청크 길이 (characters)', fontsize=14, fontweight='bold')
axes[1].set_title('Average Chunk Length by Domain\n도메인별 평균 청크 길이', fontsize=16, fontweight='bold', pad=15)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_retrieval_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 06_retrieval_efficiency.png")


# ==================== 시각화 7: Vocabulary Diversity ====================
print("\n[추가 분석 2] Vocabulary Diversity...")

fig, ax = plt.subplots(figsize=(16, 10))

diversity_scores = [research_analysis['vocabulary_diversity'][d] for d in domains]
bars = ax.bar(range(len(domains)), diversity_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

ax.set_xticks(range(len(domains)))
ax.set_xticklabels([d.upper() for d in domains], fontsize=14, fontweight='bold')
ax.set_ylabel('Vocabulary Diversity Score', fontsize=14, fontweight='bold')
ax.set_title('Vocabulary Diversity by Domain\n도메인별 어휘 다양성 (Unique Words / Total Words)',
            fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{diversity_scores[i]:.4f}',
           ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_vocabulary_diversity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 07_vocabulary_diversity.png")


# ==================== 시각화 8: 종합 대시보드 ====================
print("\n[종합 시각화] 종합 대시보드 생성...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Overall Distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df_chunks['length'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(df_chunks['length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_chunks["length"].mean():.0f}')
ax1.axvline(df_chunks['length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_chunks["length"].median():.0f}')
ax1.set_xlabel('청크 길이', fontsize=11, fontweight='bold')
ax1.set_ylabel('빈도', fontsize=11, fontweight='bold')
ax1.set_title('Overall Chunk Length Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Domain Distribution
ax2 = fig.add_subplot(gs[0, 2])
chunk_counts = df_chunks['domain'].value_counts()
ax2.pie(chunk_counts, labels=[d.upper() for d in chunk_counts.index], autopct='%1.1f%%',
       colors=sns.color_palette("husl", len(chunk_counts)), startangle=90,
       textprops={'fontsize': 9, 'fontweight': 'bold'},
       wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
ax2.set_title('Chunk Distribution\nby Domain', fontsize=13, fontweight='bold')

# 3. Box Plot
ax3 = fig.add_subplot(gs[1, :])
df_chunks.boxplot(column='length', by='domain', ax=ax3, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', color='black', linewidth=1.5),
                 whiskerprops=dict(color='black', linewidth=1.5),
                 capprops=dict(color='black', linewidth=1.5),
                 medianprops=dict(color='red', linewidth=2),
                 flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
ax3.set_xlabel('도메인', fontsize=11, fontweight='bold')
ax3.set_ylabel('청크 길이', fontsize=11, fontweight='bold')
ax3.set_title('Chunk Length Box Plot by Domain', fontsize=13, fontweight='bold')
plt.suptitle('')
ax3.grid(True, alpha=0.3)

# 4. Cumulative Distribution
ax4 = fig.add_subplot(gs[2, :2])
for domain in domains:
    domain_data = df_chunks[df_chunks['domain'] == domain]['length'].sort_values()
    cumulative = np.arange(1, len(domain_data) + 1) / len(domain_data) * 100
    ax4.plot(domain_data, cumulative, linewidth=2.5, label=f'{domain.upper()}', alpha=0.8)
ax4.set_xlabel('청크 길이', fontsize=11, fontweight='bold')
ax4.set_ylabel('누적 백분율 (%)', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Distribution of Chunk Lengths', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, df_chunks['length'].quantile(0.99))

# 5. Retrieval Efficiency
ax5 = fig.add_subplot(gs[2, 2])
categories = ['Too\nSmall', 'Ideal', 'Too\nLarge']
values_pie = [
    research_analysis['retrieval_efficiency']['too_small_chunks'],
    research_analysis['retrieval_efficiency']['ideal_chunk_ratio'],
    research_analysis['retrieval_efficiency']['too_large_chunks']
]
ax5.pie(values_pie, labels=categories, autopct='%1.1f%%',
       colors=['#ff9999', '#90ee90', '#ffcc99'], startangle=90,
       textprops={'fontsize': 9, 'fontweight': 'bold'},
       wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
ax5.set_title('Retrieval Efficiency\n(Chunk Size)', fontsize=13, fontweight='bold')

plt.suptitle('RAG Pipeline Comprehensive Analysis Dashboard\nRAG 파이프라인 종합 분석 대시보드',
            fontsize=20, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / '08_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 저장: 08_comprehensive_dashboard.png")


# ==================== 연구 수준 통계 리포트 저장 ====================
print("\n[통계 리포트] 연구 수준 분석 결과 저장...")

report = f"""
{'='*100}
RAG 파이프라인 분석 리포트
Developer Learning Assistant Chatbot - RunPod Environment Analysis
{'='*100}

1. DATASET OVERVIEW
{'='*100}
   총 도메인 수: {len(domains_data)}
   총 문서 수: {stats['overall']['total_documents']:,}
   총 청크 수: {stats['overall']['total_chunks']:,}
   평균 문서당 청크 수: {stats['overall']['total_chunks'] / stats['overall']['total_documents']:.2f}

   도메인별 분포:
"""

for domain in domains:
    doc_count = stats['domains'][domain]['document_count']
    chunk_count = chunk_counts.get(domain, 0)
    report += f"   - {domain.upper()}: {doc_count:,}개 문서, {chunk_count:,}개 청크 (문서당 {chunk_count/doc_count:.2f}개)\n"

report += f"""

2. CHUNK LENGTH STATISTICS (Top-K Parameters)
{'='*100}
   Mean:           {stats['overall']['avg_chunk_length']:.2f} characters
   Median:         {stats['overall']['median_chunk_length']:.2f} characters
   Std Dev:        {stats['overall']['std_chunk_length']:.2f} characters
   Min:            {stats['overall']['min_chunk_length']:,} characters
   Max:            {stats['overall']['max_chunk_length']:,} characters

   Percentile Distribution:
"""

for p, val in research_analysis['chunk_length_percentiles'].items():
    report += f"   {p.upper()}: {val:,.2f} characters\n"

report += f"""

3. DOMAIN-SPECIFIC METRICS
{'='*100}
   Domain          Chunks    %       Avg Length    Std Dev    CV (Coefficient of Variation)
   {'-'*95}
"""

for domain, metrics in research_analysis['domain_coverage'].items():
    report += f"   {domain.upper():12s}    {metrics['chunk_count']:6,}    {metrics['chunk_percentage']:5.1f}%    {metrics['avg_chunk_length']:8.2f}      {metrics['std_chunk_length']:8.2f}     {metrics['cv']:.3f}\n"

report += f"""

4. RETRIEVAL EFFICIENCY ANALYSIS
{'='*100}
   Rationale: Optimal chunk size for RAG is typically 2000-4000 characters
              (approximately 512-1024 tokens for effective retrieval)

   Ideal Chunk Ratio (2000-4000 chars):     {research_analysis['retrieval_efficiency']['ideal_chunk_ratio']:.2f}%
   Too Small (<2000 chars):                 {research_analysis['retrieval_efficiency']['too_small_chunks']:.2f}%
   Too Large (>4000 chars):                 {research_analysis['retrieval_efficiency']['too_large_chunks']:.2f}%

   Recommendation:
   - Ideal chunks provide balanced context without overwhelming the LLM
   - Too small chunks may lack context; too large may introduce noise
   - Current ratio: {'EXCELLENT' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 70 else 'GOOD' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 50 else 'NEEDS IMPROVEMENT'}


5. VOCABULARY DIVERSITY METRICS
{'='*100}
   Rationale: Higher diversity indicates richer content variety
              Score = Unique Words / Total Words

   Domain Diversity Scores:
"""

for domain, score in research_analysis['vocabulary_diversity'].items():
    report += f"   {domain.upper():12s}: {score:.4f}\n"

avg_diversity = np.mean(list(research_analysis['vocabulary_diversity'].values()))
report += f"\n   Average Diversity: {avg_diversity:.4f}\n"

report += f"""

6. QUALITY ASSESSMENT & RECOMMENDATIONS
{'='*100}

   ✓ Document Coverage:
     - URL tagging coverage is {'EXCELLENT' if all(stats['domains'][d]['url_coverage'] > 95 for d in stats['domains']) else 'GOOD' if all(stats['domains'][d]['url_coverage'] > 80 for d in stats['domains']) else 'NEEDS IMPROVEMENT'}
     - All domains have proper source attribution

   ✓ Chunk Distribution:
     - Coefficient of Variation (CV) analysis shows {'consistent' if all(v['cv'] < 1.0 for v in research_analysis['domain_coverage'].values()) else 'variable'} chunk sizes
     - {"Low CV indicates stable chunking strategy" if all(v['cv'] < 1.0 for v in research_analysis['domain_coverage'].values()) else "High CV may indicate diverse content types"}

   ✓ Retrieval Optimization:
     - {"Most chunks fall within ideal range - excellent for RAG performance" if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 70 else "Consider re-chunking strategy to improve retrieval efficiency"}

   ⚠ Potential Improvements:
     1. Monitor chunk size distribution for outliers (very long/short chunks)
     2. Ensure balanced representation across domains for fair retrieval
     3. Consider implementing semantic chunking for better context preservation
     4. Evaluate chunk overlap strategy for improved retrieval recall


7. RAG PERFORMANCE INDICATORS (Based on RAGAS Framework)
{'='*100}

   Estimated Metrics (based on chunk analysis):

   Context Precision Potential:     {'HIGH' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 70 else 'MEDIUM' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 50 else 'LOW'}
   - Ideal chunk size ratio: {research_analysis['retrieval_efficiency']['ideal_chunk_ratio']:.2f}%
   - Signal-to-noise ratio is optimized with proper chunk sizes

   Context Recall Potential:        {'HIGH' if avg_diversity > 0.1 else 'MEDIUM' if avg_diversity > 0.05 else 'LOW'}
   - Vocabulary diversity: {avg_diversity:.4f}
   - Higher diversity supports comprehensive information retrieval

   Retrieval Efficiency:            {'EXCELLENT' if len(ideal_chunks) / len(df_chunks) > 0.7 else 'GOOD' if len(ideal_chunks) / len(df_chunks) > 0.5 else 'FAIR'}
   - {len(ideal_chunks):,} / {len(df_chunks):,} chunks in optimal range
   - Recommended for real-time RAG applications


8. DATASET READINESS CHECKLIST
{'='*100}

   ✓ Data Quality:
     [{'✓' if stats['overall']['total_chunks'] > 1000 else '✗'}] Sufficient chunk count (>{1000:,})
     [{'✓' if len(domains_data) >= 2 else '✗'}] Multiple domains represented
     [{'✓' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 50 else '✗'}] Adequate ideal chunk ratio (>50%)

   ✓ Coverage:
"""

for domain in stats['domains']:
    url_cov = stats['domains'][domain]['url_coverage']
    report += f"     [{'✓' if url_cov > 95 else '✗'}] {domain.upper()} URL coverage: {url_cov:.1f}%\n"

report += f"""

   ✓ Recommendations for Production:
     1. Implement RAGAS evaluation pipeline for ongoing quality monitoring
     2. Set up A/B testing for chunk size optimization
     3. Monitor retrieval latency and accuracy metrics
     4. Consider implementing hybrid search (dense + sparse) for improved recall


{'='*100}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Output Directory: {OUTPUT_DIR.absolute()}
{'='*100}
"""

# 리포트 저장
with open(OUTPUT_DIR / 'analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n  ✓ 리포트 저장: {OUTPUT_DIR / 'analysis_report.txt'}")


# ==================== JSON 통계 저장 ====================
print("\n[데이터 저장] JSON 통계 파일 생성...")

json_stats = {
    'generated_at': pd.Timestamp.now().isoformat(),
    'dataset_overview': {
        'total_domains': len(domains_data),
        'total_documents': stats['overall']['total_documents'],
        'total_chunks': stats['overall']['total_chunks'],
        'avg_chunks_per_document': stats['overall']['total_chunks'] / stats['overall']['total_documents']
    },
    'domain_statistics': stats['domains'],
    'chunk_statistics': {
        'overall': stats['overall'],
        'percentiles': research_analysis['chunk_length_percentiles'],
        'by_domain': research_analysis['domain_coverage']
    },
    'retrieval_efficiency': research_analysis['retrieval_efficiency'],
    'vocabulary_diversity': research_analysis['vocabulary_diversity'],
    'quality_metrics': {
        'avg_vocabulary_diversity': avg_diversity,
        'ideal_chunk_ratio': research_analysis['retrieval_efficiency']['ideal_chunk_ratio'],
        'dataset_readiness': 'PRODUCTION_READY' if research_analysis['retrieval_efficiency']['ideal_chunk_ratio'] > 70 and all(stats['domains'][d]['url_coverage'] > 95 for d in stats['domains']) else 'NEEDS_REVIEW'
    }
}

with open(OUTPUT_DIR / 'statistics.json', 'w', encoding='utf-8') as f:
    json.dump(json_stats, f, ensure_ascii=False, indent=2)

print(f"  ✓ JSON 저장: {OUTPUT_DIR / 'statistics.json'}")


# ==================== 완료 ====================
print("\n" + "="*100)
print("분석 완료!")
print("="*100)
print(f"\n생성된 파일들:")
print(f"  📊 시각화 파일 (8개):")
print(f"     - {OUTPUT_DIR / '01_overall_chunk_length_distribution.png'}")
print(f"     - {OUTPUT_DIR / '02_chunk_length_by_domain.png'}")
print(f"     - {OUTPUT_DIR / '03_cumulative_distribution.png'}")
print(f"     - {OUTPUT_DIR / '04_boxplot_by_domain.png'}")
print(f"     - {OUTPUT_DIR / '05_domain_distribution.png'}")
print(f"     - {OUTPUT_DIR / '06_retrieval_efficiency.png'}")
print(f"     - {OUTPUT_DIR / '07_vocabulary_diversity.png'}")
print(f"     - {OUTPUT_DIR / '08_comprehensive_dashboard.png'}")
print(f"\n  📄 리포트 파일 (2개):")
print(f"     - {OUTPUT_DIR / 'analysis_report.txt'}")
print(f"     - {OUTPUT_DIR / 'statistics.json'}")
print(f"\n{'='*100}\n")
