#!/bin/bash
# Export artifacts from Runpod to local machine
# Usage: bash export_artifacts.sh

set -e

echo "==========================================="
echo "Artifacts Export Script"
echo "==========================================="
echo ""

# Check if running in the correct directory
if [ ! -d "artifacts" ]; then
    echo "Error: artifacts/ directory not found!"
    echo "Please run this script from experiments/rag_pipeline/"
    exit 1
fi

# Create export directory
EXPORT_DIR="artifacts_export_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPORT_DIR"

echo "[1/4] Checking artifacts..."

# Check what exists
CHROMA_EXISTS=false
CHUNKS_EXISTS=false
RAGAS_EXISTS=false

if [ -d "artifacts/chroma_db" ]; then
    CHROMA_SIZE=$(du -sh artifacts/chroma_db | awk '{print $1}')
    echo "  ✓ ChromaDB found (${CHROMA_SIZE})"
    CHROMA_EXISTS=true
fi

if [ -f "artifacts/chunks.parquet" ]; then
    CHUNKS_SIZE=$(du -sh artifacts/chunks.parquet | awk '{print $1}')
    echo "  ✓ Chunks file found (${CHUNKS_SIZE})"
    CHUNKS_EXISTS=true
fi

if [ -d "artifacts/ragas_evals" ] && [ "$(ls -A artifacts/ragas_evals 2>/dev/null)" ]; then
    RAGAS_COUNT=$(ls artifacts/ragas_evals/*.json 2>/dev/null | wc -l)
    echo "  ✓ Ragas results found (${RAGAS_COUNT} evaluations)"
    RAGAS_EXISTS=true
fi

echo ""
echo "[2/4] Creating archives..."

# Archive 1: Vector DB and chunks (essential)
if [ "$CHROMA_EXISTS" = true ] || [ "$CHUNKS_EXISTS" = true ]; then
    echo "  Creating vector_db_backup.tar.gz..."
    tar -czf "${EXPORT_DIR}/vector_db_backup.tar.gz" \
        $([ "$CHUNKS_EXISTS" = true ] && echo "artifacts/chunks.parquet") \
        $([ "$CHROMA_EXISTS" = true ] && echo "artifacts/chroma_db") \
        2>/dev/null || true

    VECT_SIZE=$(du -sh "${EXPORT_DIR}/vector_db_backup.tar.gz" | awk '{print $1}')
    echo "    ✓ vector_db_backup.tar.gz (${VECT_SIZE})"
fi

# Archive 2: Ragas results (small)
if [ "$RAGAS_EXISTS" = true ]; then
    echo "  Creating ragas_results.tar.gz..."
    tar -czf "${EXPORT_DIR}/ragas_results.tar.gz" artifacts/ragas_evals/

    RAGAS_SIZE=$(du -sh "${EXPORT_DIR}/ragas_results.tar.gz" | awk '{print $1}')
    echo "    ✓ ragas_results.tar.gz (${RAGAS_SIZE})"
fi

# Archive 3: All artifacts (backup)
echo "  Creating full_artifacts_backup.tar.gz..."
tar -czf "${EXPORT_DIR}/full_artifacts_backup.tar.gz" artifacts/
FULL_SIZE=$(du -sh "${EXPORT_DIR}/full_artifacts_backup.tar.gz" | awk '{print $1}')
echo "    ✓ full_artifacts_backup.tar.gz (${FULL_SIZE})"

echo ""
echo "[3/4] Creating metadata file..."

cat > "${EXPORT_DIR}/EXPORT_INFO.txt" << EOF
=========================================
Artifacts Export Information
=========================================

Export Date: $(date)
Export Location: $(pwd)
Hostname: $(hostname)

Contents:
---------
EOF

if [ "$CHROMA_EXISTS" = true ]; then
    echo "✓ ChromaDB vector index" >> "${EXPORT_DIR}/EXPORT_INFO.txt"
fi

if [ "$CHUNKS_EXISTS" = true ]; then
    CHUNK_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('artifacts/chunks.parquet')))" 2>/dev/null || echo "unknown")
    echo "✓ Chunks file (${CHUNK_COUNT} chunks)" >> "${EXPORT_DIR}/EXPORT_INFO.txt"
fi

if [ "$RAGAS_EXISTS" = true ]; then
    echo "✓ Ragas evaluation results (${RAGAS_COUNT} files)" >> "${EXPORT_DIR}/EXPORT_INFO.txt"
fi

cat >> "${EXPORT_DIR}/EXPORT_INFO.txt" << EOF

Files in this export:
---------------------
$(ls -lh ${EXPORT_DIR}/*.tar.gz | awk '{print $9, "("$5")"}')

To restore on local machine:
----------------------------
1. Download the export directory to your local machine
2. Extract vector_db_backup.tar.gz:
   tar -xzf vector_db_backup.tar.gz -C /path/to/testrag/experiments/rag_pipeline/

3. Extract ragas_results.tar.gz (optional):
   tar -xzf ragas_results.tar.gz -C /path/to/testrag/experiments/rag_pipeline/

Verification:
-------------
After extraction, verify the files:
- artifacts/chunks.parquet should exist
- artifacts/chroma_db/ directory should have index files
- artifacts/ragas_evals/ should have evaluation results

Then you can use the pipeline locally without re-indexing!
EOF

cat "${EXPORT_DIR}/EXPORT_INFO.txt"

echo ""
echo "[4/4] Export summary"
echo "==========================================="
echo "Export directory: ${EXPORT_DIR}/"
echo ""
echo "Total size: $(du -sh ${EXPORT_DIR} | awk '{print $1}')"
echo ""
echo "Files created:"
ls -lh ${EXPORT_DIR}/ | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "==========================================="
echo "✓ Export complete!"
echo ""
echo "Next steps:"
echo "  1. Download ${EXPORT_DIR}/ to your local machine"
echo "  2. Extract the archives in your local testrag directory"
echo "  3. Verify with: python answerer.py 'Test question'"
echo ""
echo "Download command (run from local machine):"
echo "  scp -r root@<runpod-ip>:/workspace/testrag/experiments/rag_pipeline/${EXPORT_DIR} ."
echo ""
