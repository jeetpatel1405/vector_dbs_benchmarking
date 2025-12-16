#!/bin/bash
#
# AUTOMATED OVERNIGHT RUN - ALL 7 DATABASES WITH LARGE CORPUS
# ===========================================================
# This script runs all 7 vector databases with the large Wikipedia corpus
# using real sentence-transformers embeddings.
#
# Estimated total time: ~10-12 hours (7 databases × 1.5 hours each)
#
# USAGE:
#   nohup bash Scripts/run_all_databases_overnight.sh > overnight_run.log 2>&1 &
#
# MONITOR:
#   tail -f overnight_run.log
#   bash Scripts/check_overnight_progress.sh
#

set -e  # Exit on error

START_TIME=$(date +%s)

echo "=================================================================="
echo "   LARGE CORPUS BENCHMARK - ALL 7 DATABASES - OVERNIGHT RUN"
echo "=================================================================="
echo "Start time: $(date)"
echo "Output log: overnight_run.log"
echo ""
echo "Databases to benchmark:"
echo "  1. Qdrant"
echo "  2. Weaviate"
echo "  3. Milvus"
echo "  4. FAISS"
echo "  5. Chroma"
echo "  6. OpenSearch"
echo "  7. PGVector"
echo ""
echo "Configuration:"
echo "  - Corpus: Wikipedia (enwiki-latest-pages-articles1.xml)"
echo "  - Chunks: ~2.25 million"
echo "  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384D)"
echo "  - Chunk size: 512 chars, 50 char overlap"
echo "  - Top-K values: [1, 3, 5, 10, 20]"
echo "=================================================================="
echo ""

# Database configurations
DATABASES=("qdrant" "weaviate" "milvus" "faiss" "chroma" "opensearch" "pgvector")
TOTAL=${#DATABASES[@]}
CURRENT=0
FAILED=()
SUCCEEDED=()

# Create results summary file
SUMMARY_FILE="results/overnight_run_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p results
echo "Overnight Run Summary - $(date)" > "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to update each database script
update_database_config() {
    local db=$1
    local script="Scripts/run_${db}_benchmark.py"

    if [ ! -f "$script" ]; then
        echo "⚠️  Warning: Script not found: $script"
        return 1
    fi

    # Create backup
    cp "$script" "${script}.bak"

    # Update output directory
    python3 << EOF
import re
with open('$script', 'r') as f:
    content = f.read()

# Update output_dir
content = re.sub(
    r"'output_dir': '[^']*'",
    "'output_dir': 'results/${db}_large_corpus_real_embeddings'",
    content
)

# Update embedding model
content = re.sub(
    r"'embedding_model': '[^']*'",
    "'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'",
    content
)

# Update embedding type
content = re.sub(
    r"'embedding_type': '[^']*'",
    "'embedding_type': 'sentence-transformers'",
    content
)

with open('$script', 'w') as f:
    f.write(content)
EOF

    return 0
}

# Run benchmark for each database
for db in "${DATABASES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=================================================================="
    echo "  [$CURRENT/$TOTAL] Starting $db benchmark"
    echo "=================================================================="
    echo "Database: $db"
    echo "Started: $(date)"

    DB_START=$(date +%s)

    # Update configuration
    echo "Configuring $db..."
    if ! update_database_config "$db"; then
        echo "❌ Failed to configure $db - skipping"
        FAILED+=("$db (configuration failed)")
        continue
    fi

    # Run benchmark
    echo "Running benchmark..."
    if python Scripts/run_${db}_benchmark.py 2>&1 | tee "results/${db}_large_corpus.log"; then
        DB_END=$(date +%s)
        DB_DURATION=$((DB_END - DB_START))
        DB_MINUTES=$((DB_DURATION / 60))

        echo ""
        echo "✅ $db completed successfully"
        echo "   Duration: ${DB_MINUTES} minutes"
        echo "   Results: results/${db}_large_corpus_real_embeddings/"

        SUCCEEDED+=("$db")

        # Update summary
        echo "[$CURRENT/$TOTAL] ✅ $db - ${DB_MINUTES} min - $(date)" >> "$SUMMARY_FILE"

    else
        DB_END=$(date +%s)
        DB_DURATION=$((DB_END - DB_START))
        DB_MINUTES=$((DB_DURATION / 60))

        echo ""
        echo "❌ $db failed with exit code $?"
        echo "   Duration before failure: ${DB_MINUTES} minutes"
        echo "   Check logs: results/${db}_large_corpus.log"

        FAILED+=("$db")

        # Update summary
        echo "[$CURRENT/$TOTAL] ❌ $db - FAILED after ${DB_MINUTES} min - $(date)" >> "$SUMMARY_FILE"
    fi

    echo "Finished: $(date)"
    echo ""

    # Brief pause between databases
    if [ $CURRENT -lt $TOTAL ]; then
        echo "Waiting 30 seconds before next database..."
        sleep 30
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=================================================================="
echo "              ALL BENCHMARKS COMPLETE"
echo "=================================================================="
echo "End time: $(date)"
echo "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Summary:"
echo "  Succeeded: ${#SUCCEEDED[@]}/${TOTAL}"
echo "  Failed: ${#FAILED[@]}/${TOTAL}"
echo ""

if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    echo "✅ Successful databases:"
    for db in "${SUCCEEDED[@]}"; do
        echo "   - $db"
        echo "     Results: results/${db}_large_corpus_real_embeddings/"
    done
    echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ Failed databases:"
    for db in "${FAILED[@]}"; do
        echo "   - $db"
    done
    echo ""
fi

echo "Summary file: $SUMMARY_FILE"
echo "Individual logs: results/<database>_large_corpus.log"
echo ""
echo "=================================================================="

# Append final summary to summary file
echo "" >> "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "Final Summary" >> "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m" >> "$SUMMARY_FILE"
echo "Succeeded: ${#SUCCEEDED[@]}/${TOTAL}" >> "$SUMMARY_FILE"
echo "Failed: ${#FAILED[@]}/${TOTAL}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

exit 0
