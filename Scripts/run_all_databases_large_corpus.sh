#!/bin/bash
#
# Run ALL 7 databases with large Wikipedia corpus - OVERNIGHT JOB
# Estimated total time: ~3 hours (7 databases × 26 min each)
#
# Usage: nohup bash Scripts/run_all_databases_large_corpus.sh > overnight_run.log 2>&1 &
#

set -e  # Exit on error

echo "========================================="
echo "LARGE CORPUS BENCHMARK - ALL 7 DATABASES"
echo "========================================="
echo "Start time: $(date)"
echo ""

DATABASES=("qdrant" "weaviate" "milvus" "faiss" "chroma" "opensearch" "pgvector")
TOTAL=${#DATABASES[@]}
CURRENT=0

for db in "${DATABASES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================="
    echo "[$CURRENT/$TOTAL] Running $db benchmark"
    echo "========================================="
    echo "Started: $(date)"

    # Update config for each database
    sed -i.bak "s/'output_dir': 'results\/.*'/'output_dir': 'results\/${db}_large_corpus_real_embeddings'/" \
        Scripts/run_${db}_benchmark.py

    sed -i.bak "s/'embedding_model': '.*'/'embedding_model': 'sentence-transformers\/all-MiniLM-L6-v2'/" \
        Scripts/run_${db}_benchmark.py

    sed -i.bak "s/'embedding_type': '.*'/'embedding_type': 'sentence-transformers'/" \
        Scripts/run_${db}_benchmark.py

    # Run benchmark
    if python Scripts/run_${db}_benchmark.py; then
        echo "✅ $db completed successfully"
    else
        echo "❌ $db failed with exit code $?"
    fi

    echo "Finished: $(date)"

    # Brief pause between databases
    sleep 10
done

echo ""
echo "========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results saved in:"
for db in "${DATABASES[@]}"; do
    echo "  - results/${db}_large_corpus_real_embeddings/"
done
