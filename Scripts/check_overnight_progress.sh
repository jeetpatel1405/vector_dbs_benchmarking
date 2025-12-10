#!/bin/bash
#
# Check progress of overnight benchmark run
#

echo "=================================================================="
echo "           OVERNIGHT BENCHMARK - PROGRESS CHECK"
echo "=================================================================="
echo "Checked at: $(date)"
echo ""

# Check if overnight run is active
if pgrep -f "run_all_databases_overnight.sh" > /dev/null; then
    echo "✅ Overnight run is ACTIVE"
    echo ""
else
    echo "⏹️  Overnight run is NOT RUNNING"
    echo ""
fi

# Check individual database processes
echo "Active processes:"
for db in qdrant weaviate milvus faiss chroma opensearch pgvector; do
    if pgrep -f "run_${db}_benchmark.py" > /dev/null; then
        echo "  🔄 $db - RUNNING"
    fi
done
echo ""

# Check completed results
echo "Completed databases:"
COMPLETED=0
for db in qdrant weaviate milvus faiss chroma opensearch pgvector; do
    RESULT_DIR="results/${db}_large_corpus_real_embeddings"
    if [ -f "$RESULT_DIR/results.json" ]; then
        COMPLETED=$((COMPLETED + 1))
        SIZE=$(du -sh "$RESULT_DIR" 2>/dev/null | cut -f1)
        echo "  ✅ $db - Complete ($SIZE)"
    fi
done
echo ""
echo "Progress: $COMPLETED/7 databases complete"
echo ""

# Show recent log entries
if [ -f "overnight_run.log" ]; then
    echo "Recent log entries (last 20 lines):"
    echo "--------------------------------------------------"
    tail -20 overnight_run.log
    echo "--------------------------------------------------"
fi

# Check if Option 1 (Qdrant) is still running
if [ -f "large_corpus_run.log" ]; then
    QDRANT_BATCHES=$(tail -1 large_corpus_run.log | grep -oP 'Batches:\s+\K[0-9]+%' | head -1 || echo "N/A")
    if [ "$QDRANT_BATCHES" != "N/A" ]; then
        echo ""
        echo "Option 1 (Qdrant initial run): $QDRANT_BATCHES complete"
    fi
fi

echo ""
echo "=================================================================="
echo ""
echo "Commands:"
echo "  Monitor overnight log: tail -f overnight_run.log"
echo "  Monitor Option 1: tail -f large_corpus_run.log"
echo "  Check this status: bash Scripts/check_overnight_progress.sh"
echo ""
