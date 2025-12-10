#!/bin/bash
#
# Wait for Option 1 to complete, then automatically start Option 2 (overnight run)
#

echo "=================================================================="
echo "  AUTOMATED SEQUENCER: Option 1 → Option 2"
echo "=================================================================="
echo "This script will:"
echo "  1. Wait for Option 1 (Qdrant with real embeddings) to complete"
echo "  2. Automatically start Option 2 (all 7 databases overnight)"
echo ""
echo "Started: $(date)"
echo "=================================================================="
echo ""

# Monitor Option 1 progress
echo "Waiting for Option 1 to complete..."
echo "(Monitoring large_corpus_run.log for completion signal)"
echo ""

# Wait for the Python process to finish
while pgrep -f "run_qdrant_benchmark.py" > /dev/null; do
    # Show progress every 5 minutes
    if [ -f "large_corpus_run.log" ]; then
        PROGRESS=$(tail -50 large_corpus_run.log | grep "Batches:" | tail -1 | grep -oP '\d+%' | tail -1 || echo "...")
        echo "[$(date +%H:%M:%S)] Option 1 progress: $PROGRESS"
    fi
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=================================================================="
echo "✅ Option 1 COMPLETE!"
echo "=================================================================="
echo "Completed: $(date)"
echo ""

# Brief pause
echo "Waiting 60 seconds before starting Option 2..."
sleep 60

# Check if Option 1 was successful
if [ -f "results/qdrant_large_corpus_real_embeddings/results.json" ]; then
    echo ""
    echo "=================================================================="
    echo "🚀 Starting Option 2: All 7 Databases Overnight Run"
    echo "=================================================================="
    echo "Started: $(date)"
    echo ""

    # Start overnight run
    bash Scripts/run_all_databases_overnight.sh

else
    echo ""
    echo "⚠️  WARNING: Option 1 results not found!"
    echo "   Expected: results/qdrant_large_corpus_real_embeddings/results.json"
    echo ""
    echo "   Option 2 NOT started automatically."
    echo "   Please check Option 1 logs and start manually if needed:"
    echo "   bash Scripts/run_all_databases_overnight.sh"
    echo ""
    exit 1
fi
