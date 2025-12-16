#!/bin/bash
# Monitor Weaviate benchmark and auto-commit results when complete

echo "Monitoring Weaviate benchmark..."
echo "Will automatically commit results when complete."
echo ""

# Wait for benchmark to complete
while pgrep -f "run_weaviate_benchmark.py" > /dev/null; do
    echo "$(date): Benchmark still running..."
    sleep 300  # Check every 5 minutes
done

echo ""
echo "$(date): Benchmark completed!"
echo ""

# Wait a moment for file writes to complete
sleep 5

# Commit the query latency benchmark results
echo "Committing query latency benchmark results..."
./Scripts/commit_benchmark_results.sh weaviate results/weaviate_experiment_001

echo ""
echo "Starting ingestion benchmark..."
source venv/bin/activate && python Scripts/run_weaviate_ingestion_benchmark.py

# Wait for ingestion benchmark to complete
while pgrep -f "run_weaviate_ingestion_benchmark.py" > /dev/null; do
    echo "$(date): Ingestion benchmark still running..."
    sleep 300
done

echo ""
echo "$(date): Ingestion benchmark completed!"
echo ""

# Wait a moment for file writes to complete
sleep 5

# Commit the ingestion benchmark results
echo "Committing ingestion benchmark results..."
./Scripts/commit_benchmark_results.sh weaviate_ingestion results/weaviate_ingestion_experiment_001

echo ""
echo "✅ All Weaviate benchmarks complete and committed!"
