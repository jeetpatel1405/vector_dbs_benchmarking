#!/bin/bash
#
# Master script to run complete benchmark suite overnight
# This will:
#   1. Start all Docker services
#   2. Run all ingestion benchmarks
#   3. Run all query benchmarks
#   4. Generate summary report
#
# Usage: ./Scripts/run_all_benchmarks.sh [--ingestion-only|--query-only]
#

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/full_suite_${TIMESTAMP}"
MAIN_LOG="${LOG_DIR}/benchmark_suite.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} ${message}" | tee -a "${MAIN_LOG}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${message}" | tee -a "${MAIN_LOG}"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} ${message}" | tee -a "${MAIN_LOG}"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${message}" | tee -a "${MAIN_LOG}"
            ;;
    esac

    echo "[${timestamp}] [${level}] ${message}" >> "${MAIN_LOG}"
}

# Banner
print_banner() {
    echo ""
    echo "============================================================================="
    echo "  Vector Database Benchmark Suite"
    echo "  Started: $(date)"
    echo "  Log Directory: ${LOG_DIR}"
    echo "============================================================================="
    echo "" | tee -a "${MAIN_LOG}"
}

# Check if Docker is running
check_docker() {
    log INFO "Checking Docker status..."
    if ! docker info > /dev/null 2>&1; then
        log ERROR "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    log SUCCESS "Docker is running"
}

# Start all Docker services
start_services() {
    log INFO "Starting all Docker services..."

    # Start all services
    docker-compose up -d pgvector qdrant weaviate milvus opensearch chroma

    log INFO "Waiting for services to be healthy (this may take 2-3 minutes)..."
    sleep 10

    # Wait for services with timeout
    local timeout=180
    local elapsed=0
    local interval=5

    while [ $elapsed -lt $timeout ]; do
        local healthy=$(docker-compose ps | grep -c "healthy" || echo "0")
        local total_services=6

        if [ "$healthy" -ge "$total_services" ]; then
            log SUCCESS "All services are healthy!"
            docker-compose ps | tee -a "${MAIN_LOG}"
            return 0
        fi

        log INFO "Healthy services: ${healthy}/${total_services} - waiting..."
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log WARNING "Some services may not be fully healthy yet. Proceeding anyway..."
    docker-compose ps | tee -a "${MAIN_LOG}"
}

# Run a single benchmark script
run_benchmark() {
    local script=$1
    local name=$(basename "$script" .py)
    local log_file="${LOG_DIR}/${name}.log"

    log INFO "Running: ${name}"
    echo "  Log: ${log_file}"

    local start_time=$(date +%s)

    # Activate venv and run benchmark
    if source venv/bin/activate && python "$script" > "${log_file}" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_min=$((duration / 60))
        local duration_sec=$((duration % 60))

        log SUCCESS "${name} completed in ${duration_min}m ${duration_sec}s"
        echo "${name},SUCCESS,${duration}" >> "${LOG_DIR}/benchmark_results.csv"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log ERROR "${name} failed after ${duration}s - see ${log_file}"
        echo "${name},FAILED,${duration}" >> "${LOG_DIR}/benchmark_results.csv"
        return 1
    fi
}

# Run all ingestion benchmarks
run_ingestion_benchmarks() {
    log INFO "========================================"
    log INFO "Starting INGESTION benchmarks"
    log INFO "========================================"

    local benchmarks=(
        "Scripts/run_qdrant_ingestion_benchmark.py"
        "Scripts/run_faiss_ingestion_benchmark.py"
        "Scripts/run_chroma_ingestion_benchmark.py"
        "Scripts/run_milvus_ingestion_benchmark.py"
        "Scripts/run_weaviate_ingestion_benchmark.py"
        "Scripts/run_pgvector_ingestion_benchmark.py"
        "Scripts/run_opensearch_ingestion_benchmark.py"
    )

    local total=${#benchmarks[@]}
    local current=0
    local failed=0

    for benchmark in "${benchmarks[@]}"; do
        current=$((current + 1))
        log INFO "Progress: ${current}/${total}"

        if ! run_benchmark "$benchmark"; then
            failed=$((failed + 1))
        fi

        echo "" | tee -a "${MAIN_LOG}"
    done

    if [ $failed -eq 0 ]; then
        log SUCCESS "All ${total} ingestion benchmarks completed successfully!"
    else
        log WARNING "${failed}/${total} ingestion benchmarks failed"
    fi

    return $failed
}

# Run all query benchmarks
run_query_benchmarks() {
    log INFO "========================================"
    log INFO "Starting QUERY benchmarks"
    log INFO "========================================"

    local benchmarks=(
        "Scripts/run_qdrant_benchmark.py"
        "Scripts/run_faiss_benchmark.py"
        "Scripts/run_chroma_benchmark.py"
        "Scripts/run_milvus_benchmark.py"
        "Scripts/run_weaviate_benchmark.py"
        "Scripts/run_pgvector_benchmark.py"
        "Scripts/run_opensearch_benchmark.py"
    )

    local total=${#benchmarks[@]}
    local current=0
    local failed=0

    for benchmark in "${benchmarks[@]}"; do
        current=$((current + 1))
        log INFO "Progress: ${current}/${total}"

        if ! run_benchmark "$benchmark"; then
            failed=$((failed + 1))
        fi

        echo "" | tee -a "${MAIN_LOG}"
    done

    if [ $failed -eq 0 ]; then
        log SUCCESS "All ${total} query benchmarks completed successfully!"
    else
        log WARNING "${failed}/${total} query benchmarks failed"
    fi

    return $failed
}

# Generate summary report
generate_summary() {
    log INFO "Generating summary report..."

    local summary_file="${LOG_DIR}/SUMMARY.md"

    cat > "${summary_file}" << EOF
# Benchmark Suite Summary

**Started:** ${TIMESTAMP}
**Completed:** $(date)

## Results

EOF

    if [ -f "${LOG_DIR}/benchmark_results.csv" ]; then
        echo "| Benchmark | Status | Duration (s) |" >> "${summary_file}"
        echo "|-----------|--------|--------------|" >> "${summary_file}"

        while IFS=',' read -r name status duration; do
            echo "| ${name} | ${status} | ${duration} |" >> "${summary_file}"
        done < "${LOG_DIR}/benchmark_results.csv"
    fi

    cat >> "${summary_file}" << EOF

## Files

- Main log: \`benchmark_suite.log\`
- Individual logs: \`*_benchmark.log\`
- Results directory: \`${LOG_DIR}\`

## Next Steps

1. Review individual benchmark logs for detailed results
2. Check result plots in each benchmark's output directory
3. Compare performance metrics across vector databases

EOF

    log SUCCESS "Summary report generated: ${summary_file}"

    # Display summary
    echo ""
    echo "============================================================================="
    cat "${summary_file}"
    echo "============================================================================="
}

# Cleanup function
cleanup() {
    log INFO "Benchmark suite finished"
    log INFO "Results saved to: ${LOG_DIR}"
}

# Main execution
main() {
    local mode="full"

    # Parse arguments
    if [ "$1" == "--ingestion-only" ]; then
        mode="ingestion"
    elif [ "$1" == "--query-only" ]; then
        mode="query"
    fi

    print_banner

    # Initialize results CSV
    echo "Benchmark,Status,Duration" > "${LOG_DIR}/benchmark_results.csv"

    # Check Docker
    check_docker

    # Start services (unless query-only mode)
    if [ "$mode" != "query" ]; then
        start_services
    fi

    local suite_start=$(date +%s)

    # Run benchmarks
    local ingestion_failed=0
    local query_failed=0

    if [ "$mode" == "full" ] || [ "$mode" == "ingestion" ]; then
        run_ingestion_benchmarks
        ingestion_failed=$?
    fi

    if [ "$mode" == "full" ] || [ "$mode" == "query" ]; then
        run_query_benchmarks
        query_failed=$?
    fi

    local suite_end=$(date +%s)
    local total_duration=$((suite_end - suite_start))
    local total_hours=$((total_duration / 3600))
    local total_mins=$(((total_duration % 3600) / 60))

    # Generate summary
    generate_summary

    log INFO "Total suite duration: ${total_hours}h ${total_mins}m"

    # Cleanup
    cleanup

    # Exit with error if any benchmarks failed
    local total_failed=$((ingestion_failed + query_failed))
    if [ $total_failed -gt 0 ]; then
        exit 1
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main
main "$@"
