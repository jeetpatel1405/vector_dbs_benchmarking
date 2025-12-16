# Quick Start Guide

Get the Vector Database Benchmarking System up and running in 5 minutes!

## Prerequisites

- **Python 3.8+** - Check with `python3 --version`
- **Docker** (optional) - Required for pgvector, Qdrant, Weaviate, Milvus, OpenSearch
- **Git** - To clone the repository

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to project directory
cd /Users/rezarassool/Source/vector_dbs_benchmarking

# Run setup script
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify imports
- ✅ Create necessary directories

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p results vector_stores/chroma_db vector_stores/faiss_index Plots

# Verify
python verify_imports.py
```

## Testing

### Quick Test (No Docker Required)

Test the embedded databases (FAISS and Chroma):

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run tests
python test_adapters.py
```

Expected output:
```
Testing FAISS
============================================================
✅ FAISS - SUCCESS

Testing Chroma
============================================================
✅ Chroma - SUCCESS
```

### Full Test (All 7 Databases)

```bash
# Start Docker services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Run all tests
chmod +x test.sh
./test.sh
```

Or run the test script:
```bash
python test_adapters.py
```

Expected output:
```
SUMMARY
============================================================
FAISS                ✅ PASS
Chroma               ✅ PASS
Qdrant               ✅ PASS
pgvector             ✅ PASS
Weaviate             ✅ PASS
Milvus               ✅ PASS
OpenSearch           ✅ PASS

Total: 7/7 adapters passed

🎉 All adapters working!
```

## Running Benchmarks

### Quick Start - No Docker Required

Run FAISS or Chroma (embedded databases):

```bash
# Activate environment
source venv/bin/activate

# Run FAISS benchmark (fastest, in-memory)
python Scripts/run_faiss_benchmark.py

# Run Chroma benchmark
python Scripts/run_chroma_benchmark.py
```

Results appear in `results/faiss_experiment_001/` or `results/chroma_experiment_001/`

### Client-Server Databases

For databases requiring Docker, start them individually to avoid resource contention:

```bash
# 1. Start database
docker-compose up -d qdrant  # or pgvector, weaviate, milvus, opensearch

# 2. Wait for startup (5-30 seconds depending on database)
sleep 15

# 3. Run benchmark
source venv/bin/activate
python Scripts/run_qdrant_benchmark.py

# 4. Stop database
docker-compose stop qdrant
```

**Available benchmark scripts**:
- `Scripts/run_faiss_benchmark.py` - FAISS (no Docker)
- `Scripts/run_chroma_benchmark.py` - Chroma (no Docker)
- `Scripts/run_qdrant_benchmark.py` - Qdrant
- `Scripts/run_pgvector_benchmark.py` - pgvector
- `Scripts/run_weaviate_benchmark.py` - Weaviate
- `Scripts/run_milvus_benchmark.py` - Milvus (requires etcd + minio)
- `Scripts/run_opensearch_benchmark.py` - OpenSearch

### Cross-Database Comparison

After running benchmarks on multiple databases:

```bash
python Scripts/create_comparison.py
```

This generates `results/all_databases_comparison.png` with 4-panel comparison across all databases.

## Project Structure

```
vector_dbs_benchmarking/
├── setup.sh              # Automated setup script
├── test.sh               # Automated test script
├── test_adapters.py      # Adapter test suite
├── verify_imports.py     # Import verification
├── src/
│   ├── benchmark_runner.py        # Main orchestrator
│   ├── vector_dbs/                # Database adapters
│   │   ├── pgvector_adapter.py
│   │   ├── qdrant_adapter.py
│   │   ├── weaviate_adapter.py
│   │   ├── milvus_adapter.py
│   │   ├── chroma_adapter.py
│   │   ├── faiss_adapter.py
│   │   └── opensearch_adapter.py
│   ├── embeddings/                # Embedding generators
│   ├── utils/                     # Chunking strategies
│   ├── monitoring/                # Resource monitoring
│   └── parsers/                   # Document parsers
├── configs/
│   └── default.yaml               # Default configuration
├── docker-compose.yml             # Docker services
└── requirements.txt               # Python dependencies
```

## Troubleshooting

### Import Errors

```bash
# Verify all imports
python verify_imports.py

# If errors, reinstall dependencies
pip install -r requirements.txt
```

### Docker Services Not Starting

```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Port Conflicts

If you get "port already in use" errors:

```bash
# Check what's using the ports
lsof -i :5432   # pgvector
lsof -i :6333   # Qdrant
lsof -i :8080   # Weaviate
lsof -i :9200   # OpenSearch
lsof -i :19530  # Milvus

# Kill the process or change ports in docker-compose.yml
```

### Module Not Found

```bash
# Make sure you're in the project directory
cd /Users/rezarassool/Source/vector_dbs_benchmarking

# Activate virtual environment
source venv/bin/activate

# Verify Python can find modules
python -c "import sys; print(sys.path)"
```

## Next Steps

1. **Read the documentation:**
   - `README.MD` - Project overview and results
   - `BENCHMARK_VERIFICATION.md` - Validation report for all databases
   - `PROJECT_STATE.md` - Current status and technical details
   - `CONTRIBUTOR_GUIDE.md` - How to extend the system

2. **Run all benchmarks:**
   - Try each of the 7 database benchmarks
   - Run `Scripts/create_comparison.py` for cross-database comparison
   - Review results in `results/` directory

3. **Contribute:**
   - Add quality metrics (Precision@K, NDCG, MRR)
   - Implement concurrent query testing
   - Scale to larger test corpus
   - See Phase 3 roadmap in README.MD

## Common Commands

```bash
# Setup
./setup.sh
source venv/bin/activate

# Test all adapters
python test_adapters.py

# Run individual benchmarks
python Scripts/run_faiss_benchmark.py
python Scripts/run_chroma_benchmark.py
python Scripts/run_qdrant_benchmark.py
python Scripts/run_pgvector_benchmark.py
python Scripts/run_weaviate_benchmark.py
python Scripts/run_milvus_benchmark.py
python Scripts/run_opensearch_benchmark.py

# Generate comparison
python Scripts/create_comparison.py

# Docker management
docker-compose up -d           # Start all services
docker-compose ps              # Check service status
docker-compose logs -f qdrant  # View logs
docker-compose down            # Stop all services

# Virtual environment
source venv/bin/activate       # Activate
deactivate                     # Deactivate
```

## Getting Help

1. Check the troubleshooting section above
2. Review error messages in `docker-compose logs`
3. Run `python verify_imports.py` to check dependencies
4. Check `PHASE_2_COMPLETE.md` for architecture details

## Support

For issues or questions:
- Check existing documentation in the project
- Review the implementation plans
- Run diagnostic scripts (`verify_imports.py`, `test_adapters.py`)

---

**Ready to benchmark?** Run `./setup.sh` followed by `./test.sh`!
