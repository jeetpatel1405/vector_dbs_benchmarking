# Running the Full Benchmark Suite

## Quick Start

To run the complete benchmark suite overnight:

```bash
./Scripts/run_all_benchmarks.sh
```

This will:
1. Start all Docker services (Qdrant, FAISS, Chroma, Milvus, Weaviate, pgvector, OpenSearch)
2. Run all ingestion benchmarks (~6-14 hours depending on XML corpus)
3. Run all query benchmarks (~2-4 hours)
4. Generate a comprehensive summary report

## Command Options

### Run Everything (Default)
```bash
./Scripts/run_all_benchmarks.sh
```

### Run Only Ingestion Benchmarks
```bash
./Scripts/run_all_benchmarks.sh --ingestion-only
```

### Run Only Query Benchmarks
```bash
./Scripts/run_all_benchmarks.sh --query-only
```

## Prerequisites

1. **Docker Desktop** must be running
2. **Virtual environment** must exist at `venv/`
3. **Test corpus** must be in place at `Data/test_corpus/documents/`

## Expected Duration

With the 1GB Wikipedia XML corpus:

| Benchmark Type | Estimated Time |
|----------------|----------------|
| Ingestion (all DBs) | 6-14 hours |
| Query (all DBs) | 2-4 hours |
| **Total** | **8-18 hours** |

With smaller text corpus only (20 documents):

| Benchmark Type | Estimated Time |
|----------------|----------------|
| Ingestion (all DBs) | 15-30 minutes |
| Query (all DBs) | 10-20 minutes |
| **Total** | **25-50 minutes** |

## Output

All results are saved to `results/full_suite_TIMESTAMP/`:

```
results/full_suite_20251119_183045/
├── SUMMARY.md                              # Overview of all results
├── benchmark_suite.log                     # Complete execution log
├── benchmark_results.csv                   # Summary CSV
├── run_qdrant_ingestion_benchmark.log     # Individual logs
├── run_qdrant_benchmark.log
├── run_faiss_ingestion_benchmark.log
├── run_faiss_benchmark.log
├── ...
└── [individual benchmark result directories]
```

## Monitoring Progress

### View live progress
```bash
tail -f results/full_suite_*/benchmark_suite.log
```

### Check Docker services
```bash
docker-compose ps
```

### Check current benchmark
```bash
ps aux | grep python | grep benchmark
```

## Running in Background

### Using nohup
```bash
nohup ./Scripts/run_all_benchmarks.sh > /dev/null 2>&1 &
```

### Using screen
```bash
screen -S benchmarks
./Scripts/run_all_benchmarks.sh
# Detach with Ctrl+A, D
# Reattach with: screen -r benchmarks
```

### Using tmux
```bash
tmux new -s benchmarks
./Scripts/run_all_benchmarks.sh
# Detach with Ctrl+B, D
# Reattach with: tmux attach -t benchmarks
```

## Troubleshooting

### Docker Services Not Starting

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Start fresh
docker-compose up -d
```

### Out of Memory

If benchmarks fail due to memory issues:

1. Close other applications
2. Increase Docker memory limit (Docker Desktop → Settings → Resources)
3. Run benchmarks one at a time manually

### Individual Benchmark Fails

Check the specific log file:
```bash
cat results/full_suite_*/run_DBNAME_benchmark.log
```

## Manual Execution

If you prefer to run benchmarks individually:

```bash
# Start all services
docker-compose up -d

# Run individual ingestion benchmark
source venv/bin/activate
python Scripts/run_qdrant_ingestion_benchmark.py

# Run individual query benchmark
python Scripts/run_qdrant_benchmark.py
```

## Corpus Selection

### To use only small text corpus (faster)

Remove or temporarily move the large XML file:
```bash
mv Data/test_corpus/documents/enwiki-latest-pages-articles1.xml ~/backup/
```

### To use full corpus including XML

Ensure the XML file is present:
```bash
ls -lh Data/test_corpus/documents/enwiki-latest-pages-articles1.xml
```

## Post-Benchmark Analysis

After benchmarks complete:

1. **Review Summary**: `cat results/full_suite_*/SUMMARY.md`
2. **Compare Performance**: Check results CSVs in individual benchmark directories
3. **View Plots**: Open PNG files generated in results directories
4. **Aggregate Results**: Use provided analysis scripts (if available)

## Best Practices

- ✅ Run overnight or during off-hours
- ✅ Ensure stable power supply (for laptops, keep plugged in)
- ✅ Close memory-intensive applications
- ✅ Monitor disk space (results can be large)
- ✅ Use screen/tmux for long-running sessions
- ✅ Check logs periodically for errors

## Support

For issues or questions:
- Check individual benchmark logs first
- Review Docker service health: `docker-compose ps`
- Check resource usage: `docker stats`
- Refer to main project documentation
