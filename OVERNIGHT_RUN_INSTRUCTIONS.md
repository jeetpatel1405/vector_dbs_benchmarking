# Overnight Benchmark Run Instructions

## 📋 Overview

You have three automated scripts ready for running large-scale benchmarks:

### Current Status:
- **Option 1**: ✅ Running (Qdrant with real embeddings, ~1.5 hours)
- **Option 2**: ⏳ Ready (All 7 databases, ~10-12 hours)
- **Option 3**: ⏳ Ready (Quick test with random embeddings, ~20 min)

---

## 🚀 Starting the Overnight Run

### Option A: Automatic Sequencing (RECOMMENDED)

Start a script that waits for Option 1 to finish, then automatically begins Option 2:

```bash
nohup bash Scripts/start_overnight_after_option1.sh > sequencer.log 2>&1 &
```

This will:
1. Monitor Option 1 progress
2. Wait for completion
3. Automatically start Option 2 (all 7 databases)
4. Log everything to `sequencer.log` and `overnight_run.log`

### Option B: Manual Start (after Option 1 completes)

Wait for Option 1 to finish, then manually start:

```bash
nohup bash Scripts/run_all_databases_overnight.sh > overnight_run.log 2>&1 &
```

---

## 📊 Monitoring Progress

### Check Overall Status
```bash
bash Scripts/check_overnight_progress.sh
```

Shows:
- Which databases are complete
- Current running database
- Progress percentage
- Recent log entries

### Live Log Monitoring
```bash
# Option 1 (Qdrant initial run)
tail -f large_corpus_run.log

# Option 2 (All databases overnight)
tail -f overnight_run.log

# Automatic sequencer
tail -f sequencer.log
```

### Quick Progress Check
```bash
# See how many databases are complete
ls -la results/*_large_corpus_real_embeddings/ | grep "^d" | wc -l

# Check current database
ps aux | grep "run_.*_benchmark.py"
```

---

## ⏰ Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Option 1: Qdrant | ~1.5 hours | 🔄 Running (0.5% complete) |
| Option 2: All 7 DBs | ~10-12 hours | ⏳ Queued |
| **TOTAL** | **~12 hours** | Ready for overnight |

### Expected Completion Time:
- **If started at 3:45 PM**: Complete by ~4:00 AM tomorrow
- **If started at 10:00 PM**: Complete by ~10:00 AM tomorrow

---

## 📁 Results Location

Each database will save results to:
```
results/
├── qdrant_large_corpus_real_embeddings/
│   ├── results.json
│   ├── config.json
│   └── performance_quality.png
├── weaviate_large_corpus_real_embeddings/
├── milvus_large_corpus_real_embeddings/
├── faiss_large_corpus_real_embeddings/
├── chroma_large_corpus_real_embeddings/
├── opensearch_large_corpus_real_embeddings/
└── pgvector_large_corpus_real_embeddings/
```

Individual logs:
```
results/
├── qdrant_large_corpus.log
├── weaviate_large_corpus.log
├── ...
└── overnight_run_summary_YYYYMMDD_HHMMSS.txt
```

---

## 🔍 Troubleshooting

### If a database fails:
The script continues with remaining databases. Check:
```bash
# See which failed
cat results/overnight_run_summary_*.txt

# Check specific database log
less results/<database>_large_corpus.log
```

### If you need to restart:
```bash
# Stop all
pkill -f "run_.*_benchmark.py"
pkill -f "run_all_databases_overnight.sh"

# Restart from specific database
# Edit Scripts/run_all_databases_overnight.sh line 56:
# DATABASES=("faiss" "chroma" "opensearch" "pgvector")  # Skip completed ones
bash Scripts/run_all_databases_overnight.sh
```

### Check Docker services:
```bash
docker-compose ps
```

---

## 📈 What to Expect Tomorrow

When you return, you should have:

1. **Complete results for 7 databases**
   - Ingestion timing (parsing, embedding, insertion)
   - Query latency (median, P95, P99) for Top-K [1,3,5,10,20]
   - IR metrics (Recall@K, Precision@K, MRR)
   - Resource usage (CPU, memory)

2. **~2.25 million chunks** processed per database

3. **Real embeddings** with publication-quality metrics

4. **Ready for paper Figure 4 & Figure 5**
   - Ingestion performance comparison
   - Query performance analysis
   - Quality-latency trade-offs

---

## ✅ Recommended Action NOW

Start the automatic sequencer:

```bash
cd /Users/rezarassool/Source/vector_dbs_benchmarking

nohup bash Scripts/start_overnight_after_option1.sh > sequencer.log 2>&1 &

echo "Background PID: $!"
```

Then you can disconnect and everything will run automatically!

---

## 📧 Quick Status Email Template

Tomorrow morning, check status and results:

```bash
bash Scripts/check_overnight_progress.sh > status.txt
cat status.txt
```

---

**Last Updated:** $(date)
**Option 1 Status:** Running (0.5% complete, ~88 min remaining)
**Option 2 Status:** Ready to auto-start
