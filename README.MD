# Vector Database Benchmarking System

A comprehensive benchmarking framework for comparing 7 open-source vector databases in RAG (Retrieval-Augmented Generation) applications.

## Overview

This system provides a unified interface for benchmarking vector databases with identical workloads, enabling fair performance comparisons across:

- **pgvector** - PostgreSQL with vector extension
- **Qdrant** - Purpose-built vector database
- **Weaviate** - AI-native vector search engine
- **Milvus** - Distributed vector database
- **Chroma** - Embedded vector database
- **FAISS** - Facebook's similarity search library
- **OpenSearch** - k-NN search with OpenSearch

## Features

✅ **Universal Interface** - All databases implement the same `RAGBenchmark` interface
✅ **Zero Code Duplication** - Common logic in abstract base class
✅ **Comprehensive Metrics** - Ingestion time, query latency, accuracy, resource usage
✅ **Multiple Chunking Strategies** - Fixed, sentence, paragraph, semantic
✅ **Flexible Embeddings** - OpenAI, Sentence Transformers, or custom
✅ **Docker Support** - One-command deployment of all databases
✅ **Export Results** - JSON, CSV, and comparison tables
✅ **Resource Monitoring** - CPU, memory, disk I/O tracking

## Quick Start

### 1. Setup

```bash
# Run automated setup
./setup.sh
```

### 2. Test

```bash
# Test embedded databases (no Docker required)
source venv/bin/activate
python test_adapters.py
```

### 3. Run Benchmarks

```bash
python -m src.benchmark_runner --config configs/default.yaml
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed instructions.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[PHASE_2_VERIFICATION.md](PHASE_2_VERIFICATION.md)** - ✅ Complete verification report
- **[PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md)** - Implementation summary
- **[PHASE_2_PLAN.md](PHASE_2_PLAN.md)** - Detailed implementation guide
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Full system design

## Project Status

- ✅ Phase 1: Core framework porting
- ✅ Phase 2: All 7 database adapters
- ⏳ Phase 3: Advanced metrics & testing
- ⏳ Phase 4: Documentation & publication

---

**Ready to start?** Run `./setup.sh` then `python test_adapters.py`
