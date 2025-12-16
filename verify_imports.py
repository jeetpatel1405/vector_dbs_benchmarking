#!/usr/bin/env python3
"""Verify that all ported modules can be imported successfully."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Verifying imports for vector_dbs_benchmarking project...")
print("=" * 60)

errors = []
successes = []

# Test imports
tests = [
    ("Core: Base Benchmark", "from src.vector_dbs.base_benchmark import BaseBenchmark, BenchmarkResults"),
    ("Core: RAG Benchmark", "from src.vector_dbs.rag_benchmark import RAGBenchmark, RAGBenchmarkResults"),
    ("Core: Benchmark Runner", "from src.benchmark_runner import BenchmarkRunner"),
    ("Utils: Chunking", "from src.utils.chunking import Chunk, get_chunking_strategy"),
    ("Monitoring: Resource Monitor", "from src.monitoring.resource_monitor import ResourceMonitor"),
    ("Embeddings: Generator", "from src.embeddings.embedding_generator import get_embedding_generator, EMBEDDING_CONFIGS"),
    ("Parsers: Document Parser", "from src.parsers.document_parser import Document, DocumentParser"),
    ("Database: Qdrant Adapter", "from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark"),
]

for name, import_statement in tests:
    try:
        exec(import_statement)
        successes.append(name)
        print(f"✅ {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"❌ {name}: {e}")

print("\n" + "=" * 60)
print(f"Results: {len(successes)}/{len(tests)} imports successful")

if errors:
    print("\n❌ ERRORS:")
    for name, error in errors:
        print(f"  - {name}: {error}")
    sys.exit(1)
else:
    print("\n✅ All imports successful! Framework is ready to use.")
    sys.exit(0)
