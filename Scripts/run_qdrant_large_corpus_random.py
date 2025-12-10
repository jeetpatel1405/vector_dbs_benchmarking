#!/usr/bin/env python3
"""
Quick test: Qdrant benchmark with large corpus using RANDOM embeddings.
Purpose: Validate infrastructure at scale (~18 minutes vs 26 minutes with real embeddings)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main benchmark script's main function
import importlib.util
spec = importlib.util.spec_from_file_location("qdrant_bench",
    project_root / "Scripts" / "run_qdrant_benchmark.py")
qdrant_bench = importlib.util.module_from_spec(spec)

# Override CONFIG before loading
import json

qdrant_bench.CONFIG = {
    'corpus_path': 'Data/test_corpus/documents',
    'test_cases_path': 'Data/test_corpus/test_cases.json',
    'output_dir': 'results/qdrant_large_corpus_random_embeddings',
    'qdrant_config': {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'benchmark_test'
    },
    'embedding_model': 'random-384',
    'embedding_type': 'random',  # Fast random embeddings for infrastructure test
    'chunk_size': 512,
    'chunk_overlap': 50,
    'chunk_strategy': 'fixed',
    'top_k_values': [1, 3, 5, 10, 20],
    'batch_size': 100
}

# Load and execute
spec.loader.exec_module(qdrant_bench)
sys.exit(qdrant_bench.main())
