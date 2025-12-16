#!/usr/bin/env python3
"""Quick test to check IngestionMetrics timing attributes."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.parsers.document_parser import IngestionMetrics

# Create a sample IngestionMetrics instance
metrics = IngestionMetrics(
    num_documents=5,
    num_chunks=50,
    total_parsing_time=1.5,
    total_embedding_time=3.2,
    total_insertion_time=0.8,
    avg_parsing_time_per_doc=0.3,
    avg_embedding_time_per_chunk=0.064,
    avg_insertion_time_per_chunk=0.016,
    total_size_bytes=12000,
    chunk_sizes=[200, 250, 180, 220, 240]
)

print("IngestionMetrics Test:")
print(f"total_parsing_time: {metrics.total_parsing_time}")
print(f"parsing_time property: {metrics.parsing_time}")
print(f"hasattr(metrics, 'parsing_time'): {hasattr(metrics, 'parsing_time')}")

print(f"total_embedding_time: {metrics.total_embedding_time}")
print(f"embedding_time property: {metrics.embedding_time}")
print(f"hasattr(metrics, 'embedding_time'): {hasattr(metrics, 'embedding_time')}")

print(f"total_insertion_time: {metrics.total_insertion_time}")
print(f"insertion_time property: {metrics.insertion_time}")
print(f"hasattr(metrics, 'insertion_time'): {hasattr(metrics, 'insertion_time')}")

# Test the exact pattern used in benchmark scripts
parsing_time = metrics.parsing_time if hasattr(metrics, 'parsing_time') else 0
embedding_time = metrics.embedding_time if hasattr(metrics, 'embedding_time') else 0
insertion_time = metrics.insertion_time if hasattr(metrics, 'insertion_time') else 0

print(f"\nScript pattern results:")
print(f"parsing_time: {parsing_time}")
print(f"embedding_time: {embedding_time}")
print(f"insertion_time: {insertion_time}")
