#!/usr/bin/env python3
"""Test script to verify resource metrics are captured."""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import DocumentParser, Document

def create_test_documents():
    """Create a small test dataset."""
    documents = []
    for i in range(3):
        doc = Document(
            id=f'test_doc_{i}',
            content=f'This is test document {i}. ' * 20,  # Small content
            metadata={'title': f'Test Doc {i}'},
            source=f'test_{i}.txt'
        )
        documents.append(doc)
    return documents

def main():
    print("="*50)
    print("Testing Resource Metrics Capture")
    print("="*50)

    # Configuration
    config = {
        'persist_directory': './vector_stores/test_chroma_db',
        'collection_name': 'test_collection'
    }

    # Create embedding generator (random for fast testing)
    embedding_gen = get_embedding_generator('random', dimension=384)

    # Create benchmark
    benchmark = ChromaRAGBenchmark(
        db_config=config,
        embedding_generator=embedding_gen,
        chunk_size=256,
        chunk_strategy='fixed'
    )

    try:
        # Connect
        print("\n[1/4] Connecting to Chroma...")
        benchmark.connect()

        # Create collection
        print("\n[2/4] Creating collection...")
        benchmark.create_collection(384)

        # Test documents
        print("\n[3/4] Creating test documents...")
        documents = create_test_documents()

        # Ingest with resource monitoring
        print("\n[4/4] Ingesting documents with resource monitoring...")
        ingestion_metrics = benchmark.ingest_documents(documents, batch_size=10, monitor_resources=True)

        print(f"\n✅ Ingestion completed!")
        print(f"  Documents: {ingestion_metrics.num_documents}")
        print(f"  Chunks: {ingestion_metrics.num_chunks}")
        print(f"  Parsing time: {ingestion_metrics.total_parsing_time:.3f}s")
        print(f"  Embedding time: {ingestion_metrics.total_embedding_time:.3f}s")
        print(f"  Insertion time: {ingestion_metrics.total_insertion_time:.3f}s")

        # Check resource metrics
        if hasattr(ingestion_metrics, 'ingestion_resource_metrics') and ingestion_metrics.ingestion_resource_metrics:
            resource_metrics = ingestion_metrics.ingestion_resource_metrics
            print(f"\n📊 Resource Metrics:")
            print(f"  Duration: {resource_metrics.duration:.2f}s")
            print(f"  CPU avg: {resource_metrics.cpu_avg:.1f}% (max: {resource_metrics.cpu_max:.1f}%)")
            print(f"  Memory avg: {resource_metrics.memory_avg_mb:.1f}MB (max: {resource_metrics.memory_max_mb:.1f}MB)")
            print(f"  Disk read: {resource_metrics.disk_read_total_mb:.3f}MB")
            print(f"  Disk write: {resource_metrics.disk_write_total_mb:.3f}MB")
            print(f"  Snapshots: {len(resource_metrics.snapshots)}")

            # Test to_dict conversion
            ingestion_dict = ingestion_metrics.to_dict()
            if 'ingestion_resources' in ingestion_dict and ingestion_dict['ingestion_resources']:
                print(f"\n✅ Resource metrics successfully included in to_dict()!")
                print(f"  ingestion_resources keys: {list(ingestion_dict['ingestion_resources'].keys())}")
            else:
                print(f"\n❌ Resource metrics missing from to_dict()!")
        else:
            print(f"\n❌ No resource metrics captured!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            benchmark.cleanup()
            benchmark.disconnect()
        except:
            pass

if __name__ == '__main__':
    main()
