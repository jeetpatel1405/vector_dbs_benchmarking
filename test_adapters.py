#!/usr/bin/env python3
"""Quick test script for all database adapters."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import Document

# Import all adapters
from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark
from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark
from src.vector_dbs.weaviate_adapter import WeaviateRAGBenchmark
from src.vector_dbs.milvus_adapter import MilvusRAGBenchmark
from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark
from src.vector_dbs.faiss_adapter import FAISSRAGBenchmark
from src.vector_dbs.opensearch_adapter import OpenSearchRAGBenchmark


def create_test_data():
    """Create simple test documents and queries."""
    documents = [
        Document(
            id='doc1',
            content='Machine learning is a subset of artificial intelligence. ' * 5,
            metadata={'title': 'ML Intro'},
            source='test1.txt'
        ),
        Document(
            id='doc2',
            content='Vector databases enable semantic search capabilities. ' * 5,
            metadata={'title': 'Vector DBs'},
            source='test2.txt'
        )
    ]

    queries = [
        'What is machine learning?',
        'How do vector databases work?'
    ]

    return documents, queries


def test_adapter(adapter_class, config, name):
    """Test a single adapter."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    try:
        # Create embedding generator (random for fast testing)
        embedding_gen = get_embedding_generator('random', dimension=384)

        # Create benchmark
        benchmark = adapter_class(
            db_config=config,
            embedding_generator=embedding_gen,
            chunk_size=256,
            chunk_strategy='fixed'
        )

        # Get test data
        documents, queries = create_test_data()

        # Run benchmark
        results = benchmark.run_full_benchmark(
            documents=documents,
            query_texts=queries,
            top_k=3,
            batch_size=10
        )

        print(f"\n✅ {name} - SUCCESS")
        print(f"  Documents: {results.num_documents}")
        print(f"  Chunks: {results.num_chunks}")
        print(f"  Avg Query Latency: {results.avg_query_latency*1000:.2f}ms")
        print(f"  QPS: {results.queries_per_second:.2f}")
        return True

    except Exception as e:
        print(f"\n❌ {name} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all adapters."""
    print("Vector Database Adapter Testing")
    print("="*60)

    results = {}

    # Test FAISS (no Docker required)
    results['FAISS'] = test_adapter(
        FAISSRAGBenchmark,
        {'index_path': './test_faiss', 'index_type': 'Flat'},
        'FAISS'
    )

    # Test Chroma (no Docker required)
    results['Chroma'] = test_adapter(
        ChromaRAGBenchmark,
        {'persist_directory': './test_chroma', 'collection_name': 'test'},
        'Chroma'
    )

    # Test Qdrant (requires Docker)
    print("\n\nℹ️  The following tests require Docker services to be running")
    print("   Run: docker-compose up -d")

    results['Qdrant'] = test_adapter(
        QdrantRAGBenchmark,
        {'host': 'localhost', 'port': 6333, 'collection_name': 'test'},
        'Qdrant'
    )

    # Test pgvector (requires Docker)
    results['pgvector'] = test_adapter(
        PgvectorRAGBenchmark,
        {
            'host': 'localhost',
            'port': 5432,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres',
            'table_name': 'test'
        },
        'pgvector'
    )

    # Test Weaviate (requires Docker)
    results['Weaviate'] = test_adapter(
        WeaviateRAGBenchmark,
        {'host': 'localhost', 'port': 8080, 'class_name': 'Test'},
        'Weaviate'
    )

    # Test Milvus (requires Docker)
    results['Milvus'] = test_adapter(
        MilvusRAGBenchmark,
        {'host': 'localhost', 'port': 19530, 'collection_name': 'test'},
        'Milvus'
    )

    # Test OpenSearch (requires Docker)
    results['OpenSearch'] = test_adapter(
        OpenSearchRAGBenchmark,
        {'host': 'localhost', 'port': 9200, 'index_name': 'test'},
        'OpenSearch'
    )

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for adapter, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{adapter:20s} {status}")

    print(f"\nTotal: {passed}/{total} adapters passed")

    if passed == total:
        print("\n🎉 All adapters working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} adapter(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
