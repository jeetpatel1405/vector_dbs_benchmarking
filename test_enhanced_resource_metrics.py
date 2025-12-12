#!/usr/bin/env python3
"""Enhanced test script with larger dataset to show meaningful resource usage."""

import sys
import time
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark
from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import Document

def create_large_test_documents(num_docs=10):
    """Create a larger test dataset to generate more meaningful resource usage."""
    documents = []

    # Sample content that's more substantial
    base_content = """
    Climate change represents one of the most pressing challenges of our time. The Earth's climate system
    is complex and interconnected, involving atmospheric, oceanic, terrestrial, and cryospheric components.
    Human activities, particularly the emission of greenhouse gases, have significantly altered the
    composition of the atmosphere. Carbon dioxide levels have increased by over 40% since pre-industrial
    times, primarily due to fossil fuel combustion and deforestation. Methane concentrations have more
    than doubled, largely from agricultural practices and energy production. These changes are driving
    unprecedented warming, with global average temperatures rising by approximately 1.1°C since 1880.

    The impacts of climate change are far-reaching and multifaceted. Rising sea levels threaten coastal
    communities and infrastructure. Extreme weather events are becoming more frequent and intense,
    including heat waves, droughts, floods, and storms. Arctic sea ice is declining at an alarming rate,
    contributing to ice-albedo feedback that accelerates warming. Ocean acidification, caused by
    increased CO2 absorption, poses serious threats to marine ecosystems.

    Mitigation strategies focus on reducing greenhouse gas emissions through renewable energy adoption,
    energy efficiency improvements, sustainable transportation, and carbon capture technologies.
    Adaptation measures help communities and ecosystems adjust to unavoidable climate impacts.
    International cooperation through agreements like the Paris Accord is essential for coordinated
    global action. Individual actions, while important, must be complemented by systemic changes
    in policy, technology, and economic structures.
    """

    for i in range(num_docs):
        # Create unique content by varying the base content
        unique_content = f"Document {i+1} - {base_content}" * 3  # Triple the content

        doc = Document(
            id=f'large_doc_{i:03d}',
            content=unique_content,
            metadata={
                'title': f'Climate Change Document {i+1}',
                'category': 'climate_science',
                'size': len(unique_content)
            },
            source=f'climate_doc_{i+1}.txt'
        )
        documents.append(doc)

    return documents

def main():
    print("="*60)
    print("Testing Resource Metrics with Larger Dataset")
    print("="*60)

    # Configuration
    config = {
        'persist_directory': './vector_stores/large_test_chroma_db',
        'collection_name': 'large_test_collection'
    }

    # Use sentence-transformers for more realistic resource usage
    print("\n[1/5] Initializing embedding generator (sentence-transformers)...")
    embedding_gen = get_embedding_generator('sentence-transformers', model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create benchmark
    benchmark = ChromaRAGBenchmark(
        db_config=config,
        embedding_generator=embedding_gen,
        chunk_size=512,  # Larger chunks
        chunk_strategy='fixed'
    )

    try:
        # Connect
        print("\n[2/5] Connecting to Chroma...")
        benchmark.connect()

        # Create collection
        print("\n[3/5] Creating collection...")
        benchmark.create_collection(384)

        # Create larger test documents
        print("\n[4/5] Creating larger test dataset...")
        documents = create_large_test_documents(15)  # 15 documents with substantial content
        total_chars = sum(len(doc.content) for doc in documents)
        print(f"   Created {len(documents)} documents with {total_chars:,} total characters")

        # Ingest with resource monitoring
        print("\n[5/5] Ingesting documents with resource monitoring...")
        print("   (This will take longer to show meaningful resource usage)")

        start_time = time.time()
        ingestion_metrics = benchmark.ingest_documents(documents, batch_size=5, monitor_resources=True)
        end_time = time.time()

        print(f"\n✅ Ingestion completed in {end_time - start_time:.2f}s!")
        print(f"   Documents: {ingestion_metrics.num_documents}")
        print(f"   Chunks: {ingestion_metrics.num_chunks}")
        print(f"   Parsing time: {ingestion_metrics.total_parsing_time:.3f}s")
        print(f"   Embedding time: {ingestion_metrics.total_embedding_time:.3f}s")
        print(f"   Insertion time: {ingestion_metrics.total_insertion_time:.3f}s")

        # Check resource metrics in detail
        if hasattr(ingestion_metrics, 'ingestion_resource_metrics') and ingestion_metrics.ingestion_resource_metrics:
            resource_metrics = ingestion_metrics.ingestion_resource_metrics
            print(f"\n📊 Detailed Resource Metrics:")
            print(f"   Duration: {resource_metrics.duration:.2f}s")
            print(f"   CPU:")
            print(f"     Average: {resource_metrics.cpu_avg:.1f}%")
            print(f"     Maximum: {resource_metrics.cpu_max:.1f}%")
            print(f"     Minimum: {resource_metrics.cpu_min:.1f}%")
            print(f"   Memory:")
            print(f"     Average: {resource_metrics.memory_avg_mb:.1f} MB")
            print(f"     Maximum: {resource_metrics.memory_max_mb:.1f} MB")
            print(f"     Minimum: {resource_metrics.memory_min_mb:.1f} MB")
            print(f"   Disk I/O:")
            print(f"     Read: {resource_metrics.disk_read_total_mb:.3f} MB")
            print(f"     Write: {resource_metrics.disk_write_total_mb:.3f} MB")
            print(f"   Network I/O:")
            print(f"     Sent: {resource_metrics.network_sent_total_mb:.3f} MB")
            print(f"     Received: {resource_metrics.network_recv_total_mb:.3f} MB")
            print(f"   Snapshots collected: {len(resource_metrics.snapshots)}")

            # Test serialization
            ingestion_dict = ingestion_metrics.to_dict()
            if 'ingestion_resources' in ingestion_dict and ingestion_dict['ingestion_resources']:
                print(f"\n✅ Resource metrics successfully serialized!")
                res_dict = ingestion_dict['ingestion_resources']
                print(f"   Serialized structure: {list(res_dict.keys())}")
                print(f"   CPU data: {res_dict.get('cpu', {})}")
                print(f"   Memory data: {res_dict.get('memory', {})}")
            else:
                print(f"\n❌ Resource metrics not properly serialized!")

            # Test a few queries with resource monitoring
            print(f"\n🔍 Testing query resource monitoring...")
            queries = [
                "What is climate change?",
                "How do greenhouse gases affect temperature?",
                "What are the impacts of global warming?"
            ]

            from src.monitoring.resource_monitor import ResourceMonitor
            query_resource_monitor = ResourceMonitor(interval=0.1)
            query_resource_monitor.start()

            for i, query in enumerate(queries):
                query_embedding = embedding_gen.generate_embedding(query)
                result_ids, query_time, similarity_scores = benchmark.query(query_embedding, top_k=5)
                print(f"   Query {i+1}: {len(result_ids)} results in {query_time*1000:.2f}ms")
                time.sleep(0.1)  # Small delay to see resource changes

            query_resources = query_resource_monitor.stop()
            print(f"\n📊 Query Resource Usage:")
            print(f"   Duration: {query_resources.duration:.2f}s")
            print(f"   CPU avg: {query_resources.cpu_avg:.1f}% (max: {query_resources.cpu_max:.1f}%)")
            print(f"   Memory avg: {query_resources.memory_avg_mb:.1f}MB")

        else:
            print(f"\n❌ No ingestion resource metrics captured!")

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
