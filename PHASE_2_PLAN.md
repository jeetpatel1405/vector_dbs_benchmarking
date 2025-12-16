# Phase 2: Database Adapter Completion

**Goal:** Port and implement all 7 vector database adapters to complete the core benchmarking framework

**Duration:** 2-3 weeks (40-50 hours)
**Priority:** P0 (Critical Path)
**Dependencies:** Phase 1 (Complete ✅)

---

## Overview

Phase 2 focuses on completing the database adapter layer by:
1. Porting the 2 remaining adapters from the adjacent project (pgvector, Weaviate)
2. Implementing 4 new adapters using the established pattern (Milvus, Chroma, FAISS, OpenSearch)
3. Adding Pinecone support (optional, requires API key)
4. Creating integration tests for each adapter
5. Documenting setup procedures for each database

---

## Objectives

### Primary Objectives
- ✅ **Universal Interface**: All 7 databases implement the same `RAGBenchmark` interface
- ✅ **Zero Code Duplication**: Each adapter is 200-250 lines with no repeated logic
- ✅ **Plug-and-Play**: Add new databases by registering adapters with `BenchmarkRunner`
- ✅ **Production-Ready**: Each adapter includes error handling, connection validation, cleanup

### Success Metrics
- [ ] All 7 databases pass the same test suite
- [ ] Benchmark runner can execute all databases in parallel
- [ ] Each database has Docker Compose configuration
- [ ] Documentation covers setup for each database
- [ ] Integration tests achieve >80% code coverage

---

## Task Breakdown

### Task 2.1: Port pgvector Adapter ⏱️ 2-3 hours

**Source:** `/Users/rezarassool/Source/vector-db-benchmark/benchmarks/pgvector_rag_benchmark.py`
**Destination:** `src/vector_dbs/pgvector_adapter.py`

**Steps:**
1. Copy `pgvector_rag_benchmark.py` from adjacent project
2. Update imports to new structure:
   ```python
   from src.vector_dbs.rag_benchmark import RAGBenchmark
   from src.utils.chunking import Chunk
   ```
3. Rename class to `PgvectorRAGBenchmark`
4. Test connection to PostgreSQL with pgvector extension
5. Verify HNSW and IVFFlat index creation
6. Test batch insertion and querying

**Configuration (configs/default.yaml):**
```yaml
databases:
  pgvector:
    host: localhost
    port: 5432
    database: vectordb
    user: postgres
    password: postgres
    table_name: rag_benchmark
    index_type: hnsw  # or ivfflat
    m: 16  # HNSW parameter
    ef_construction: 64  # HNSW parameter
```

**Docker Service:**
```yaml
pgvector:
  image: ankane/pgvector:latest
  environment:
    POSTGRES_PASSWORD: postgres
    POSTGRES_DB: vectordb
  ports:
    - "5432:5432"
```

**Acceptance Criteria:**
- [ ] Adapter connects to PostgreSQL
- [ ] Creates table with vector column
- [ ] Creates HNSW or IVFFlat index
- [ ] Inserts 1000+ chunks in batches
- [ ] Queries return top-k results with latency <50ms
- [ ] Cleanup drops table successfully

---

### Task 2.2: Port Weaviate Adapter ⏱️ 2-3 hours

**Source:** `/Users/rezarassool/Source/vector-db-benchmark/benchmarks/weaviate_rag_benchmark.py`
**Destination:** `src/vector_dbs/weaviate_adapter.py`

**Steps:**
1. Copy `weaviate_rag_benchmark.py` from adjacent project
2. Update imports to new structure
3. Rename class to `WeaviateRAGBenchmark`
4. Test connection to Weaviate instance
5. Verify schema creation with vector properties
6. Test batch object insertion
7. Test vector search with nearVector

**Configuration (configs/default.yaml):**
```yaml
databases:
  weaviate:
    host: localhost
    port: 8080
    class_name: RAGBenchmark
    vectorizer: none  # We provide our own embeddings
```

**Docker Service:**
```yaml
weaviate:
  image: semitechnologies/weaviate:latest
  environment:
    AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
    PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    QUERY_DEFAULTS_LIMIT: 25
  ports:
    - "8080:8080"
```

**Acceptance Criteria:**
- [ ] Adapter connects to Weaviate
- [ ] Creates class schema with vector property
- [ ] Batch imports objects with embeddings
- [ ] nearVector queries return results
- [ ] Cleanup deletes class schema
- [ ] Handles connection errors gracefully

---

### Task 2.3: Implement Milvus Adapter ⏱️ 6-8 hours

**Destination:** `src/vector_dbs/milvus_adapter.py`
**Template:** Use `qdrant_adapter.py` as reference

**Implementation Guide:**

```python
"""Milvus RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class MilvusRAGBenchmark(RAGBenchmark):
    """Milvus implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 19530)
        self.collection_name = db_config.get('collection_name', 'rag_benchmark')
        self.index_type = db_config.get('index_type', 'IVF_FLAT')  # or HNSW

        self.connection = None
        self.collection = None

    def connect(self) -> None:
        """Connect to Milvus."""
        from pymilvus import connections, utility

        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        from pymilvus import connections
        connections.disconnect("default")
        print("Disconnected from Milvus")

    def create_collection(self, dimension: int) -> None:
        """Create Milvus collection."""
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility

        # Drop collection if exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped existing collection: {self.collection_name}")

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_num", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields, description="RAG Benchmark Collection")
        self.collection = Collection(self.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": self.index_type,
            "params": {"nlist": 128} if self.index_type == "IVF_FLAT" else {"M": 16, "efConstruction": 64}
        }
        self.collection.create_index("embedding", index_params)
        print(f"Created collection '{self.collection_name}' with {self.index_type} index")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into Milvus."""
        start_time = time.time()

        # Prepare data
        ids = list(range(len(chunks)))
        texts = [chunk.text[:65535] for chunk in chunks]  # Milvus VARCHAR limit
        chunk_ids = [chunk.id for chunk in chunks]
        doc_ids = [chunk.metadata.get('doc_id', '') for chunk in chunks]
        chunk_nums = [chunk.metadata.get('chunk_num', i) for i, chunk in enumerate(chunks)]

        # Insert in batches
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size].tolist()
            batch_texts = texts[i:i+batch_size]
            batch_chunk_ids = chunk_ids[i:i+batch_size]
            batch_doc_ids = doc_ids[i:i+batch_size]
            batch_chunk_nums = chunk_nums[i:i+batch_size]

            self.collection.insert([
                batch_ids,
                batch_embeddings,
                batch_texts,
                batch_chunk_ids,
                batch_doc_ids,
                batch_chunk_nums
            ])

        # Flush to persist
        self.collection.flush()

        # Load collection for search
        self.collection.load()

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """Query Milvus for similar chunks."""
        start_time = time.time()

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10} if self.index_type == "IVF_FLAT" else {"ef": 64}
        }

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )

        query_time = time.time() - start_time

        # Extract IDs from results
        result_ids = [hit.id for hit in results[0]]

        return result_ids, query_time

    def cleanup(self) -> None:
        """Clean up Milvus resources."""
        from pymilvus import utility

        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
```

**Configuration:**
```yaml
databases:
  milvus:
    host: localhost
    port: 19530
    collection_name: rag_benchmark
    index_type: HNSW  # or IVF_FLAT
```

**Docker Service:**
```yaml
# Milvus requires etcd and minio
etcd:
  image: quay.io/coreos/etcd:v3.5.5
  environment:
    - ETCD_AUTO_COMPACTION_MODE=revision
    - ETCD_AUTO_COMPACTION_RETENTION=1000
    - ETCD_QUOTA_BACKEND_BYTES=4294967296
    - ETCD_SNAPSHOT_COUNT=50000
  volumes:
    - etcd_data:/etcd
  command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

minio:
  image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  environment:
    MINIO_ACCESS_KEY: minioadmin
    MINIO_SECRET_KEY: minioadmin
  volumes:
    - minio_data:/minio_data
  command: minio server /minio_data
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    interval: 30s
    timeout: 20s
    retries: 3

milvus:
  image: milvusdb/milvus:v2.3.3
  command: ["milvus", "run", "standalone"]
  environment:
    ETCD_ENDPOINTS: etcd:2379
    MINIO_ADDRESS: minio:9000
  volumes:
    - milvus_data:/var/lib/milvus
  ports:
    - "19530:19530"
    - "9091:9091"
  depends_on:
    - etcd
    - minio

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

**Acceptance Criteria:**
- [ ] Connects to Milvus cluster
- [ ] Creates collection with vector field
- [ ] Supports IVF_FLAT and HNSW indexes
- [ ] Batch inserts with flush
- [ ] Queries return accurate results
- [ ] Handles schema changes gracefully

---

### Task 2.4: Implement Chroma Adapter ⏱️ 4-6 hours

**Destination:** `src/vector_dbs/chroma_adapter.py`

**Implementation Guide:**

```python
"""Chroma RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class ChromaRAGBenchmark(RAGBenchmark):
    """Chroma implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        super().__init__(db_config, embedding_generator, **kwargs)

        self.persist_directory = db_config.get('persist_directory', './chroma_db')
        self.collection_name = db_config.get('collection_name', 'rag_benchmark')

        self.client = None
        self.collection = None

    def connect(self) -> None:
        """Connect to Chroma."""
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        print(f"Connected to Chroma at {self.persist_directory}")

    def disconnect(self) -> None:
        """Disconnect from Chroma."""
        # Chroma client doesn't need explicit disconnect
        print("Disconnected from Chroma")

    def create_collection(self, dimension: int) -> None:
        """Create Chroma collection."""
        # Delete collection if exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        # Create collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created collection '{self.collection_name}'")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into Chroma."""
        start_time = time.time()

        # Prepare data
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                'chunk_id': chunk.id,
                'doc_id': chunk.metadata.get('doc_id', ''),
                'chunk_num': chunk.metadata.get('chunk_num', i),
                'source': chunk.metadata.get('source', ''),
            }
            for i, chunk in enumerate(chunks)
        ]

        # Insert in batches
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """Query Chroma for similar chunks."""
        start_time = time.time()

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        query_time = time.time() - start_time

        # Extract IDs and convert to integers
        result_ids = []
        if results and 'ids' in results and len(results['ids']) > 0:
            result_ids = [int(id.split('_')[1]) for id in results['ids'][0]]

        return result_ids, query_time

    def cleanup(self) -> None:
        """Clean up Chroma resources."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
```

**Configuration:**
```yaml
databases:
  chroma:
    persist_directory: ./vector_stores/chroma_db
    collection_name: rag_benchmark
```

**Docker Service (Optional - Chroma can run embedded):**
```yaml
chroma:
  image: chromadb/chroma:latest
  ports:
    - "8000:8000"
  volumes:
    - chroma_data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - ANONYMIZED_TELEMETRY=FALSE

volumes:
  chroma_data:
```

**Acceptance Criteria:**
- [ ] Works in embedded mode (no Docker required)
- [ ] Works in client-server mode
- [ ] Persists data to disk
- [ ] Supports cosine similarity
- [ ] Returns metadata with results
- [ ] Handles collection recreation

---

### Task 2.5: Implement FAISS Adapter ⏱️ 4-6 hours

**Destination:** `src/vector_dbs/faiss_adapter.py`

**Key Considerations:**
- FAISS is a library, not a database (no client-server)
- Supports multiple index types (Flat, IVF, HNSW)
- Requires separate metadata storage (use dict or SQLite)
- Very fast for in-memory operations

**Implementation Skeleton:**

```python
"""FAISS RAG benchmark implementation."""
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class FAISSRAGBenchmark(RAGBenchmark):
    """FAISS implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        super().__init__(db_config, embedding_generator, **kwargs)

        self.index_path = db_config.get('index_path', './faiss_index')
        self.index_type = db_config.get('index_type', 'Flat')  # Flat, IVF, HNSW

        self.index = None
        self.metadata_store = {}  # Maps index ID to chunk metadata

    def connect(self) -> None:
        """Connect to FAISS (no-op for embedded library)."""
        import faiss
        print(f"FAISS version: {faiss.__version__}")

    def disconnect(self) -> None:
        """Disconnect (no-op for FAISS)."""
        pass

    def create_collection(self, dimension: int) -> None:
        """Create FAISS index."""
        import faiss

        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        print(f"Created FAISS {self.index_type} index with dimension {dimension}")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into FAISS."""
        import faiss

        start_time = time.time()

        # Train index if needed (IVF)
        if isinstance(self.index, faiss.IndexIVF):
            print("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors
        self.index.add(embeddings)

        # Store metadata separately
        for i, chunk in enumerate(chunks):
            self.metadata_store[i] = {
                'text': chunk.text,
                'chunk_id': chunk.id,
                'doc_id': chunk.metadata.get('doc_id', ''),
                'chunk_num': chunk.metadata.get('chunk_num', i),
            }

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """Query FAISS for similar chunks."""
        start_time = time.time()

        # Search
        query_vector = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        query_time = time.time() - start_time

        result_ids = indices[0].tolist()
        return result_ids, query_time

    def cleanup(self) -> None:
        """Clean up FAISS resources."""
        self.index = None
        self.metadata_store = {}
        print("Cleaned up FAISS index")
```

**Configuration:**
```yaml
databases:
  faiss:
    index_path: ./vector_stores/faiss_index
    index_type: HNSW  # Flat, IVF, or HNSW
```

**No Docker Service Needed** (embedded library)

**Acceptance Criteria:**
- [ ] Supports Flat, IVF, HNSW index types
- [ ] Trains IVF index automatically
- [ ] Stores and retrieves metadata
- [ ] Can save/load index to disk
- [ ] Handles large batch insertions

---

### Task 2.6: Implement OpenSearch Adapter ⏱️ 6-8 hours

**Destination:** `src/vector_dbs/opensearch_adapter.py`

**Key Considerations:**
- OpenSearch k-NN plugin required
- Supports HNSW and IVF indexes
- RESTful API via opensearch-py client
- Similar to Elasticsearch

**Implementation Skeleton:**

```python
"""OpenSearch RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class OpenSearchRAGBenchmark(RAGBenchmark):
    """OpenSearch implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 9200)
        self.index_name = db_config.get('index_name', 'rag_benchmark')
        self.index_method = db_config.get('index_method', 'hnsw')  # hnsw or ivf

        self.client = None

    def connect(self) -> None:
        """Connect to OpenSearch."""
        from opensearchpy import OpenSearch

        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )

        info = self.client.info()
        print(f"Connected to OpenSearch {info['version']['number']}")

    def disconnect(self) -> None:
        """Disconnect from OpenSearch."""
        if self.client:
            self.client.close()
            print("Disconnected from OpenSearch")

    def create_collection(self, dimension: int) -> None:
        """Create OpenSearch index with k-NN settings."""
        # Delete index if exists
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"Deleted existing index: {self.index_name}")

        # Define index mapping
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": self.index_method,
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            } if self.index_method == "hnsw" else {}
                        }
                    },
                    "text": {"type": "text"},
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "chunk_num": {"type": "integer"}
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=index_body)
        print(f"Created OpenSearch index '{self.index_name}' with {self.index_method}")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into OpenSearch."""
        from opensearchpy import helpers

        start_time = time.time()

        # Prepare documents for bulk insert
        actions = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            action = {
                "_index": self.index_name,
                "_id": i,
                "_source": {
                    "embedding": embedding.tolist(),
                    "text": chunk.text,
                    "chunk_id": chunk.id,
                    "doc_id": chunk.metadata.get('doc_id', ''),
                    "chunk_num": chunk.metadata.get('chunk_num', i)
                }
            }
            actions.append(action)

        # Bulk insert
        helpers.bulk(self.client, actions, chunk_size=batch_size)

        # Refresh index
        self.client.indices.refresh(index=self.index_name)

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """Query OpenSearch for similar chunks."""
        start_time = time.time()

        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": top_k
                    }
                }
            }
        }

        response = self.client.search(index=self.index_name, body=query_body)

        query_time = time.time() - start_time

        result_ids = [int(hit['_id']) for hit in response['hits']['hits']]
        return result_ids, query_time

    def cleanup(self) -> None:
        """Clean up OpenSearch resources."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"Deleted index: {self.index_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
```

**Configuration:**
```yaml
databases:
  opensearch:
    host: localhost
    port: 9200
    index_name: rag_benchmark
    index_method: hnsw  # or ivf
```

**Docker Service:**
```yaml
opensearch:
  image: opensearchproject/opensearch:2.11.0
  environment:
    - discovery.type=single-node
    - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
    - "DISABLE_SECURITY_PLUGIN=true"
  ports:
    - "9200:9200"
    - "9600:9600"
  volumes:
    - opensearch_data:/usr/share/opensearch/data

volumes:
  opensearch_data:
```

**Acceptance Criteria:**
- [ ] Connects to OpenSearch cluster
- [ ] Creates k-NN enabled index
- [ ] Supports HNSW and IVF methods
- [ ] Bulk inserts documents
- [ ] k-NN search returns results
- [ ] Handles index deletion

---

### Task 2.7: Add Pinecone Adapter (Optional) ⏱️ 4-6 hours

**Destination:** `src/vector_dbs/pinecone_adapter.py`

**Note:** Requires Pinecone API key (cloud service)

**Configuration:**
```yaml
databases:
  pinecone:
    api_key: ${PINECONE_API_KEY}
    environment: us-west1-gcp
    index_name: rag-benchmark
    metric: cosine
```

**Acceptance Criteria:**
- [ ] Connects with API key
- [ ] Creates serverless or pod-based index
- [ ] Upserts vectors in batches
- [ ] Queries return results
- [ ] Handles rate limits gracefully

---

### Task 2.8: Update Benchmark Runner Registration ⏱️ 2 hours

**Objective:** Enable all adapters to be registered and used

**Updates to `src/benchmark_runner.py`:**

```python
# Add at top of file
from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark
from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark
from src.vector_dbs.weaviate_adapter import WeaviateRAGBenchmark
from src.vector_dbs.milvus_adapter import MilvusRAGBenchmark
from src.vector_dbs.chroma_adapter import ChromaRAGBenchmark
from src.vector_dbs.faiss_adapter import FAISSRAGBenchmark
from src.vector_dbs.opensearch_adapter import OpenSearchRAGBenchmark
# from src.vector_dbs.pinecone_adapter import PineconeRAGBenchmark  # Optional

class BenchmarkRunner:
    def __init__(self, config_path: str = "configs/default.yaml"):
        super().__init__()

        # Auto-register all adapters
        self.register_benchmark('pgvector', PgvectorRAGBenchmark)
        self.register_benchmark('qdrant', QdrantRAGBenchmark)
        self.register_benchmark('weaviate', WeaviateRAGBenchmark)
        self.register_benchmark('milvus', MilvusRAGBenchmark)
        self.register_benchmark('chroma', ChromaRAGBenchmark)
        self.register_benchmark('faiss', FAISSRAGBenchmark)
        self.register_benchmark('opensearch', OpenSearchRAGBenchmark)
        # self.register_benchmark('pinecone', PineconeRAGBenchmark)  # Optional
```

---

### Task 2.9: Create Integration Tests ⏱️ 8-10 hours

**Objective:** Ensure all adapters work correctly

**File:** `tests/integration/test_all_adapters.py`

```python
"""Integration tests for all database adapters."""
import pytest
import numpy as np

from src.embeddings.embedding_generator import get_embedding_generator
from src.parsers.document_parser import Document


# Fixture for test data
@pytest.fixture
def test_documents():
    return [
        Document(
            id='doc1',
            content='Machine learning is a subset of artificial intelligence. ' * 10,
            metadata={'title': 'ML Intro'},
            source='test1.txt'
        ),
        Document(
            id='doc2',
            content='Vector databases enable semantic search capabilities. ' * 10,
            metadata={'title': 'Vector DBs'},
            source='test2.txt'
        )
    ]


@pytest.fixture
def test_queries():
    return [
        'What is machine learning?',
        'How do vector databases work?',
        'Explain semantic search'
    ]


@pytest.fixture
def embedding_gen():
    return get_embedding_generator('random', dimension=384)


# Test each adapter
@pytest.mark.integration
class TestPgvectorAdapter:
    def test_full_benchmark(self, test_documents, test_queries, embedding_gen):
        from src.vector_dbs.pgvector_adapter import PgvectorRAGBenchmark

        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres',
            'table_name': 'test_benchmark'
        }

        benchmark = PgvectorRAGBenchmark(
            db_config=config,
            embedding_generator=embedding_gen,
            chunk_size=256,
            chunk_strategy='sentence'
        )

        results = benchmark.run_full_benchmark(
            documents=test_documents,
            query_texts=test_queries,
            top_k=5
        )

        assert results.num_documents == 2
        assert results.num_chunks > 0
        assert results.avg_query_latency > 0
        assert results.queries_per_second > 0


@pytest.mark.integration
class TestQdrantAdapter:
    def test_full_benchmark(self, test_documents, test_queries, embedding_gen):
        from src.vector_dbs.qdrant_adapter import QdrantRAGBenchmark

        config = {
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'test_benchmark'
        }

        benchmark = QdrantRAGBenchmark(
            db_config=config,
            embedding_generator=embedding_gen,
            chunk_size=256,
            chunk_strategy='sentence'
        )

        results = benchmark.run_full_benchmark(
            documents=test_documents,
            query_texts=test_queries,
            top_k=5
        )

        assert results.num_documents == 2
        assert results.num_chunks > 0


# Similar tests for Weaviate, Milvus, Chroma, FAISS, OpenSearch...
```

**Run tests:**
```bash
# All integration tests
pytest tests/integration/ -v

# Specific adapter
pytest tests/integration/test_all_adapters.py::TestQdrantAdapter -v

# With coverage
pytest tests/integration/ --cov=src --cov-report=html
```

---

### Task 2.10: Documentation ⏱️ 4-6 hours

**Create setup guides for each database:**

1. **`docs/setup/pgvector_setup.md`**
   - PostgreSQL installation
   - pgvector extension setup
   - Connection testing

2. **`docs/setup/qdrant_setup.md`**
   - Docker setup
   - gRPC vs HTTP
   - Collection management

3. **`docs/setup/weaviate_setup.md`**
   - Docker setup
   - Schema configuration
   - Authentication options

4. **`docs/setup/milvus_setup.md`**
   - Docker Compose with etcd and minio
   - Index types comparison
   - Performance tuning

5. **`docs/setup/chroma_setup.md`**
   - Embedded vs client-server
   - Persistence configuration
   - Migration guide

6. **`docs/setup/faiss_setup.md`**
   - Index types explained
   - Memory requirements
   - Save/load patterns

7. **`docs/setup/opensearch_setup.md`**
   - k-NN plugin installation
   - Index configuration
   - Cluster setup

**Create comparison guide:**

`docs/DATABASE_COMPARISON.md` - Feature matrix comparing all 7 databases

---

## Timeline

### Week 1: Port Existing Adapters
- **Day 1-2:** Task 2.1 - Port pgvector (3h)
- **Day 2-3:** Task 2.2 - Port Weaviate (3h)
- **Day 3-4:** Task 2.8 - Update runner registration (2h)
- **Day 4-5:** Task 2.9 - Basic integration tests (4h)

### Week 2: Implement New Adapters (Part 1)
- **Day 1-2:** Task 2.3 - Implement Milvus (8h)
- **Day 3-4:** Task 2.4 - Implement Chroma (6h)
- **Day 5:** Task 2.9 - Tests for Milvus and Chroma (4h)

### Week 3: Implement New Adapters (Part 2)
- **Day 1-2:** Task 2.5 - Implement FAISS (6h)
- **Day 3-4:** Task 2.6 - Implement OpenSearch (8h)
- **Day 5:** Task 2.9 - Tests for FAISS and OpenSearch (4h)

### Week 4: Documentation & Polish
- **Day 1-3:** Task 2.10 - Documentation for all databases (12h)
- **Day 4:** Task 2.7 - Pinecone (optional, 6h)
- **Day 5:** Final testing and cleanup (4h)

**Total Estimated Time:** 40-50 hours over 3-4 weeks

---

## Success Criteria

### Functional Requirements
- [ ] All 7 core databases have working adapters
- [ ] All adapters implement the same `RAGBenchmark` interface
- [ ] Benchmark runner can execute scenarios for any database
- [ ] Docker Compose starts all databases successfully
- [ ] Integration tests pass for all adapters

### Code Quality
- [ ] No code duplication between adapters
- [ ] Each adapter is 200-300 lines of code
- [ ] Proper error handling in all methods
- [ ] Type hints for all function signatures
- [ ] Docstrings for all classes and methods

### Documentation
- [ ] Setup guide exists for each database
- [ ] Configuration examples in `configs/default.yaml`
- [ ] README updated with adapter list
- [ ] Comparison table showing database features
- [ ] Troubleshooting guide for common issues

### Testing
- [ ] Integration test for each adapter
- [ ] Tests cover connect, create, insert, query, cleanup
- [ ] Code coverage >80% for adapter code
- [ ] Performance benchmarks captured for each database
- [ ] All tests pass in CI/CD pipeline

---

## Deliverables

1. **Code:**
   - ✅ `src/vector_dbs/pgvector_adapter.py`
   - ✅ `src/vector_dbs/weaviate_adapter.py`
   - ✅ `src/vector_dbs/milvus_adapter.py`
   - ✅ `src/vector_dbs/chroma_adapter.py`
   - ✅ `src/vector_dbs/faiss_adapter.py`
   - ✅ `src/vector_dbs/opensearch_adapter.py`
   - ⚠️ `src/vector_dbs/pinecone_adapter.py` (optional)

2. **Tests:**
   - ✅ `tests/integration/test_all_adapters.py`
   - ✅ `tests/unit/test_vector_dbs/` (unit tests for each adapter)

3. **Configuration:**
   - ✅ Updated `configs/default.yaml` with all databases
   - ✅ Updated `docker-compose.yml` with all services

4. **Documentation:**
   - ✅ `docs/setup/*.md` (7 setup guides)
   - ✅ `docs/DATABASE_COMPARISON.md`
   - ✅ Updated `README.md`

5. **Infrastructure:**
   - ✅ Updated `requirements.txt` with all client libraries
   - ✅ `.github/workflows/test.yml` (CI/CD for tests)

---

## Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Database connection issues | Medium | High | Docker health checks, retry logic |
| Version incompatibilities | Low | Medium | Pin all versions in requirements.txt |
| Index creation failures | Medium | Medium | Proper error handling, validation |
| Memory issues with FAISS | Low | Medium | Configurable batch sizes |
| API rate limits (Pinecone) | Medium | Low | Rate limiting, backoff strategy |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | Medium | Medium | Strict adherence to interface |
| Timeline delays | Low | Medium | Buffer time in estimates |
| Docker setup complexity | Medium | Low | Comprehensive documentation |
| Testing infrastructure gaps | Low | High | Pytest fixtures and mocks |

---

## Phase 2 Completion Checklist

- [ ] All 7 database adapters implemented
- [ ] All adapters pass integration tests
- [ ] Docker Compose starts all services
- [ ] Documentation complete for all databases
- [ ] Benchmark runner executes multi-database scenarios
- [ ] Results export works for all databases
- [ ] Code coverage >80%
- [ ] No critical bugs in issue tracker
- [ ] README updated with Phase 2 completion
- [ ] PHASE_3_PLAN.md created

---

**Last Updated:** 2025-10-22
**Version:** 1.0
**Status:** Ready to Start
**Owner:** Development Team
