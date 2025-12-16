"""Milvus RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class MilvusRAGBenchmark(RAGBenchmark):
    """Milvus implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize Milvus RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - host: Milvus host
                - port: Milvus port
                - collection_name: Collection name
                - index_type: 'IVF_FLAT' or 'HNSW'
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 19530)
        self.collection_name = db_config.get('collection_name', 'rag_benchmark')
        self.index_type = db_config.get('index_type', 'IVF_FLAT')

        self.collection = None

    def connect(self) -> None:
        """Connect to Milvus."""
        try:
            from pymilvus import connections, utility

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )

            # Test connection
            server_version = utility.get_server_version()
            print(f"Connected to Milvus {server_version} at {self.host}:{self.port}")

        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        from pymilvus import connections
        connections.disconnect("default")
        print("Disconnected from Milvus")

    def create_collection(self, dimension: int) -> None:
        """Create Milvus collection."""
        from pymilvus import (
            CollectionSchema,
            FieldSchema,
            DataType,
            Collection,
            utility
        )

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
        if self.index_type == "IVF_FLAT":
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        elif self.index_type == "HNSW":
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 64}
            }
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

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
        chunk_ids = [chunk.id[:256] for chunk in chunks]  # Limit to max_length
        doc_ids = [chunk.metadata.get('doc_id', '')[:256] for chunk in chunks]
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

            if (i + batch_size) % 1000 == 0:
                print(f"Inserted {min(i + batch_size, len(chunks))}/{len(chunks)} entities")

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
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query Milvus for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        start_time = time.time()

        # Set search params based on index type
        if self.index_type == "IVF_FLAT":
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
        elif self.index_type == "HNSW":
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
        else:
            search_params = {"metric_type": "COSINE"}

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_num"]
        )

        query_time = time.time() - start_time

        # Extract chunk numbers and similarity scores from results
        result_ids = []
        similarity_scores = []

        if results and len(results) > 0:
            for hit in results[0]:
                chunk_num = hit.entity.get('chunk_num')
                if chunk_num is not None:
                    result_ids.append(chunk_num)
                    # Milvus returns similarity scores for COSINE metric
                    # Score is already in [0, 1] range where higher is better
                    similarity_scores.append(float(hit.distance))

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up Milvus resources."""
        from pymilvus import utility

        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
