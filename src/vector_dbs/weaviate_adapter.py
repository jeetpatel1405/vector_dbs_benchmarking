"""Weaviate RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class WeaviateRAGBenchmark(RAGBenchmark):
    """Weaviate implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize Weaviate RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - host: Weaviate host
                - port: Weaviate port
                - class_name: Weaviate class name
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 8080)
        self.class_name = db_config.get('class_name', 'RAGBenchmark')

        self.client = None

    def connect(self) -> None:
        """Connect to Weaviate."""
        try:
            import weaviate

            # Connect to local Weaviate instance
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port
            )

            # Test connection
            if self.client.is_ready():
                print(f"Connected to Weaviate at {self.host}:{self.port}")
            else:
                raise ConnectionError("Weaviate is not ready")

        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()
            print("Disconnected from Weaviate")

    def create_collection(self, dimension: int) -> None:
        """Create Weaviate collection."""
        import weaviate.classes as wvc

        # Delete collection if exists
        try:
            self.client.collections.delete(self.class_name)
            print(f"Deleted existing collection: {self.class_name}")
        except Exception:
            pass

        # Create collection
        self.client.collections.create(
            name=self.class_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="chunk_num",
                    data_type=wvc.config.DataType.INT
                ),
                wvc.config.Property(
                    name="doc_id",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="strategy",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="start_index",
                    data_type=wvc.config.DataType.INT
                ),
                wvc.config.Property(
                    name="end_index",
                    data_type=wvc.config.DataType.INT
                )
            ]
        )

        print(f"Created collection '{self.class_name}' with dimension {dimension}")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into Weaviate."""
        start_time = time.time()

        collection = self.client.collections.get(self.class_name)

        # Prepare data objects
        with collection.batch.dynamic() as batch:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                properties = {
                    'text': chunk.text,
                    'chunk_id': chunk.id,
                    'chunk_num': chunk.metadata.get('chunk_num', i),
                    'doc_id': chunk.metadata.get('doc_id', ''),
                    'source': chunk.metadata.get('source', ''),
                    'strategy': chunk.metadata.get('strategy', ''),
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index
                }

                batch.add_object(
                    properties=properties,
                    vector=embedding.tolist()
                )

                if (i + 1) % 1000 == 0:
                    print(f"Inserted {i + 1}/{len(chunks)} objects")

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} objects in {insert_time:.2f}s")

        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query Weaviate for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        start_time = time.time()

        collection = self.client.collections.get(self.class_name)

        response = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=top_k,
            return_metadata=['distance']
        )

        query_time = time.time() - start_time

        # Extract chunk numbers and similarity scores from results
        result_ids = []
        similarity_scores = []

        for obj in response.objects:
            chunk_num = obj.properties.get('chunk_num', 0)
            result_ids.append(chunk_num)

            # Weaviate returns distance (lower is better)
            # Convert to similarity: similarity = 1 - distance
            if obj.metadata and obj.metadata.distance is not None:
                similarity_scores.append(float(1.0 - obj.metadata.distance))
            else:
                similarity_scores.append(0.0)

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up Weaviate resources."""
        try:
            self.client.collections.delete(self.class_name)
            print(f"Deleted collection: {self.class_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
