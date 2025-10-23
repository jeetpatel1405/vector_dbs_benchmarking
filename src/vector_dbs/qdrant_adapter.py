"""Qdrant RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class QdrantRAGBenchmark(RAGBenchmark):
    """Qdrant implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize Qdrant RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - host: Qdrant host
                - port: Qdrant port
                - collection_name: Collection name
                - prefer_grpc: Use gRPC instead of HTTP
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 6333)
        self.collection_name = db_config.get('collection_name', 'rag_benchmark')
        self.prefer_grpc = db_config.get('prefer_grpc', False)

        self.client = None

    def connect(self) -> None:
        """Connect to Qdrant."""
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=self.prefer_grpc
            )

            # Test connection
            collections = self.client.get_collections()
            print(f"Connected to Qdrant at {self.host}:{self.port}")
            print(f"Existing collections: {len(collections.collections)}")

        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        if self.client:
            self.client.close()
            print("Disconnected from Qdrant")

    def create_collection(self, dimension: int) -> None:
        """Create Qdrant collection."""
        from qdrant_client.models import Distance, VectorParams

        # Delete collection if exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection '{self.collection_name}' with dimension {dimension}")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into Qdrant."""
        from qdrant_client.models import PointStruct

        start_time = time.time()

        # Prepare points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    'text': chunk.text,
                    'chunk_id': chunk.id,
                    'chunk_num': chunk.metadata.get('chunk_num', i),
                    'doc_id': chunk.metadata.get('doc_id', ''),
                    'source': chunk.metadata.get('source', ''),
                    'strategy': chunk.metadata.get('strategy', ''),
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index
                }
            )
            points.append(point)

        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            if (i + batch_size) % 1000 == 0:
                print(f"Inserted {min(i + batch_size, len(points))}/{len(points)} points")

        insert_time = time.time() - start_time
        print(f"Inserted {len(points)} points in {insert_time:.2f}s")

        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query Qdrant for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        start_time = time.time()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        query_time = time.time() - start_time

        # Extract IDs and similarity scores
        result_ids = [hit.id for hit in results]
        similarity_scores = [hit.score for hit in results]

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up Qdrant resources."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
