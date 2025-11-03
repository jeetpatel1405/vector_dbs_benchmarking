"""Chroma RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class ChromaRAGBenchmark(RAGBenchmark):
    """Chroma implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize Chroma RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - persist_directory: Directory to persist Chroma data
                - collection_name: Collection name
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.persist_directory = db_config.get('persist_directory', './chroma_db')
        self.collection_name = db_config.get('collection_name', 'rag_benchmark')

        self.client = None
        self.collection = None

    def connect(self) -> None:
        """Connect to Chroma."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"Connected to Chroma at {self.persist_directory}")

        except Exception as e:
            print(f"Failed to connect to Chroma: {e}")
            raise

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

        # Create collection with cosine similarity
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

            if (i + batch_size) % 1000 == 0:
                print(f"Inserted {min(i + batch_size, len(chunks))}/{len(chunks)} documents")

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query Chroma for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        start_time = time.time()

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        query_time = time.time() - start_time

        # Extract chunk numbers from metadata and similarity scores
        result_ids = []
        similarity_scores = []

        if results and 'metadatas' in results and len(results['metadatas']) > 0:
            for metadata in results['metadatas'][0]:
                chunk_num = metadata.get('chunk_num', 0)
                result_ids.append(chunk_num)

            # Chroma returns distances (lower is better for cosine)
            # Convert to similarity: similarity = 1 - distance for cosine
            if 'distances' in results and len(results['distances']) > 0:
                similarity_scores = [1.0 - d for d in results['distances'][0]]
            else:
                similarity_scores = [0.0] * len(result_ids)

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up Chroma resources."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
