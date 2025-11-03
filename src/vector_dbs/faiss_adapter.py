"""FAISS RAG benchmark implementation."""
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class FAISSRAGBenchmark(RAGBenchmark):
    """FAISS implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize FAISS RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - index_path: Path to save FAISS index
                - index_type: 'Flat', 'IVF', or 'HNSW'
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.index_path = db_config.get('index_path', './faiss_index')
        self.index_type = db_config.get('index_type', 'Flat')

        self.index = None
        self.metadata_store = {}  # Maps index ID to chunk metadata

    def connect(self) -> None:
        """Connect to FAISS (no-op for embedded library)."""
        try:
            import faiss
            print(f"FAISS version: {faiss.__version__}")
        except Exception as e:
            print(f"Failed to import FAISS: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect (no-op for FAISS)."""
        print("Disconnected from FAISS")

    def create_collection(self, dimension: int) -> None:
        """Create FAISS index."""
        import faiss

        if self.index_type == 'Flat':
            # Flat (exact search) - uses L2 distance
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == 'IVF':
            # IVF (approximate search)
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = self.db_config.get('nlist', 100)  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.index_type == 'HNSW':
            # HNSW (approximate search)
            m = self.db_config.get('m', 32)  # number of connections
            self.index = faiss.IndexHNSWFlat(dimension, m)
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

        # Train index if needed (IVF requires training)
        if isinstance(self.index, faiss.IndexIVF):
            print(f"Training IVF index with {len(embeddings)} vectors...")
            train_start = time.time()
            self.index.train(embeddings.astype(np.float32))
            train_time = time.time() - train_start
            print(f"Training completed in {train_time:.2f}s")

        # Add vectors to index
        self.index.add(embeddings.astype(np.float32))

        # Store metadata separately (FAISS doesn't support metadata natively)
        for i, chunk in enumerate(chunks):
            self.metadata_store[i] = {
                'text': chunk.text,
                'chunk_id': chunk.id,
                'doc_id': chunk.metadata.get('doc_id', ''),
                'chunk_num': chunk.metadata.get('chunk_num', i),
                'source': chunk.metadata.get('source', ''),
            }

        insert_time = time.time() - start_time
        print(f"Inserted {len(chunks)} chunks in {insert_time:.2f}s")
        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query FAISS for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        import faiss

        # Set nprobe for IVF (number of clusters to search)
        if isinstance(self.index, faiss.IndexIVF):
            nprobe = self.db_config.get('nprobe', 10)
            self.index.nprobe = nprobe

        start_time = time.time()

        # Search (FAISS expects 2D array)
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        query_time = time.time() - start_time

        # Extract chunk numbers from metadata and convert distances to similarities
        result_ids = []
        similarity_scores = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.metadata_store:  # -1 means not found
                chunk_num = self.metadata_store[idx]['chunk_num']
                result_ids.append(chunk_num)
                # FAISS with L2 distance: convert to similarity (1 / (1 + distance))
                # For cosine similarity index, distances are already similarity scores
                # Assuming cosine here (IP with normalized vectors)
                similarity_scores.append(float(dist))

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up FAISS resources."""
        self.index = None
        self.metadata_store = {}
        print("Cleaned up FAISS index")

    def save_index(self, path: str = None) -> None:
        """Save FAISS index and metadata to disk."""
        import faiss
        import pickle

        path = path or self.index_path
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save index
        index_file = f"{path}/index.faiss"
        faiss.write_index(self.index, index_file)

        # Save metadata
        metadata_file = f"{path}/metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata_store, f)

        print(f"Saved FAISS index to {path}")

    def load_index(self, path: str = None) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss
        import pickle

        path = path or self.index_path

        # Load index
        index_file = f"{path}/index.faiss"
        self.index = faiss.read_index(index_file)

        # Load metadata
        metadata_file = f"{path}/metadata.pkl"
        with open(metadata_file, 'rb') as f:
            self.metadata_store = pickle.load(f)

        print(f"Loaded FAISS index from {path}")
