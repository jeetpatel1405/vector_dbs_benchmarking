"""OpenSearch RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class OpenSearchRAGBenchmark(RAGBenchmark):
    """OpenSearch implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize OpenSearch RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - host: OpenSearch host
                - port: OpenSearch port
                - index_name: Index name
                - index_method: 'hnsw' or 'ivf'
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 9200)
        self.index_name = db_config.get('index_name', 'rag_benchmark')
        self.index_method = db_config.get('index_method', 'hnsw')

        self.client = None

    def connect(self) -> None:
        """Connect to OpenSearch."""
        try:
            from opensearchpy import OpenSearch

            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False
            )

            info = self.client.info()
            print(f"Connected to OpenSearch {info['version']['number']} at {self.host}:{self.port}")

        except Exception as e:
            print(f"Failed to connect to OpenSearch: {e}")
            raise

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

        # Define index mapping with k-NN
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
                    "chunk_num": {"type": "integer"},
                    "source": {"type": "keyword"}
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
                    "chunk_num": chunk.metadata.get('chunk_num', i),
                    "source": chunk.metadata.get('source', '')
                }
            }
            actions.append(action)

        # Bulk insert
        success, failed = helpers.bulk(self.client, actions, chunk_size=batch_size, raise_on_error=False)

        # Refresh index to make documents searchable
        self.client.indices.refresh(index=self.index_name)

        insert_time = time.time() - start_time
        print(f"Inserted {success} chunks in {insert_time:.2f}s")
        if failed:
            print(f"Warning: {len(failed)} chunks failed to insert")

        return insert_time

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query OpenSearch for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
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

        # Extract chunk numbers and scores from results
        result_ids = []
        similarity_scores = []

        for hit in response['hits']['hits']:
            chunk_num = hit['_source'].get('chunk_num', 0)
            result_ids.append(chunk_num)

            # OpenSearch with cosinesimil returns: _score = 1 / (2 - cosine_similarity)
            # Convert back to cosine similarity: cosine_similarity = 2 - (1 / _score)
            score = hit.get('_score', 0.0)
            if score > 0:
                cosine_similarity = 2.0 - (1.0 / score)
                # Clamp to [0, 1] range (should already be in range, but safety check)
                cosine_similarity = max(0.0, min(1.0, cosine_similarity))
            else:
                cosine_similarity = 0.0
            similarity_scores.append(float(cosine_similarity))

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up OpenSearch resources."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"Deleted index: {self.index_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
