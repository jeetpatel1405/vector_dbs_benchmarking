"""pgvector RAG benchmark implementation."""
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

from src.vector_dbs.rag_benchmark import RAGBenchmark
from src.utils.chunking import Chunk


class PgvectorRAGBenchmark(RAGBenchmark):
    """pgvector (PostgreSQL) implementation of RAG benchmark."""

    def __init__(self, db_config: Dict[str, Any], embedding_generator, **kwargs):
        """
        Initialize pgvector RAG benchmark.

        Args:
            db_config: Configuration dict with keys:
                - host: PostgreSQL host
                - port: PostgreSQL port
                - database: Database name
                - user: Database user
                - password: Database password
                - table_name: Table name
                - index_type: 'ivfflat' or 'hnsw'
            embedding_generator: Embedding generator instance
            **kwargs: Additional arguments for RAGBenchmark
        """
        super().__init__(db_config, embedding_generator, **kwargs)

        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 5432)
        self.database = db_config.get('database', 'vectordb')
        self.user = db_config.get('user', 'postgres')
        self.password = db_config.get('password', 'postgres')
        self.table_name = db_config.get('table_name', 'rag_benchmark')
        self.index_type = db_config.get('index_type', 'ivfflat')

        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor()

            # Enable pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()

            print(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")

        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Disconnected from PostgreSQL")

    def create_collection(self, dimension: int) -> None:
        """Create table for vectors."""
        # Drop table if exists
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.conn.commit()

        # Create table with vector column and metadata
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                id SERIAL PRIMARY KEY,
                embedding vector({dimension}),
                text TEXT,
                chunk_id TEXT,
                chunk_num INTEGER,
                doc_id TEXT,
                source TEXT,
                strategy TEXT,
                start_index INTEGER,
                end_index INTEGER
            )
        """)
        self.conn.commit()

        print(f"Created table '{self.table_name}' with vector({dimension})")

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> float:
        """Insert chunks into PostgreSQL."""
        start_time = time.time()

        # Prepare data for batch insert
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append((
                embedding.tolist(),
                chunk.text,
                chunk.id,
                chunk.metadata.get('chunk_num', i),
                chunk.metadata.get('doc_id', ''),
                chunk.metadata.get('source', ''),
                chunk.metadata.get('strategy', ''),
                chunk.start_index,
                chunk.end_index
            ))

        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            execute_batch(
                self.cursor,
                f"""
                INSERT INTO {self.table_name}
                (embedding, text, chunk_id, chunk_num, doc_id, source, strategy, start_index, end_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                batch
            )
            if (i + batch_size) % 1000 == 0:
                print(f"Inserted {min(i + batch_size, len(data))}/{len(data)} rows")

        self.conn.commit()

        # Create vector index
        self._create_vector_index(embeddings.shape[1])

        insert_time = time.time() - start_time
        print(f"Inserted {len(data)} rows in {insert_time:.2f}s")

        return insert_time

    def _create_vector_index(self, dimension: int) -> None:
        """Create vector index after data insertion."""
        print(f"Creating {self.index_type} index...")
        start_time = time.time()

        if self.index_type == 'ivfflat':
            # IVFFlat index
            lists = self.db_config.get('lists', 100)
            self.cursor.execute(f"""
                CREATE INDEX ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
        elif self.index_type == 'hnsw':
            # HNSW index
            m = self.db_config.get('m', 16)
            ef_construction = self.db_config.get('ef_construction', 64)
            self.cursor.execute(f"""
                CREATE INDEX ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
            """)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.conn.commit()
        index_time = time.time() - start_time
        print(f"Created {self.index_type} index in {index_time:.2f}s")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float, List[float]]:
        """
        Query PostgreSQL for similar chunks.

        Returns:
            Tuple of (result_ids, query_time, similarity_scores)
            - result_ids: List of chunk IDs
            - query_time: Time taken for query in seconds
            - similarity_scores: Cosine similarity scores for each result (0-1)
        """
        # Set probes for IVFFlat
        if self.index_type == 'ivfflat':
            probes = self.db_config.get('probes', 10)
            self.cursor.execute(f"SET ivfflat.probes = {probes}")

        start_time = time.time()

        # Query using cosine distance (cast array to vector type)
        # Return both chunk_num and distance
        self.cursor.execute(f"""
            SELECT chunk_num, embedding <=> %s::vector as distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT %s
        """, (query_embedding.tolist(), top_k))

        query_time = time.time() - start_time

        # Extract chunk numbers and convert distances to similarities
        results = self.cursor.fetchall()
        result_ids = [row[0] for row in results]
        # Cosine distance: convert to similarity (1 - distance)
        similarity_scores = [float(1.0 - row[1]) for row in results]

        return result_ids, query_time, similarity_scores

    def cleanup(self) -> None:
        """Clean up PostgreSQL resources."""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self.conn.commit()
            print(f"Dropped table: {self.table_name}")
        except Exception as e:
            print(f"Cleanup error: {e}")
