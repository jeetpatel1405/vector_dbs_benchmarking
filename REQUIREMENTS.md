# Requirements Document: Benchmarking Open-Source Vector Databases for RAG

## 1. Executive Summary

This document outlines the functional and non-functional requirements for a benchmarking system that evaluates open-source vector databases in the context of Retrieval-Augmented Generation (RAG) systems. The project aims to provide empirical performance data across ingestion and query operations to guide production system design decisions.

**Project Status:** Accepted for Journal Publication

## 2. Project Objectives

### 2.1 Primary Objective
Benchmark open-source vector databases integrated with interactive interfaces (specifically Open WebUI) across three critical dimensions:
- Efficiency
- Scalability
- Accuracy in interactive knowledge retrieval applications

### 2.2 Secondary Objectives
- Provide practical guidance for developers, researchers, and architects building scalable RAG applications
- Establish baseline metrics for ongoing research into performance optimization
- Reveal trade-offs between ingestion efficiency and query accuracy
- Inform design decisions for production RAG systems

## 3. Scope

### 3.1 In Scope
- Open-source vector databases
- Integration with Open WebUI platform
- Document ingestion performance metrics
- Query-level performance metrics
- Variable chunk size configurations
- System resource consumption analysis
- Comparative performance analysis

### 3.2 Out of Scope
- Proprietary/closed-source vector databases (except for baseline comparisons)
- Non-RAG use cases
- Production deployment infrastructure
- End-user application development
- Real-time streaming ingestion scenarios

## 4. Functional Requirements

### 4.1 Vector Database Support

**REQ-DB-001:** The system SHALL support the following vector databases:
- ChromaDB
- FAISS (Facebook AI Similarity Search)
- Qdrant
- pgvector (PostgreSQL extension)
- Pinecone
- Milvus
- Weaviate

**REQ-DB-002:** The system SHALL provide a unified interface for interacting with different vector databases.

**REQ-DB-003:** Each vector database integration SHALL support both ingestion and retrieval operations.

### 4.2 Document Ingestion

**REQ-ING-001:** The system SHALL support loading documents from a specified directory.

**REQ-ING-002:** The system SHALL parse documents and split them into chunks using configurable parameters.

**REQ-ING-003:** The system SHALL support variable chunk sizes for document processing.

**REQ-ING-004:** The system SHALL support configurable chunk overlap settings.

**REQ-ING-005:** The system SHALL measure and record parsing duration for each document.

**REQ-ING-006:** The system SHALL measure and record embedding generation duration for document chunks.

**REQ-ING-007:** The system SHALL track total ingestion time per document.

**REQ-ING-008:** The system SHALL persist ingested knowledge bases for subsequent querying.

### 4.3 Query Processing

**REQ-QRY-001:** The system SHALL support structured knowledge base querying.

**REQ-QRY-002:** The system SHALL measure query latency (response time).

**REQ-QRY-003:** The system SHALL measure retrieval recall metrics.

**REQ-QRY-004:** The system SHALL measure retrieval precision metrics.

**REQ-QRY-005:** The system SHALL support similarity-based answer accuracy validation.

**REQ-QRY-006:** The system SHALL support gold standard answer comparison.

**REQ-QRY-007:** The system SHALL format retrieved context into RAG prompts for LLM generation.

### 4.4 Embedding Generation

**REQ-EMB-001:** The system SHALL use consistent embedding models across all vector databases.

**REQ-EMB-002:** The system SHALL support configurable embedding model selection.

**REQ-EMB-003:** The default embedding model SHALL be `sentence-transformers/all-MiniLM-L6-v2`.

**REQ-EMB-004:** The system SHALL track embedding generation time.

### 4.5 LLM Integration

**REQ-LLM-001:** The system SHALL integrate with Ollama for local LLM inference.

**REQ-LLM-002:** The system SHALL support configurable LLM model selection.

**REQ-LLM-003:** The default LLM SHALL be Mistral via Ollama.

**REQ-LLM-004:** The system SHALL format retrieved context and queries into appropriate prompts.

**REQ-LLM-005:** The system SHALL measure LLM generation time separately from retrieval time.

### 4.6 Metrics Collection

**REQ-MET-001:** The system SHALL collect the following ingestion metrics:
- Document parsing time
- Embedding generation time
- Total ingestion time
- Number of chunks created
- Storage space consumed

**REQ-MET-002:** The system SHALL collect the following query metrics:
- Query latency (end-to-end)
- Retrieval time
- Number of documents retrieved
- Recall score
- Precision score
- Accuracy score (similarity-based)

**REQ-MET-003:** The system SHALL collect the following resource consumption metrics:
- CPU utilization
- Memory usage
- Disk I/O
- Network I/O (where applicable)

**REQ-MET-004:** The system SHALL export metrics in machine-readable format (CSV, JSON).

**REQ-MET-005:** The system SHALL support real-time metrics logging during experiments.

### 4.7 Test Case Management

**REQ-TST-001:** The system SHALL support structured test cases with:
- Query text
- Expected/gold standard answer

**REQ-TST-002:** The system SHALL execute test cases consistently across all vector databases.

**REQ-TST-003:** The system SHALL validate responses against gold standard answers using:
- Cosine similarity of embeddings
- Configurable accuracy threshold (default: 0.8)

### 4.8 Pipeline Server Architecture

**REQ-PIP-001:** The system SHALL implement a custom Open WebUI pipeline server.

**REQ-PIP-002:** The pipeline server SHALL handle both document ingestion and query processing.

**REQ-PIP-003:** The pipeline server SHALL replicate authentic operational conditions.

**REQ-PIP-004:** The pipeline server SHALL support systematic document loading.

**REQ-PIP-005:** The pipeline server SHALL isolate database-specific operations to prevent cross-contamination.

## 5. Non-Functional Requirements

### 5.1 Performance

**REQ-NFR-001:** The benchmarking system SHALL minimize overhead to avoid skewing database performance metrics.

**REQ-NFR-002:** The system SHALL complete ingestion benchmarks for datasets up to 10GB within reasonable time limits.

**REQ-NFR-003:** Query latency measurements SHALL have millisecond precision.

**REQ-NFR-004:** The system SHALL support concurrent query execution for throughput testing.

### 5.2 Scalability

**REQ-NFR-005:** The system SHALL support benchmarking with knowledge bases containing up to 1 million documents.

**REQ-NFR-006:** The system SHALL support chunk sizes ranging from 128 to 4096 tokens.

**REQ-NFR-007:** The system SHALL handle multiple chunk size configurations in a single experimental run.

### 5.3 Reproducibility

**REQ-NFR-008:** All experimental configurations SHALL be version-controlled.

**REQ-NFR-009:** The system SHALL document all dependencies and version numbers.

**REQ-NFR-010:** Random seeds SHALL be configurable for reproducible experiments.

**REQ-NFR-011:** The system SHALL generate experiment metadata for each benchmark run.

### 5.4 Reliability

**REQ-NFR-012:** The system SHALL handle database connection failures gracefully.

**REQ-NFR-013:** The system SHALL retry failed operations with exponential backoff.

**REQ-NFR-014:** The system SHALL log all errors with sufficient context for debugging.

**REQ-NFR-015:** Incomplete benchmark runs SHALL be recoverable without full restart.

### 5.5 Usability

**REQ-NFR-016:** The system SHALL provide clear configuration files for experimental parameters.

**REQ-NFR-017:** The system SHALL generate human-readable summary reports.

**REQ-NFR-018:** The system SHALL provide visualization scripts for comparative analysis.

**REQ-NFR-019:** Documentation SHALL include setup instructions for all supported databases.

### 5.6 Maintainability

**REQ-NFR-020:** The codebase SHALL follow PEP 8 style guidelines for Python.

**REQ-NFR-021:** Each database integration SHALL be modular and independently testable.

**REQ-NFR-022:** The system SHALL use dependency injection for database clients.

**REQ-NFR-023:** Configuration SHALL be externalized from code.

### 5.7 Portability

**REQ-NFR-024:** The system SHALL run on Linux, macOS, and Windows.

**REQ-NFR-025:** The system SHALL support containerized deployment via Docker.

**REQ-NFR-026:** Database dependencies SHALL be isolated using virtual environments or containers.

## 6. Technical Specifications

### 6.1 Programming Language
- **Primary:** Python 3.8+
- **Rationale:** Extensive ML/AI library ecosystem, LangChain support

### 6.2 Core Dependencies
- LangChain (document processing, vector store abstractions)
- HuggingFace Transformers (embeddings)
- Ollama (local LLM inference)
- NumPy (numerical computations)
- Scikit-learn (similarity metrics)
- Pydantic (data validation)

### 6.3 Vector Database Clients
- `chromadb` - Chroma client
- `faiss-cpu` or `faiss-gpu` - FAISS library
- `qdrant-client` - Qdrant client
- `opensearch-py` - OpenSearch client
- `psycopg2` + `pgvector` - PostgreSQL with vector extension
- `pinecone-client` - Pinecone client

### 6.4 Data Storage
- **Configuration:** YAML or JSON files
- **Metrics Output:** CSV for tabular data, JSON for structured logs
- **Vector Stores:** Database-specific persistence (varies by DB)

### 6.5 Experimental Configuration

**Default Parameters:**
- Chunk Size: 1024 tokens
- Chunk Overlap: 128 tokens
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: Mistral via Ollama
- Accuracy Threshold: 0.8 (cosine similarity)
- Top-K Retrieval: 5 documents

**Variable Parameters (for experimentation):**
- Chunk Sizes: [128, 256, 512, 1024, 2048, 4096]
- Top-K Values: [1, 3, 5, 10, 20]
- Embedding Models: Multiple options for sensitivity analysis

## 7. Performance Metrics Definitions

### 7.1 Ingestion Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| Parsing Time | Time to load and parse document | seconds |
| Embedding Time | Time to generate embeddings for all chunks | seconds |
| Indexing Time | Time to insert vectors into database | seconds |
| Total Ingestion Time | End-to-end time for document ingestion | seconds |
| Chunks Created | Number of text chunks generated | count |
| Storage Size | Disk space consumed by vector index | MB/GB |

### 7.2 Query Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| Query Latency | End-to-end time from query to response | milliseconds |
| Retrieval Time | Time to retrieve relevant documents | milliseconds |
| Generation Time | LLM inference time | milliseconds |
| Recall | Proportion of relevant documents retrieved | 0.0-1.0 |
| Precision | Proportion of retrieved documents that are relevant | 0.0-1.0 |
| Accuracy | Cosine similarity of response to gold answer | 0.0-1.0 |
| Documents Retrieved | Number of chunks returned | count |

### 7.3 Resource Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| CPU Usage | Average CPU utilization during operation | percentage |
| Memory Usage | Peak memory consumption | MB/GB |
| Disk I/O | Read/write operations per second | ops/sec |
| Network I/O | Data transferred (for remote databases) | MB/s |

## 8. Experimental Design

### 8.1 Test Scenarios

**Scenario 1: Baseline Ingestion**
- Single document ingestion with default parameters
- Measure parsing, embedding, and indexing time
- Compare across all vector databases

**Scenario 2: Chunk Size Sensitivity**
- Vary chunk size: [128, 256, 512, 1024, 2048, 4096]
- Measure impact on ingestion time and query accuracy
- Fixed document set

**Scenario 3: Query Performance**
- Fixed knowledge base
- Structured test cases with gold answers
- Measure latency, recall, precision, accuracy
- Compare across all vector databases

**Scenario 4: Scalability Testing**
- Gradually increase knowledge base size
- Measure ingestion time growth rate
- Measure query latency degradation
- Identify inflection points

**Scenario 5: Resource Consumption**
- Monitor CPU, memory, disk, network during operations
- Identify resource bottlenecks
- Compare resource efficiency across databases

### 8.2 Controlled Variables
- Same hardware environment
- Same embedding model
- Same LLM for generation
- Same document corpus
- Same test queries

### 8.3 Output Deliverables
- Raw metrics data (CSV/JSON)
- Comparative visualizations (plots)
- Statistical analysis summary
- Performance trade-off matrices
- Configuration recommendations

## 9. Data Requirements

### 9.1 Document Corpus
- **Format:** Plain text (.txt) files
- **Size:** Variable (from KB to GB scale)
- **Domain:** Representative of knowledge base use cases
- **Location:** `docs/` directory

### 9.2 Test Queries
- **Format:** Structured test cases (query + gold answer)
- **Coverage:** Representative of real user queries
- **Difficulty:** Range from simple factual to complex reasoning
- **Validation:** Gold answers verified for correctness

### 9.3 Benchmark Results
- **Storage:** `Data/Analysis/` directory
- **Format:** CSV for metrics, PNG for plots
- **Retention:** All experimental runs preserved with metadata

## 10. Visualization Requirements

**REQ-VIZ-001:** The system SHALL generate comparison plots for:
- Ingestion time by database and chunk size
- Query latency distributions
- Recall/precision scatter plots
- Resource consumption over time
- Accuracy vs. latency trade-offs

**REQ-VIZ-002:** Plots SHALL be publication-ready quality (300 DPI minimum).

**REQ-VIZ-003:** Plots SHALL use consistent color schemes and legends.

**REQ-VIZ-004:** The system SHALL support export to PNG, PDF, and SVG formats.

## 11. Success Criteria

The project will be considered successful when:

1. All six vector databases are successfully integrated and benchmarked
2. Reproducible experiments demonstrate consistent results (±5% variance)
3. Statistical significance is achieved for comparative claims (p < 0.05)
4. Performance trade-offs are clearly documented and visualized
5. Results provide actionable guidance for database selection
6. Baseline metrics enable future research comparisons
7. Findings are accepted for peer-reviewed publication

## 12. Constraints and Assumptions

### 12.1 Constraints
- Limited to open-source vector databases
- Experiments run on single-node hardware (no distributed systems)
- Evaluation focused on RAG use cases specifically
- Integration limited to Open WebUI platform

### 12.2 Assumptions
- Ollama service is available and operational
- Sufficient compute resources for embedding generation
- Network connectivity for databases requiring external services
- Document corpus is representative of production knowledge bases
- Test queries reflect authentic user information needs

## 13. Dependencies

### 13.1 External Services
- Ollama server (local or remote)
- Vector database servers (may be containerized)
- Embedding model downloads from HuggingFace

### 13.2 Infrastructure
- Docker (for database containerization)
- Python virtual environment
- Sufficient disk space for vector indexes
- GPU acceleration (optional, for faster embeddings)

## 14. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Database version incompatibilities | High | Medium | Pin dependency versions, use containers |
| Resource exhaustion during large-scale tests | Medium | Medium | Implement batch processing, monitoring |
| Inconsistent timing measurements | High | Low | Use high-precision timers, multiple runs |
| Network latency affecting remote DBs | Medium | Medium | Isolate network I/O, use local deployments |
| LLM availability/performance variance | Medium | Low | Cache embeddings, use consistent Ollama config |

## 15. Future Enhancements (Out of Current Scope)

- Multi-node distributed database configurations
- Cloud-hosted vector database services
- Additional embedding models (OpenAI, Cohere)
- Hybrid search (dense + sparse vectors)
- Advanced RAG techniques (HyDE, multi-hop reasoning)
- Real-time ingestion benchmarking
- Multi-modal document processing (images, tables)
- User study on result quality perception

## 16. References

- Project Repository: https://github.com/Kwaai-AI-Lab/vector_dbs_benchmarking
- Open WebUI Documentation
- LangChain Documentation
- Individual vector database documentation (Chroma, FAISS, Qdrant, OpenSearch, pgvector, Pinecone)

## 17. Appendix: Configuration Examples

### 17.1 Environment Variables
```bash
GEMINI_API_KEY=<optional_for_alternate_embedding>
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
ACCURACY_THRESHOLD=0.8
```

### 17.2 Test Case Structure
```python
class TestCase(BaseModel):
    query: str
    gold_answer: str
```

### 17.3 Metrics Output Schema
```json
{
  "database": "chroma",
  "chunk_size": 1024,
  "ingestion": {
    "parsing_time_sec": 12.5,
    "embedding_time_sec": 45.3,
    "indexing_time_sec": 8.2,
    "total_time_sec": 66.0,
    "chunks_created": 1523
  },
  "query": {
    "test_case_id": "tc_001",
    "query_latency_ms": 234,
    "retrieval_time_ms": 45,
    "generation_time_ms": 189,
    "recall": 0.85,
    "precision": 0.92,
    "accuracy": 0.88,
    "docs_retrieved": 5
  },
  "resources": {
    "avg_cpu_percent": 42.3,
    "peak_memory_mb": 2048,
    "disk_io_ops_sec": 125
  }
}
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-22
**Status:** Active
**Approval:** Journal Accepted (Abstract)
