# Docker Setup Guide

Complete Docker Compose setup for all 7 vector databases used in benchmarking.

---

## Services Overview

| Service | Container Name | Ports | Dependencies | Status |
|---------|---------------|-------|--------------|--------|
| **pgvector** | pgvector-benchmark | 5433 | None | ✅ Ready |
| **Qdrant** | qdrant-benchmark | 6333, 6334 | None | ✅ Ready |
| **Weaviate** | weaviate-benchmark | 8080, 50051 | None | ✅ Ready |
| **Milvus** | milvus-standalone | 19530, 9091 | etcd, minio | ✅ Ready |
| **OpenSearch** | opensearch-benchmark | 9200, 9600 | None | ✅ Ready |
| **Chroma** | chroma-benchmark | 8000 | None | ✅ Ready |
| **FAISS** | - | - | None (embedded) | ✅ Ready |

### Supporting Services

| Service | Container Name | Purpose | Ports |
|---------|---------------|---------|-------|
| **etcd** | milvus-etcd | Milvus metadata storage | - |
| **MinIO** | milvus-minio | Milvus object storage | 9000, 9001 |

---

## Quick Start

### 1. Start All Services

```bash
# Start all vector databases
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Wait for Services to be Ready

```bash
# Check health of all services
docker-compose ps

# All services should show "healthy" status
# This may take 1-2 minutes, especially for Milvus
```

### 3. Verify Connectivity

```bash
# Test each service
curl http://localhost:5433  # pgvector (PostgreSQL)
curl http://localhost:6333  # Qdrant
curl http://localhost:8080/v1/.well-known/ready  # Weaviate
curl http://localhost:19530  # Milvus (will timeout - no HTTP endpoint)
curl http://localhost:9200  # OpenSearch
curl http://localhost:8000/api/v1/heartbeat  # Chroma
```

---

## Individual Service Management

### Start Specific Services

```bash
# Start only pgvector and Qdrant
docker-compose up -d pgvector qdrant

# Start Milvus and its dependencies
docker-compose up -d etcd minio milvus

# Start OpenSearch
docker-compose up -d opensearch
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop but keep volumes (data persists)
docker-compose stop

# Stop and remove volumes (CAUTION: deletes all data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f milvus
docker-compose logs -f opensearch

# Last 100 lines
docker-compose logs --tail=100 milvus
```

---

## Service Details

### pgvector (PostgreSQL with vector extension)

```yaml
Host: localhost
Port: 5433 (mapped from 5432 to avoid conflicts)
Database: vectordb
User: postgres
Password: postgres
```

**Connection String:**
```
postgresql://postgres:postgres@localhost:5433/vectordb
```

**Test Connection:**
```bash
docker exec -it pgvector-benchmark psql -U postgres -d vectordb -c "SELECT version();"
```

---

### Qdrant

```yaml
Host: localhost
HTTP Port: 6333
gRPC Port: 6334
```

**API Endpoints:**
- Health: http://localhost:6333/
- Collections: http://localhost:6333/collections
- Dashboard: http://localhost:6333/dashboard

**Test:**
```bash
curl http://localhost:6333/
```

---

### Weaviate

```yaml
Host: localhost
HTTP Port: 8080
gRPC Port: 50051
```

**API Endpoints:**
- Ready: http://localhost:8080/v1/.well-known/ready
- Live: http://localhost:8080/v1/.well-known/live
- Schema: http://localhost:8080/v1/schema

**Test:**
```bash
curl http://localhost:8080/v1/.well-known/ready
```

**Notes:**
- Anonymous access enabled for benchmarking
- No vectorizer modules (using external embeddings)

---

### Milvus

```yaml
Host: localhost
Port: 19530 (gRPC)
Admin Port: 9091
```

**Dependencies:**
- etcd: Metadata storage
- MinIO: Object storage (ports 9000, 9001)

**Health Check:**
```bash
curl http://localhost:9091/healthz
```

**MinIO Console:**
- URL: http://localhost:9001
- Username: minioadmin
- Password: minioadmin

**Test:**
```python
from pymilvus import connections
connections.connect(host='localhost', port='19530')
```

---

### OpenSearch

```yaml
Host: localhost
Port: 9200 (HTTP)
Performance Analyzer: 9600
```

**API Endpoints:**
- Health: http://localhost:9200/_cluster/health
- Info: http://localhost:9200/
- Plugins: http://localhost:9200/_cat/plugins

**Test:**
```bash
curl http://localhost:9200/
curl http://localhost:9200/_cluster/health
```

**Notes:**
- Security plugin disabled for benchmarking
- k-NN plugin pre-installed
- 512MB heap size (adjust for production)

---

### Chroma

```yaml
Host: localhost
Port: 8000
```

**API Endpoints:**
- Heartbeat: http://localhost:8000/api/v2/heartbeat
- Version: http://localhost:8000/api/v1/version
- Collections: http://localhost:8000/api/v1/collections

**Test:**
```bash
curl http://localhost:8000/api/v2/heartbeat
```

**Note:** Chroma v1 API is deprecated, use v2 endpoints where available.

**Notes:**
- Can also run in embedded mode (no Docker required)
- Persistent storage enabled

---

## Resource Requirements

### Minimum Requirements

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| pgvector | 0.5 | 256MB | 1GB |
| Qdrant | 0.5 | 256MB | 1GB |
| Weaviate | 1 | 512MB | 2GB |
| Milvus | 2 | 2GB | 5GB |
| OpenSearch | 1 | 1GB | 2GB |
| Chroma | 0.5 | 256MB | 1GB |
| **Total** | **6** | **4.5GB** | **12GB** |

### Recommended Requirements

- **CPU:** 8+ cores
- **Memory:** 16GB RAM
- **Disk:** 50GB SSD
- **Docker:** 20.10+
- **Docker Compose:** 2.0+

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Check if port is already in use
lsof -i :8080  # Replace with your port

# Restart service
docker-compose restart [service-name]
```

### Out of Memory

```bash
# Check Docker memory limits
docker stats

# Increase Docker Desktop memory allocation (Mac/Windows)
# Settings → Resources → Memory → Increase limit

# For Linux, edit /etc/docker/daemon.json
```

### Milvus Won't Start

Milvus requires both etcd and MinIO to be healthy first.

```bash
# Check dependencies
docker-compose logs etcd
docker-compose logs minio

# Restart Milvus stack
docker-compose restart etcd minio milvus
```

### OpenSearch Memory Lock Error

```bash
# Linux: Increase vm.max_map_count
sudo sysctl -w vm.max_map_count=262144

# Make permanent
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

### Port Conflicts

If ports are already in use, edit `docker-compose.yml`:

```yaml
ports:
  - "5434:5432"  # Change left side only (host port)
```

### Clean Restart

```bash
# Stop everything
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all -v

# Start fresh
docker-compose up -d
```

---

## Performance Tuning

### OpenSearch

```yaml
# Increase heap size for better performance
environment:
  - "OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g"
```

### Milvus

```yaml
# Use more workers for better throughput
environment:
  - MILVUS_PROXY_NUM=4
  - MILVUS_QUERY_NODE_NUM=2
```

### pgvector

```bash
# Increase shared_buffers and work_mem
docker exec -it pgvector-benchmark bash
# Edit postgresql.conf
```

---

## Monitoring

### Docker Stats

```bash
# Real-time resource usage
docker stats

# Specific containers
docker stats pgvector-benchmark qdrant-benchmark
```

### Health Checks

```bash
# Check all service health
docker-compose ps

# Individual health check
docker inspect --format='{{.State.Health.Status}}' milvus-standalone
```

---

## Data Persistence

All services use named volumes for data persistence:

```bash
# List volumes
docker volume ls | grep vector-dbs-benchmarking

# Inspect volume
docker volume inspect vector-dbs-benchmarking_milvus_data

# Backup volume
docker run --rm -v vector-dbs-benchmarking_milvus_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/milvus_backup.tar.gz -C /data .

# Restore volume
docker run --rm -v vector-dbs-benchmarking_milvus_data:/data \
  -v $(pwd):/backup alpine tar xzf /backup/milvus_backup.tar.gz -C /data
```

---

## Environment Variables

All services can be configured via environment variables in the benchmark runner:

```bash
# Example: Run benchmark with custom hosts
export MILVUS_HOST=custom-milvus-host
export MILVUS_PORT=19530
python -m src.benchmark_runner --database milvus
```

See `benchmark-runner` service in `docker-compose.yml` for all available variables.

---

## Network

All services run on the `vector-benchmark-network` bridge network:

```bash
# Inspect network
docker network inspect vector-benchmark-network

# Services can communicate using container names
# Example: pgvector can reach milvus at "milvus:19530"
```

---

## Service URLs Summary

Quick reference for all service endpoints:

```bash
# pgvector
postgresql://postgres:postgres@localhost:5433/vectordb

# Qdrant
http://localhost:6333

# Weaviate
http://localhost:8080

# Milvus
localhost:19530 (gRPC)
http://localhost:9091/healthz (health)

# OpenSearch
http://localhost:9200

# Chroma
http://localhost:8000

# MinIO Console
http://localhost:9001 (minioadmin/minioadmin)
```

---

## Next Steps

1. **Verify Setup:**
   ```bash
   docker-compose up -d
   docker-compose ps  # All should be "healthy"
   ```

2. **Run Tests:**
   ```bash
   python test_adapters.py
   ```

3. **Run Benchmarks:**
   ```bash
   python -m src.benchmark_runner --config configs/default.yaml
   ```

---

**Last Updated:** 2025-10-22
**Docker Compose Version:** 3.8
**Services:** 7 vector databases + 2 dependencies
