#!/bin/bash
# Test script for Vector Database Benchmarking System
# Runs comprehensive tests on all adapters

set -e  # Exit on error

echo "=========================================="
echo "Vector Database Benchmarking Tests"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Run ./setup.sh first"
        exit 1
    fi
fi
echo ""

# Check if dependencies are installed
echo "Verifying dependencies..."
python3 -c "import numpy, yaml, psutil" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Core dependencies available"
else
    echo "❌ Dependencies missing. Run ./setup.sh first"
    exit 1
fi
echo ""

# Test 1: Verify imports
echo "Test 1: Verifying all imports..."
python3 verify_imports.py

if [ $? -eq 0 ]; then
    echo "✅ Import verification passed"
else
    echo "❌ Import verification failed"
    exit 1
fi
echo ""

# Test 2: Test embedded databases (no Docker required)
echo "Test 2: Testing embedded databases (FAISS, Chroma)..."
echo "These don't require Docker and should always work."
echo ""
python3 test_adapters.py 2>&1 | grep -E "(Testing|SUCCESS|FAILED|SUMMARY)" || true
echo ""

# Test 3: Check Docker services
echo "Test 3: Checking Docker services..."
docker_available=false

if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo "✅ Docker is available"
    docker_available=true

    # Check which services are running
    echo ""
    echo "Checking database services:"

    # Qdrant (port 6333)
    if nc -z localhost 6333 2>/dev/null; then
        echo "✅ Qdrant is running (port 6333)"
    else
        echo "⚠️  Qdrant is not running (port 6333)"
        echo "   Start with: docker-compose up -d qdrant"
    fi

    # pgvector (port 5432)
    if nc -z localhost 5432 2>/dev/null; then
        echo "✅ pgvector is running (port 5432)"
    else
        echo "⚠️  pgvector is not running (port 5432)"
        echo "   Start with: docker-compose up -d pgvector"
    fi

    # Weaviate (port 8080)
    if nc -z localhost 8080 2>/dev/null; then
        echo "✅ Weaviate is running (port 8080)"
    else
        echo "⚠️  Weaviate is not running (port 8080)"
        echo "   Start with: docker-compose up -d weaviate"
    fi

    # Milvus (port 19530)
    if nc -z localhost 19530 2>/dev/null; then
        echo "✅ Milvus is running (port 19530)"
    else
        echo "⚠️  Milvus is not running (port 19530)"
        echo "   Start with: docker-compose up -d milvus"
    fi

    # OpenSearch (port 9200)
    if nc -z localhost 9200 2>/dev/null; then
        echo "✅ OpenSearch is running (port 9200)"
    else
        echo "⚠️  OpenSearch is not running (port 9200)"
        echo "   Start with: docker-compose up -d opensearch"
    fi
else
    echo "⚠️  Docker is not available"
    echo "   Install and start Docker to test all databases"
fi
echo ""

# Test 4: Full adapter tests
echo "Test 4: Running full adapter test suite..."
echo ""
python3 test_adapters.py
test_result=$?
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ $test_result -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "Your benchmarking system is ready to use."
    exit 0
else
    echo "⚠️  Some tests failed"
    echo ""
    echo "Common issues:"
    echo "1. Docker services not running → Run: docker-compose up -d"
    echo "2. Dependencies missing → Run: ./setup.sh"
    echo "3. Port conflicts → Check if ports 5432, 6333, 8080, 9200, 19530 are free"
    echo ""
    exit 1
fi
