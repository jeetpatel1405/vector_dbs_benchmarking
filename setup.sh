#!/bin/bash
# Setup script for Vector Database Benchmarking System
# This script installs all dependencies and prepares the environment

set -e  # Exit on error

echo "=========================================="
echo "Vector Database Benchmarking Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

required_version="3.8"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi
echo "✅ Python version OK"
echo ""

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
echo "This may take a few minutes..."
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p results
mkdir -p vector_stores/chroma_db
mkdir -p vector_stores/faiss_index
mkdir -p Plots
echo "✅ Directories created"
echo ""

# Verify imports
echo "Verifying imports..."
python3 verify_imports.py

if [ $? -eq 0 ]; then
    echo "✅ All imports verified successfully"
else
    echo "⚠️  Some imports failed - check verify_imports.py output above"
fi
echo ""

# Check Docker availability
echo "Checking Docker availability..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"

    if docker ps &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "⚠️  Docker is installed but daemon is not running"
        echo "   Start Docker to use: pgvector, Qdrant, Weaviate, Milvus, OpenSearch"
    fi
else
    echo "⚠️  Docker is not installed"
    echo "   Install Docker to use: pgvector, Qdrant, Weaviate, Milvus, OpenSearch"
    echo "   You can still use FAISS and Chroma (embedded databases)"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Test embedded databases (no Docker required):"
echo "   python test_adapters.py"
echo ""
echo "3. Start Docker services for full testing:"
echo "   docker-compose up -d"
echo "   sleep 30  # Wait for services to be ready"
echo ""
echo "4. Test all databases:"
echo "   python test_adapters.py"
echo ""
echo "5. Run benchmark:"
echo "   python -m src.benchmark_runner --config configs/default.yaml"
echo ""
