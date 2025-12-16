#!/bin/bash
# Script to copy benchmark results to dated folder and commit to git

DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)

DB_NAME=$1
SOURCE_DIR=$2

if [ -z "$DB_NAME" ] || [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <db_name> <source_results_dir>"
    echo "Example: $0 weaviate results/weaviate_experiment_001"
    exit 1
fi

# Create dated results directory
RESULTS_DIR="results/${DB_NAME}_results_${DATE}"
mkdir -p "$RESULTS_DIR"

# Copy results to dated folder
echo "Copying results from $SOURCE_DIR to $RESULTS_DIR..."
cp -r "$SOURCE_DIR"/* "$RESULTS_DIR/" 2>/dev/null || true

# Check if there are any files to commit
if [ ! "$(ls -A $RESULTS_DIR)" ]; then
    echo "No results found in $SOURCE_DIR"
    exit 1
fi

echo "Results copied to $RESULTS_DIR"
ls -lh "$RESULTS_DIR"

# Add to git
echo ""
echo "Adding results to git..."
git add "$RESULTS_DIR"

# Commit
COMMIT_MSG="Add ${DB_NAME} benchmark results - ${DATE}

Results include:
- Query latency and quality metrics
- Performance visualizations
- Configuration used

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git commit -m "$COMMIT_MSG"

# Push
echo ""
echo "Pushing to remote..."
git push

echo ""
echo "✅ Results committed and pushed successfully!"
echo "📁 Results location: $RESULTS_DIR"
