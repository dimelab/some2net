#!/bin/bash
# Clear all Python cache and restart Streamlit

echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

echo "🧹 Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null

echo "✅ Cache cleared!"
echo ""
echo "Now restart Streamlit with:"
echo "  streamlit run src/cli/app.py"
