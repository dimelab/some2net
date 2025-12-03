#!/bin/bash

# Kill any existing streamlit processes
pkill -f "streamlit run" 2>/dev/null

# Wait a moment
sleep 1

# Run streamlit with explicit config
echo "Starting Streamlit app with configuration:"
echo "- Max upload size: 2000MB (2GB)"
echo "- Max message size: 2000MB"
echo ""

streamlit run src/cli/app.py \
    --server.maxUploadSize=2000 \
    --server.maxMessageSize=2000 \
    --server.port=8501

