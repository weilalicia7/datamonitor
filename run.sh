#!/bin/bash

echo "============================================"
echo "  SACT Intelligent Scheduling System"
echo "  Velindre Cancer Centre"
echo "============================================"
echo ""
echo "Starting application..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found."
    echo "Please run: python3 -m venv venv"
    echo "Then: pip install -r requirements.txt"
    echo ""
fi

# Run the Streamlit application
streamlit run app.py --server.port 8501 --server.headless false
