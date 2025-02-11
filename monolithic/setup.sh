#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Step 1: Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Step 2: Disable firewalld if active
if systemctl is-active --quiet firewalld; then
    echo "Disabling firewalld to prevent port blocking..."
    sudo systemctl stop firewalld
    sudo systemctl disable firewalld
else
    echo "firewalld is not running."
fi

# Step 3: Confirm ports are accessible (8501 for Streamlit, 11434 for Ollama)
echo "Ensuring required ports are open..."
sudo firewall-cmd --zone=public --add-port=8501/tcp --permanent || true
sudo firewall-cmd --zone=public --add-port=11434/tcp --permanent || true
sudo firewall-cmd --reload || true

# Step 4: Start Ollama Server in Background
echo "Starting Ollama server..."
nohup ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Give Ollama some time to start up
sleep 5
echo "Ollama server started with PID: $OLLAMA_PID"

# Step 5: Run the Streamlit App
echo "Starting the Streamlit app..."
streamlit run chatbot_app.py

# Step 6: Cleanup - Stop Ollama Server when Streamlit exits
echo "Stopping Ollama server..."
kill $OLLAMA_PID
