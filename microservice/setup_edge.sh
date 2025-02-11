#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Step 1: Install necessary packages
echo "Updating package lists..."
sudo apt-get update

# Step 2: Install Python3 and pip if not installed
echo "Installing Python3 and pip..."
sudo apt-get install -y python3 python3-pip

# Step 3: Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Step 4: Install required Python libraries
echo "Installing Python dependencies..."
pip3 install -r requirements_edge.txt

# Step 5: Disable firewalld if active (if using firewalld)
if systemctl is-active --quiet firewalld; then
    echo "Disabling firewalld to prevent port blocking..."
    sudo systemctl stop firewalld
    sudo systemctl disable firewalld
else
    echo "firewalld is not running."
fi

# Step 6: Open ports for Streamlit (8501) and Ollama (11434)
echo "Ensuring required ports are open..."
sudo ufw allow 8501/tcp || true
sudo ufw allow 11434/tcp || true

# Step 7: Start Ollama Server in Background
echo "Starting Ollama server..."
nohup ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Give Ollama some time to start up
sleep 5
echo "Ollama server started with PID: $OLLAMA_PID"

# Step 8: Run the Streamlit App
echo "Starting the Streamlit app..."
streamlit run edge_chatbot_app.py

# Step 9: Cleanup - Stop Ollama Server when Streamlit exits
echo "Stopping Ollama server..."
kill $OLLAMA_PID
