#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Step 1: Install necessary packages
# echo "Updating package lists..."
# sudo apt-get update

# Step 2: Install Python3 and pip if not installed
# echo "Installing Python3 and pip..."
# sudo apt-get install -y python3 python3-pip

# Step 3: Upgrade pip
# echo "Upgrading pip..."
# pip3 install --upgrade pip

# Step 4: Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Step 5: Install required Python libraries
# echo "Installing Python dependencies..."
# pip3 install -r requirements_edge.txt

# Step 6: Configure firewalld to allow Streamlit (8501) and Ollama (11434)
echo "Configuring firewalld rules..."
# sudo firewall-cmd --permanent --add-port=8501/tcp  # Streamlit default port
# sudo firewall-cmd --permanent --add-port=11434/tcp # Ollama default port
# sudo firewall-cmd --reload

# Step 7: Start firewalld if it's not running
# if ! systemctl is-active --quiet firewalld; then
#     echo "Starting firewalld..."
#     sudo systemctl start firewalld
# else
#     echo "firewalld is already running."
# fi

# Step 8: Start Ollama Server in Background
echo "Starting Ollama server..."
nohup ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Give Ollama some time to start up
sleep 5
echo "Ollama server started with PID: $OLLAMA_PID"

# Step 9: Run the Streamlit App
echo "Starting the Streamlit app..."
streamlit run edge_chatbot_app.py

# Step 10: Cleanup - Stop Ollama Server when Streamlit exits
echo "Stopping Ollama server..."
kill $OLLAMA_PID
