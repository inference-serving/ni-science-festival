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
# echo "Installing Python dependencies..."
# pip3 install -r requirements_cloud.txt

# Step 5: Disable firewalld if active (Ubuntu systems typically use UFW instead)
if systemctl is-active --quiet ufw; then
    echo "Disabling UFW to prevent port blocking..."
    sudo ufw disable
else
    echo "UFW is not active."
fi

# Step 6: Open port 8000 for FastAPI service
echo "Opening port 8000 for FastAPI..."
sudo ufw allow 8000/tcp || true

# Step 7: Start Ollama Server in the background
echo "Starting Ollama server..."
nohup ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Give Ollama time to start
sleep 5
echo "Ollama server started with PID: $OLLAMA_PID"

# Step 8: Start the FastAPI cloud service
echo "Starting FastAPI cloud microservice..."
nohup uvicorn cloud_chatbot_app:app --host 0.0.0.0 --port 8000 > fastapi_server.log 2>&1 &

echo "Cloud microservice is running at http://$(curl -s ifconfig.me):8000"

# Cleanup will not occur here since we expect the cloud service to be long-running.
