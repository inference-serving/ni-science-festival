import cv2
import torch
import streamlit as st
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Streamlit app setup for object detection
st.title("Real-time Object Detection on Jetson Nano")
st.write("Using YOLOv5 Nano model")

# Open camera
cap = cv2.VideoCapture(0)

# Placeholder for the video feed
stframe = st.empty()

# Start capturing and processing frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from camera.")
        break

    # Perform inference using YOLOv5 model
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Convert BGR to RGB for Streamlit display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the annotated frame in Streamlit
    stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    # Stop button to break the loop
    if st.button("Stop"):
        break

# Release the camera resource
cap.release()
