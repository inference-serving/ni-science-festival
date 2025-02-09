import cv2
import torch
import time
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px

# Streamlit App Setup
st.title("Real-time Object Detection on Jetson Nano")
st.write("Compare **YOLOv5 Nano** (small) and **YOLOv5 Large** (large) models with live latency metrics.")

# Load YOLOv5 Models (cached for performance)
@st.cache_resource
def load_models():
    small_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    large_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    return small_model, large_model

small_model, large_model = load_models()

# Model Selection Dropdown
model_choice = st.selectbox(
    "Select the Model for Detection:",
    ("YOLOv5 Nano (Small, Faster)", "YOLOv5 Large (Large, Accurate)")
)

# Initialize session state for streaming, latency tracking, and cumulative data
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if 'latencies' not in st.session_state:
    st.session_state.latencies = []  # Latencies for the current session

if 'cumulative_latencies' not in st.session_state:
    st.session_state.cumulative_latencies = []  # Cumulative latencies across sessions

if 'cap' not in st.session_state:
    st.session_state.cap = None  # Initialize camera variable in session state

# Start/Stop Buttons
if st.button("Start Detection"):
    st.session_state.streaming = True
    st.session_state.latencies = []  # Reset latency list on new session
    
    # Initialize the camera only when starting
    st.session_state.cap = cv2.VideoCapture(0)
    if not st.session_state.cap.isOpened():
        st.error("Failed to open camera. Please check your camera connection.")
        st.session_state.streaming = False  # Stop streaming if camera fails

if st.button("Stop Detection"):
    st.session_state.streaming = False
    
    # Properly release the camera resource when stopping
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None  # Reset camera in session state

    # Append current session latencies to cumulative data
    if st.session_state.latencies:
        st.session_state.cumulative_latencies.extend(st.session_state.latencies)

# Placeholders for video feed and latency charts
stframe = st.empty()
latency_chart = st.empty()
average_latency_display = st.empty()
cumulative_latency_display = st.empty()

# Perform real-time object detection
if st.session_state.streaming and st.session_state.cap is not None:
    while st.session_state.streaming:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        # Measure start time for inference
        start_time = time.time()

        # Perform inference based on selected model
        if model_choice == "YOLOv5 Nano (Small, Faster)":
            results = small_model(frame)
        else:
            results = large_model(frame)

        # Measure end time and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Store latency data for the current session
        st.session_state.latencies.append({
            'Model': model_choice,
            'Latency (ms)': latency
        })

        # Render results on the frame
        annotated_frame = results.render()[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        # Display real-time latency comparison
        latency_df = pd.DataFrame(st.session_state.latencies)
        if not latency_df.empty:
            fig = px.bar(
                latency_df,
                x=latency_df.index,
                y="Latency (ms)",
                color="Model",
                title="Real-time Latency Comparison",
                labels={'index': 'Frame Number'}
            )
            latency_chart.plotly_chart(fig, use_container_width=True)

    cap.release()

    # Display Average Latency for the Current Session
    if st.session_state.latencies:
        session_avg_latency = latency_df.groupby('Model')['Latency (ms)'].mean().reset_index()
        average_latency_display.subheader("**Average Latency for the Current Session**")
        average_latency_display.table(session_avg_latency)

# Display Cumulative Average Latency Across All Sessions
if st.session_state.cumulative_latencies:
    cumulative_df = pd.DataFrame(st.session_state.cumulative_latencies)
    cumulative_avg_latency = cumulative_df.groupby('Model')['Latency (ms)'].mean().reset_index()
    cumulative_latency_display.subheader("**Cumulative Average Latency Across All Sessions**")
    cumulative_latency_display.table(cumulative_avg_latency)

else:
    st.write("Click **Start Detection** to begin real-time object detection.")
