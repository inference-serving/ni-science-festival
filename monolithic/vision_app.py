import cv2
import torch
import time
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px

# Import jetson-stats for energy monitoring if running on Jetson device
try:
    from jtop import jtop
except ImportError:
    jtop = None

# Streamlit App Setup
st.title("Real-time Object Detection on Jetson Nano")
st.write("Compare **YOLOv5 Nano** (Small), **YOLOv5 Large** (Medium), and **YOLOv5 XLarge** (Large) models with live latency and energy metrics.")

# Check if running on Jetson device for energy measurement
is_device = st.checkbox("Run Energy Measurement (Jetson Device Only)", value=False)

# Load YOLOv5 Models (Small, Medium, Large)
@st.cache_resource
def load_models():
    small_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # Nano (Small)
    medium_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)  # Large (Medium)
    large_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)  # XLarge (Large)
    return small_model, medium_model, large_model

small_model, medium_model, large_model = load_models()

# Model Selection Dropdown
model_choice = st.selectbox(
    "Select the Model for Detection:",
    ("YOLOv5 Nano (Small, Fastest)", "YOLOv5 Large (Medium, Balanced)", "YOLOv5 XLarge (Large, Most Accurate)")
)

# Initialize session state variables
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if 'latencies' not in st.session_state:
    st.session_state.latencies = []

if 'cumulative_latencies' not in st.session_state:
    st.session_state.cumulative_latencies = []

if 'energy_log' not in st.session_state:
    st.session_state.energy_log = []

if 'cumulative_energy_log' not in st.session_state:
    st.session_state.cumulative_energy_log = []

if 'cap' not in st.session_state:
    st.session_state.cap = None

# Start/Stop Buttons
if st.button("Start Detection"):
    st.session_state.streaming = True
    st.session_state.latencies = []
    st.session_state.energy_log = []

    st.session_state.cap = cv2.VideoCapture(0)
    if not st.session_state.cap.isOpened():
        st.error("Failed to open camera. Please check your camera connection.")
        st.session_state.streaming = False

    if is_device and jtop is not None:
        st.session_state.jtop_monitor = jtop()
        st.session_state.jtop_monitor.start()

if st.button("Stop Detection"):
    st.session_state.streaming = False

    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    if is_device and jtop is not None and st.session_state.jtop_monitor:
        st.session_state.jtop_monitor.close()

    if st.session_state.latencies:
        st.session_state.cumulative_latencies.extend(st.session_state.latencies)

    if st.session_state.energy_log:
        st.session_state.cumulative_energy_log.extend(st.session_state.energy_log)

# Placeholders for displaying results
stframe = st.empty()
latency_chart = st.empty()
power_chart = st.empty()
cpu_chart = st.empty()
gpu_chart = st.empty()
average_latency_display = st.empty()
average_energy_display = st.empty()
cumulative_latency_display = st.empty()
cumulative_energy_display = st.empty()

# Final comparison placeholders
final_latency_display = st.empty()
final_power_display = st.empty()
final_cpu_display = st.empty()
final_gpu_display = st.empty()

# Real-time object detection loop
if st.session_state.streaming and st.session_state.cap is not None:
    while st.session_state.streaming:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        # Measure start time
        start_time = time.time()

        # Perform inference based on selected model
        if model_choice == "YOLOv5 Nano (Small, Fastest)":
            results = small_model(frame)
        elif model_choice == "YOLOv5 Large (Medium, Balanced)":
            results = medium_model(frame)
        else:
            results = large_model(frame)

        # Measure end time and calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        st.session_state.latencies.append({'Model': model_choice, 'Latency (ms)': latency})

        # Collect energy data if on Jetson device
        if is_device and jtop is not None and st.session_state.jtop_monitor:
            stats = st.session_state.jtop_monitor.stats
            power = stats.get('Power TOT', 0)
            cpu_values = [stats.get(f'CPU{i}', 0) for i in range(1, 7)]
            avg_cpu_usage = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            gpu_usage = stats.get('GPU', 0)

            st.session_state.energy_log.append({
                'Model': model_choice,
                'Frame': len(st.session_state.latencies),
                'Power (mW)': power,
                'CPU Usage (%)': avg_cpu_usage,
                'GPU Usage (%)': gpu_usage
            })

        # Render results on frame
        annotated_frame = results.render()[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        # Display real-time latency
        latency_df = pd.DataFrame(st.session_state.latencies)
        if not latency_df.empty:
            fig_latency = px.bar(latency_df, x=latency_df.index, y="Latency (ms)", color="Model",
                                 title="Real-time Latency per Frame", labels={'index': 'Frame Number'})
            latency_chart.plotly_chart(fig_latency, use_container_width=True)

        # Display real-time energy data if on Jetson
        if is_device and st.session_state.energy_log:
            energy_df = pd.DataFrame(st.session_state.energy_log)

            fig_power = px.line(energy_df, x='Frame', y='Power (mW)', color='Model', title="Power Consumption Over Time")
            power_chart.plotly_chart(fig_power, use_container_width=True)

            fig_cpu = px.line(energy_df, x='Frame', y='CPU Usage (%)', color='Model', title="CPU Usage Over Time")
            cpu_chart.plotly_chart(fig_cpu, use_container_width=True)

            fig_gpu = px.line(energy_df, x='Frame', y='GPU Usage (%)', color='Model', title="GPU Usage Over Time")
            gpu_chart.plotly_chart(fig_gpu, use_container_width=True)

    # Release the camera after stopping
    st.session_state.cap.release()

    # Display average latency per session
    if st.session_state.latencies:
        session_avg_latency = latency_df.groupby('Model')['Latency (ms)'].mean().reset_index()
        average_latency_display.subheader("**Average Latency for the Current Session**")
        average_latency_display.table(session_avg_latency)

    # Display average energy metrics per session
    if is_device and st.session_state.energy_log:
        energy_df = pd.DataFrame(st.session_state.energy_log)
        session_avg_energy = energy_df.groupby('Model').mean().reset_index()
        average_energy_display.subheader("**Average Energy Metrics for the Current Session**")
        average_energy_display.table(session_avg_energy[['Model', 'Power (mW)', 'CPU Usage (%)', 'GPU Usage (%)']])

# Display cumulative averages
if st.session_state.cumulative_latencies:
    cumulative_df = pd.DataFrame(st.session_state.cumulative_latencies)
    cumulative_avg_latency = cumulative_df.groupby('Model')['Latency (ms)'].mean().reset_index()
    cumulative_latency_display.subheader("**Cumulative Average Latency Across All Sessions**")
    cumulative_latency_display.table(cumulative_avg_latency)

if is_device and st.session_state.cumulative_energy_log:
    cumulative_energy_df = pd.DataFrame(st.session_state.cumulative_energy_log)
    cumulative_avg_energy = cumulative_energy_df.groupby('Model').mean().reset_index()
    cumulative_energy_display.subheader("**Cumulative Average Energy Metrics Across All Sessions**")
    cumulative_energy_display.table(cumulative_avg_energy[['Model', 'Power (mW)', 'CPU Usage (%)', 'GPU Usage (%)']])

    # Display final comparison plots
    st.subheader("**Final Comparison Plots Across All Sessions**")

    # Latency Bar Plot
    fig_final_latency = px.bar(cumulative_avg_latency.sort_values(by='Latency (ms)'), x='Model', y='Latency (ms)', 
                               title="Final Average Latency Comparison")
    final_latency_display.plotly_chart(fig_final_latency, use_container_width=True)

    # Power Consumption Bar Plot
    fig_final_power = px.bar(cumulative_avg_energy.sort_values(by='Power (mW)'), x='Model', y='Power (mW)', 
                             title="Final Average Power Consumption Comparison")
    final_power_display.plotly_chart(fig_final_power, use_container_width=True)

    # CPU Usage Bar Plot
    fig_final_cpu = px.bar(cumulative_avg_energy.sort_values(by='CPU Usage (%)'), x='Model', y='CPU Usage (%)', 
                           title="Final Average CPU Usage Comparison")
    final_cpu_display.plotly_chart(fig_final_cpu, use_container_width=True)

    # GPU Usage Bar Plot
    fig_final_gpu = px.bar(cumulative_avg_energy.sort_values(by='GPU Usage (%)'), x='Model', y='GPU Usage (%)', 
                           title="Final Average GPU Usage Comparison")
    final_gpu_display.plotly_chart(fig_final_gpu, use_container_width=True)

else:
    st.write("Click **Start Detection** to begin real-time object detection.")
