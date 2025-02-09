import streamlit as st
import ollama
import time
import pandas as pd
import plotly.express as px

# Import jetson-stats for energy monitoring if running on Jetson device
try:
    from jtop import jtop
except ImportError:
    jtop = None

# Streamlit app setup
st.title("NI Science Festival")
st.write("Compare response times and energy consumption for cloud and device deployments with interactive visualizations.")

# Check if running on Jetson device for energy measurement
is_device = st.checkbox("Run Energy Measurement (Jetson Device Only)", value=False)

# Model selection from dropdown
model_options = {
    'Device Optimized Model (tinyllama)': 'tinyllama',
    'Cloud Optimized Model (llama2)': 'llama2'
}
selected_models = st.multiselect(
    "Select one or more models to compare:",
    list(model_options.keys()),
    default=list(model_options.keys())
)

# Token limit input
max_tokens = st.number_input("Set maximum tokens for responses:", min_value=10, max_value=500, value=100, step=10)

# Initialize session state for chat history, latency, and energy tracking
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latency_data' not in st.session_state:
    st.session_state.latency_data = []
if 'energy_log' not in st.session_state:
    st.session_state.energy_log = []
if 'cumulative_latency_data' not in st.session_state:
    st.session_state.cumulative_latency_data = []
if 'cumulative_energy_log' not in st.session_state:
    st.session_state.cumulative_energy_log = []

# User input
user_input = st.text_input("You:", "")

# Buttons for start/stop
if st.button("Start Chat"):
    if is_device and jtop is not None:
        st.session_state.jtop_monitor = jtop()
        st.session_state.jtop_monitor.start()

if st.button("Stop Chat"):
    if is_device and jtop is not None and st.session_state.jtop_monitor:
        st.session_state.jtop_monitor.close()

    if st.session_state.latency_data:
        st.session_state.cumulative_latency_data.extend(st.session_state.latency_data)

    if st.session_state.energy_log:
        st.session_state.cumulative_energy_log.extend(st.session_state.energy_log)

# When user submits input
if st.button("Send") and user_input:
    with st.spinner('Generating responses...'):
        responses = []
        
        # Generate responses for each selected model
        for model_label in selected_models:
            model_name = model_options[model_label]

            # Start energy monitoring if enabled
            if is_device and jtop is not None and st.session_state.jtop_monitor:
                stats_before = st.session_state.jtop_monitor.stats

            # Measure response time
            start_time = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': user_input}],
                options={'num_predict': max_tokens}
            )
            end_time = time.time()

            latency = round(end_time - start_time, 2)
            reply = response['message']['content']

            # Collect energy data after response
            if is_device and jtop is not None and st.session_state.jtop_monitor:
                stats_after = st.session_state.jtop_monitor.stats
                power = stats_after.get('Power TOT', 0)
                cpu_values = [stats_after.get(f'CPU{i}', 0) for i in range(1, 7)]
                avg_cpu_usage = sum(cpu_values) / len(cpu_values) if cpu_values else 0
                gpu_usage = stats_after.get('GPU', 0)

                st.session_state.energy_log.append({
                    'Model': model_label,
                    'Power (mW)': power,
                    'CPU Usage (%)': avg_cpu_usage,
                    'GPU Usage (%)': gpu_usage
                })

            # Store response and latency
            responses.append((model_label, reply, latency))
            st.session_state.latency_data.append({'Model': model_label, 'Latency (s)': latency, 'Query': user_input})

        # Save the query and responses to session history
        st.session_state.chat_history.append((user_input, responses))

# Display chat history with responses and latency
for user_text, model_responses in st.session_state.chat_history:
    st.write(f"**You:** {user_text}")
    for model_label, reply, latency in model_responses:
        st.write(f"**{model_label} ({latency}s):** {reply}")
    st.write("---")

# Display Interactive Latency Comparison Bar Plot
if st.session_state.latency_data:
    latency_df = pd.DataFrame(st.session_state.latency_data)
    
    st.subheader("Interactive Latency Comparison")
    fig_latency = px.bar(
        latency_df,
        x="Query",
        y="Latency (s)",
        color="Model",
        barmode="group",
        title="Latency Comparison Between Models",
        labels={"Latency (s)": "Response Time (s)"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_latency, use_container_width=True)

# Display real-time energy data if on Jetson
if is_device and st.session_state.energy_log:
    energy_df = pd.DataFrame(st.session_state.energy_log)

    st.subheader("Real-time Energy Metrics")
    fig_power = px.line(energy_df, x=energy_df.index, y='Power (mW)', color='Model', title="Power Consumption Over Time")
    st.plotly_chart(fig_power, use_container_width=True)

    fig_cpu = px.line(energy_df, x=energy_df.index, y='CPU Usage (%)', color='Model', title="CPU Usage Over Time")
    st.plotly_chart(fig_cpu, use_container_width=True)

    fig_gpu = px.line(energy_df, x=energy_df.index, y='GPU Usage (%)', color='Model', title="GPU Usage Over Time")
    st.plotly_chart(fig_gpu, use_container_width=True)

# Display cumulative latency comparison
if st.session_state.cumulative_latency_data:
    cumulative_latency_df = pd.DataFrame(st.session_state.cumulative_latency_data)
    cumulative_avg_latency = cumulative_latency_df.groupby('Model')['Latency (s)'].mean().reset_index()

    st.subheader("**Cumulative Average Latency Across All Sessions**")
    st.table(cumulative_avg_latency)

    fig_cumulative_latency = px.bar(
        cumulative_avg_latency.sort_values(by='Latency (s)'),
        x='Model',
        y='Latency (s)',
        title="Final Average Latency Comparison",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_cumulative_latency, use_container_width=True)

# Display cumulative energy comparison
if is_device and st.session_state.cumulative_energy_log:
    cumulative_energy_df = pd.DataFrame(st.session_state.cumulative_energy_log)
    cumulative_avg_energy = cumulative_energy_df.groupby('Model').mean().reset_index()

    st.subheader("**Cumulative Average Energy Metrics Across All Sessions**")
    st.table(cumulative_avg_energy[['Model', 'Power (mW)', 'CPU Usage (%)', 'GPU Usage (%)']])

    # Final comparison plots for energy metrics
    fig_final_power = px.bar(
        cumulative_avg_energy.sort_values(by='Power (mW)'),
        x='Model',
        y='Power (mW)',
        title="Final Average Power Consumption",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_final_power, use_container_width=True)

    fig_final_cpu = px.bar(
        cumulative_avg_energy.sort_values(by='CPU Usage (%)'),
        x='Model',
        y='CPU Usage (%)',
        title="Final Average CPU Usage",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_final_cpu, use_container_width=True)

    fig_final_gpu = px.bar(
        cumulative_avg_energy.sort_values(by='GPU Usage (%)'),
        x='Model',
        y='GPU Usage (%)',
        title="Final Average GPU Usage",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_final_gpu, use_container_width=True)

else:
    st.write("Click **Start Chat** to begin comparing model responses.")
