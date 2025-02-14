import streamlit as st
import ollama
import time
import requests
import pandas as pd
import plotly.express as px

# Cloud server URL
CLOUD_SERVER_URL = "http://192.5.86.154:8000/infer"

st.title("NI Science Festival")
st.write("Compare response times and energy consumption for cloud and device deployments, including network latency.")

# Button to clear chat history and latency data
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.latency_data = []
    st.rerun()

# Simplified model selection options
model_options = {
    'small (edge)': ('tinyllama', 'local'),
    # 'large (edge)': ('llama2', 'local'),
    'small (cloud)': ('tinyllama', 'cloud'),
    'large (cloud)': ('llama2', 'cloud')
}

selected_models = st.multiselect(
    "Select one or more models to compare:",
    list(model_options.keys()),
    default=list(model_options.keys())
)

max_tokens = st.number_input("Set max tokens:", min_value=10, max_value=500, value=100, step=10)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latency_data' not in st.session_state:
    st.session_state.latency_data = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    with st.spinner('Generating responses...'):
        responses = []

        for model_label in selected_models:
            model_name, execution_location = model_options[model_label]

            if execution_location == 'local':
                # Run model locally on the device
                start_time = time.time()
                response = ollama.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': user_input}],
                    options={'num_predict': max_tokens}
                )
                end_time = time.time()
                reply = response['message']['content']
                total_latency = round(end_time - start_time, 2)
                
                # For local models, network latency is zero
                network_latency = 0
                model_latency = total_latency

            else:
                # Measure total round-trip time for cloud request
                round_trip_start = time.time()
                response = requests.post(
                    CLOUD_SERVER_URL,
                    json={"model_name": model_name, "user_input": user_input, "max_tokens": max_tokens}
                ).json()
                round_trip_end = time.time()
                
                reply = response['response']
                model_latency = response['latency']
                total_round_trip_latency = round(round_trip_end - round_trip_start, 2)

                # Calculate network latency
                network_latency = total_round_trip_latency - model_latency

            # Store responses and latency data
            responses.append((model_label, reply, model_latency, network_latency))
            st.session_state.latency_data.append({
                "Model": model_label,
                "Model Latency (s)": model_latency,
                "Network Latency (s)": network_latency,
                "Total Latency (s)": model_latency + network_latency,
                "Query": user_input
            })

        st.session_state.chat_history.append((user_input, responses))

# Display chat history
for user_text, model_responses in st.session_state.chat_history:
    st.write(f"**You:** {user_text}")
    for model_label, reply, model_latency, network_latency in model_responses:
        if network_latency > 0:
            st.write(f"**{model_label} (Model: {model_latency}s, Network: {network_latency}s):** {reply}")
        else:
            st.write(f"**{model_label} ({model_latency}s):** {reply}")
    st.write("---")

# Display latency comparison chart with network latency
if st.session_state.latency_data:
    latency_df = pd.DataFrame(st.session_state.latency_data)

    st.subheader("Latency Comparison (Model vs. Network)")

    # Melt the DataFrame to plot model and network latency as stacked bars
    melted_df = latency_df.melt(
        id_vars=["Model", "Query"],
        value_vars=["Model Latency (s)", "Network Latency (s)"],
        var_name="Latency Type",
        value_name="Latency (s)"
    )

    fig_latency = px.bar(
        melted_df,
        x="Query",
        y="Latency (s)",
        color="Latency Type",
        barmode="stack",
        facet_col="Model",
        title="Stacked Latency: Model vs. Network",
        labels={"Latency (s)": "Time (s)"},
        height=600
    )

    # Rotate x-axis labels to prevent overlap
    fig_latency.update_xaxes(tickangle=-45)

    st.plotly_chart(fig_latency, use_container_width=True)
