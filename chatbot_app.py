import streamlit as st
import ollama
import time
import pandas as pd
import plotly.express as px

# Streamlit app setup
st.title("NI Science Festival")
st.write("Compare response times for cloud and device deployments with interactive latency visualization.")

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

# Initialize session state for chat history and latency tracking
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latency_data' not in st.session_state:
    st.session_state.latency_data = []

# User input
user_input = st.text_input("You:", "")

# When user submits input
if st.button("Send") and user_input:
    with st.spinner('Generating responses...'):
        responses = []
        
        # Generate responses for each selected model
        for model_label in selected_models:
            model_name = model_options[model_label]
            
            # Measure response time
            start_time = time.time()
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': user_input}],
                options={'num_predict': max_tokens}  # Limit the number of tokens
            )
            end_time = time.time()
            
            latency = round(end_time - start_time, 2)
            reply = response['message']['content']
            
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
    
    # Create interactive bar plot using Plotly
    st.subheader("Interactive Latency Comparison")
    fig = px.bar(
        latency_df,
        x="Query",
        y="Latency (s)",
        color="Model",
        barmode="group",
        text_auto=True,
        title="Latency Comparison Between Models",
        labels={"Latency (s)": "Response Time (s)"},
        color_discrete_sequence=px.colors.qualitative.Set2  # Beautiful color palette
    )
    
    fig.update_layout(
        xaxis_title="User Query",
        yaxis_title="Latency (Seconds)",
        legend_title="Model",
        bargap=0.2,
        template='plotly_white',
        font=dict(size=14),
        title_x=0.5,
    )
    
    st.plotly_chart(fig, use_container_width=True)
