import streamlit as st
import pandas as pd
import plotly.express as px
from model_engine import SentryEngine
from streamlit_mic_recorder import mic_recorder
import io

st.set_page_config(page_title="Bio-Acoustic Sentry V2", layout="wide")

# Initialize Engine
engine = SentryEngine()

st.title("🛡️ Bio-Acoustic Sentry: Edge Training & Monitor")

# --- SIDEBAR: BACKEND TRAINING ---
with st.sidebar:
    st.header("Backend Controls")
    if st.button("🔄 Trigger Re-Training from Online Datasets"):
        with st.spinner("Scraping and Training..."):
            # Example paths - these would be your backend storage folders
            data_map = {
                "Harmful": "data/pests", 
                "Beneficial": "data/insects",
                "Human": "data/voices"
            }
            status = engine.train_on_datasets(data_map)
            st.success(status)

# --- MAIN UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Stream")
    
    # 1. File Uploader with specific constraints
    uploaded_file = st.file_uploader(
        "Upload Sound Signature", 
        type=["wav", "mp3", "mpeg"],
        help="Allowed formats: .wav, .mp3, .mpeg"
    )
    
    if uploaded_file is not None:
        st.info("Uploaded File Player")
        st.audio(uploaded_file) # Built-in Player
    
    st.divider()
    
    # 2. Microphone Input
    st.write("🎤 Device Microphone (Direct Capture)")
    audio_record = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop & Analyze",
        key='recorder'
    )

with col2:
    st.subheader("Classification Result")
    
    target_audio = None
    
    if uploaded_file:
        target_audio = uploaded_file
    elif audio_record:
        # Convert recorded bytes to a file-like object for librosa
        target_audio = io.BytesIO(audio_record['bytes'])
        st.audio(target_audio)

    if target_audio:
        with st.spinner("Analyzing against trained datasets..."):
            res = engine.analyze_audio(target_audio)
            
            st.metric("Detected Classification", res['name'])
            st.metric("Model Confidence", f"{res['confidence']:.2f}%")
            
            # Waveform Plot
            fig = px.line(res['waveform'][::50], title="Signal Signature", color_discrete_sequence=['#ff9800'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Awaiting input from microphone or file upload...")

