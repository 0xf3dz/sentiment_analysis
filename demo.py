import streamlit as st
import whisper
import numpy as np
import pandas as pd
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import queue
import threading
import time
from functions import predict_ensemble, predict_transformer, predict_claude

import sys
# Monkey-patch to prevent Streamlit from walking through torch.classes
if "torch" in sys.modules:
    import torch
    torch.classes.__path__ = []  # prevent Streamlit from trying to resolve this


st.set_page_config(layout="wide")

st.title("Sentiment and Emotion Classification")
st.markdown("Record audio or type text, and explore the classified sentiment and emotion using different custom-trained models. Navigate to the sidebar for settings and model management. Enjoy!")
st.markdown("**Note:** This app is a demo version for education purposes only. For more details, please refer to the Github repository for documentation and information on the collaborators.")

# Initialize session state variables 
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None
if 'text_input_key' not in st.session_state:
    st.session_state.text_input_key = 0
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ['model1']  # Default: Ensemble Model 

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

if st.session_state.model is None:
    with st.spinner("Loading Whisper base model..."):
        st.session_state.model = load_whisper_model()
        if st.session_state.model:
            st.success("Whisper base model loaded successfully!")

def get_predictions_from_multiple_models(text, selected_models):

    results = {}
    
    for model_name in selected_models:
        try:
            if model_name == 'model1':
                results[model_name] = predict_ensemble(text)  
            elif model_name == 'model2':
                results[model_name] = predict_transformer(text) 
            elif model_name == 'model3':
                results[model_name] = predict_claude(text) 
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")
            results[model_name] = {
                'sentiment': 'Error',
                'emotion': 'Error'
            }
    
    return results

def add_to_history(text, source_type):

    try:
        # Get predictions from selected models
        predictions = get_predictions_from_multiple_models(text, st.session_state.selected_models)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "text": text,
            "source": source_type,
            "predictions": predictions
        }
        
        st.session_state.history.append(entry)
        return True
    except Exception as e:
        st.error(f"Error processing text: {e}")
        return False

def process_audio(audio_bytes):

    if not st.session_state.model:
        st.warning("Whisper model not loaded.")
        return False
    
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
        fp.write(audio_bytes)
        temp_audio_path = fp.name
    
    try:
        with st.spinner("Transcribing audio..."):
            result = st.session_state.model.transcribe(temp_audio_path)
            transcription = result["text"]
            
            # Clean up the temp file
            os.unlink(temp_audio_path)
            
            # Add to history
            return add_to_history(transcription, "audio")
            
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        # Clean up the temp file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return False

st.subheader("Input Options")
st.write("Record your voice or type text to add entries to the history")

# Model Selection
st.subheader("Model Selection")
available_models = {
    'model1': 'Ensemble',
    'model2': 'Transformer',
    'model3': 'Claude'
}

selected_models = st.multiselect(
    "Select models for prediction:",
    options=list(available_models.keys()),
    default=st.session_state.selected_models,
    format_func=lambda x: available_models[x],
    help="You can select multiple models to compare predictions"
)

if selected_models != st.session_state.selected_models:
    st.session_state.selected_models = selected_models

if not selected_models:
    st.warning("Please select at least one model for prediction.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Record Audio")
    
    # Using audio_recorder_streamlit package for recording
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x"
    )
    
    # Check if we have new audio data
    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        st.audio(audio_bytes, format="audio/wav")
        
        if process_audio(audio_bytes):
            st.success("Audio processed and added to history!")
            st.rerun()

with col2:
    st.subheader("Type Text")
    
    # Text input with unique key
    text_input = st.text_area(
        "Type your text here:", 
        height=150,
        key=f"text_input_{st.session_state.text_input_key}"
    )
    
    # Submit button
    if st.button("Submit Text", type="primary"):
        if text_input.strip():
            if add_to_history(text_input.strip(), "text"):
                st.success("Text processed and added to history!")
                # Clear the text input by incrementing the key
                st.session_state.text_input_key += 1
                st.rerun()
        else:
            st.warning("Please enter some text before submitting.")

st.divider()

# Show latest entry
if st.session_state.history:
    st.subheader("Classification Results")
    latest = st.session_state.history[-1]
    
    # Display text
    st.markdown(f"**Text:** {latest['text']}")
    
    # Display predictions from all selected models
    if 'predictions' in latest and latest['predictions']:
        cols = st.columns(len(latest['predictions']))
        for i, (model_name, prediction) in enumerate(latest['predictions'].items()):
            with cols[i]:
                st.markdown(f"**{available_models[model_name]}**")
                # Color code sentiment
                sentiment = prediction['sentiment']
                sentiment_color = "#28a745" if sentiment.lower() in ['positive', 'pos'] else "#dc3545" if sentiment.lower() in ['negative', 'neg'] else "#ffc107"
                st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
                st.markdown(f"**Emotion:** {prediction['emotion']}")
    elif 'sentiment' in latest and 'emotion' in latest:
        sentiment_color = "#28a745" if latest['sentiment'].lower() in ['positive', 'pos'] else "#dc3545" if latest['sentiment'].lower() in ['negative', 'neg'] else "#ffc107"
        st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{latest['sentiment']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Emotion:** {latest['emotion']}")
    else:
        st.info("No predictions available. Please select at least one model and add new entries.")

st.divider()


st.subheader("History")

if st.session_state.history:

    def standardize_prediction_value(value):
        """Convert prediction values to consistent string format"""
        if isinstance(value, list):
            return ', '.join(map(str, value))  # Convert list to comma-separated string
        elif pd.isna(value) or value is None:
            return '-'  # Handle NaN/None values
        else:
            return str(value)  # Convert other types to string

    table_data = []
    for entry in st.session_state.history:
        row = {
            'Time': entry['timestamp'],
            'Format': "Audio" if entry['source'] == "audio" else "Text",
            'Text': entry['text']
        }
        
        if 'predictions' in entry:
            # New format with multiple models
            for model_name in available_models.keys():
                if model_name in entry['predictions']:
                    # Standardize the prediction values
                    sentiment = standardize_prediction_value(entry['predictions'][model_name]['sentiment'])
                    emotion = standardize_prediction_value(entry['predictions'][model_name]['emotion'])
                    
                    row[f'{available_models[model_name]} - Sentiment'] = sentiment
                    row[f'{available_models[model_name]} - Emotion'] = emotion
                else:
                    row[f'{available_models[model_name]} - Sentiment'] = '-'
                    row[f'{available_models[model_name]} - Emotion'] = '-'
        elif 'sentiment' in entry and 'emotion' in entry:
            # Old format - assign to first model for backward compatibility
            first_model = list(available_models.keys())[0]
            
            # Standardize the prediction values
            sentiment = standardize_prediction_value(entry['sentiment'])
            emotion = standardize_prediction_value(entry['emotion'])
            
            row[f'{available_models[first_model]} - Sentiment'] = sentiment
            row[f'{available_models[first_model]} - Emotion'] = emotion
            
            # Fill other models with empty values
            for model_name in list(available_models.keys())[1:]:
                row[f'{available_models[model_name]} - Sentiment'] = '-'
                row[f'{available_models[model_name]} - Emotion'] = '-'
        else:
            # No predictions available
            for model_name in available_models.keys():
                row[f'{available_models[model_name]} - Sentiment'] = '-'
                row[f'{available_models[model_name]} - Emotion'] = '-'
        
        table_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    # Additional safety check - ensure all columns are strings
    for col in df.columns:
        if col not in ['Time']:  # Skip datetime column
            df[col] = df[col].astype(str)
    
    # Configure column widths and properties
    column_config = {
        "Time": st.column_config.DatetimeColumn(
            "Time",
            width="medium",
            format="DD/MM/YY HH:mm:ss"
        ),
        "Format": st.column_config.TextColumn(  # Fixed: was "Source" but column is "Format"
            "Format",
            width="small"
        ),
        "Text": st.column_config.TextColumn(
            "Text",
            width="large",
            max_chars=150
        )
    }
    
    # Add column config for model predictions
    for model_name in available_models.keys():
        column_config[f'{available_models[model_name]} - Sentiment'] = st.column_config.TextColumn(
            f"{available_models[model_name][:8]}... - Sentiment",
            width="medium"
        )
        column_config[f'{available_models[model_name]} - Emotion'] = st.column_config.TextColumn(
            f"{available_models[model_name][:8]}... - Emotion", 
            width="medium"
        )
    
    # Display table 
    st.dataframe(
        df.iloc[::-1],  # Reverse to show most recent first
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=400  
    )
    
    # Statistics
    st.subheader("Statistics")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Total Entries", len(st.session_state.history))
    
    with col_stats2:
        audio_count = sum(1 for entry in st.session_state.history if entry['source'] == 'audio')
        st.metric("Audio Entries", audio_count)
    
    with col_stats3:
        text_count = sum(1 for entry in st.session_state.history if entry['source'] == 'text')
        st.metric("Text Entries", text_count)
    
    # Clear history button
    col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
    with col_clear2:
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_audio_bytes = None
            st.rerun()

else:
    st.info("No entries yet. Record audio or type text to see results here.")

# Export functionality
if st.session_state.history:
    st.subheader("Export Data")
    
    # Prepare export data
    export_data = []
    for entry in st.session_state.history:
        row = {
            'timestamp': entry['timestamp'],
            'source': entry['source'],
            'text': entry['text']
        }
        
        # Handle both new and old formats
        if 'predictions' in entry:
            # New format with multiple models
            for model_name in available_models.keys():
                if model_name in entry['predictions']:
                    row[f'{model_name}_sentiment'] = entry['predictions'][model_name]['sentiment']
                    row[f'{model_name}_emotion'] = entry['predictions'][model_name]['emotion']
                else:
                    row[f'{model_name}_sentiment'] = ''
                    row[f'{model_name}_emotion'] = ''
        elif 'sentiment' in entry and 'emotion' in entry:
            # Old format - assign to first model
            first_model = list(available_models.keys())[0]
            row[f'{first_model}_sentiment'] = entry['sentiment']
            row[f'{first_model}_emotion'] = entry['emotion']
            # Fill other models with empty values
            for model_name in list(available_models.keys())[1:]:
                row[f'{model_name}_sentiment'] = ''
                row[f'{model_name}_emotion'] = ''
        else:
            # No predictions
            for model_name in available_models.keys():
                row[f'{model_name}_sentiment'] = ''
                row[f'{model_name}_emotion'] = ''
        
        export_data.append(row)
    
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name=f"sentiment_analysis_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Sidebar for settings
with st.sidebar:
    # st.header("⚙️ Settings")
    
    # # Whisper model info (fixed to base)
    # st.info("🎙️ **Whisper Model:** Base (Fixed)")
    
    # st.divider()
    
    # Model management
    st.subheader("Prediction Models")
    st.markdown("**Available Models:**")
    for model_key, model_name in available_models.items():
        status = "✅" if model_key in st.session_state.selected_models else "⭕"
        st.markdown(f"{status} {model_name}")
    
    st.markdown(f"**Selected:** {len(st.session_state.selected_models)} model(s)")
    
    st.divider()
    
    # App info
    st.subheader("App Info")
    st.markdown("""
    **Instructions:**
    1. Select prediction Models 
                
        *Options:* Transformer, Ensemble (Sentiment + Emotion), Claude
                
    2. Record audio or type text
    3. View comparative results
    4. Check history table for all entries
    5. Export data if needed
    """)
    
    if st.session_state.history:
        st.metric("Session Entries", len(st.session_state.history))
