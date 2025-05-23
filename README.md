# Sentiment Analysis


## Note for the professors: 
The Jupyter notebook in the repository contains detailed, commented code for training and testing the custom ML models. These models were already trained and saved, so their performance can be directly analysed.
We encourage reviewing it to understand the model development process. 
Also, the deployed Streamlit app enables direct assessment of the custom-trained models’ performance through interactive inputs and comparative outputs against the benchmark LLM.
Please follow the detailed instructions below.


## Overview
A comprehensive sentiment and emotion analysis application that processes both audio and text inputs to classify sentiment (positive/negative) and emotions across 28 different categories using multiple AI models.

## Key Features
- **Dual Input Methods**: 
  - Audio recording with Whisper transcription
  - Direct text input via textbox
  
- **Multi-Model Analysis**:
  - **Ensemble Model**: Custom-trained Keras model combining sentiment and emotion classification
  - **Transformer Model**: Fine-tuned transformer architecture for sentiment and emotion prediction
  - **Claude Integration**: Uses Claude 3.5 Sonnet for LLM-based sentiment and emotion analysis
  
- **Comparative Analysis**: Side-by-side comparison of predictions from different models

- **Interactive Dashboard**:
  - Real-time processing and visualization
  - Historical tracking of all analyses
  - Export functionality for data collection

## Emotion Classification
Detects 28 distinct emotions:
`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`

## Installation

```bash
# Clone the repository
git clone https://github.com/0xf3dz/sentiment_analysis.git
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt
```

## Add API KEY
In `demo.py` set api_key to your own own Anthropic API key.

## Usage

```bash
# Run the Streamlit application
streamlit run demo.py
```

## Project Structure
- `demo.py`: Main Streamlit application
- `functions.py`: Core prediction functions for all models
- `models/`: Directory containing pre-trained models
- `audio_files/`: Sample audio files for testing
- `tokenizer.json`: Text tokenizer for model input processing
- `optimal_thresholds.json` & `optimal_thresholds_transformer.json`: Calibrated thresholds for emotion classification

## Requirements
- Python 3.8+
- Streamlit
- TensorFlow
- Whisper (OpenAI)
- LangChain with Anthropic integration
- See `requirements.txt` for complete dependencies

## Note
To use the Claude model integration, you need to add your Anthropic API key to the `functions.py` file.
