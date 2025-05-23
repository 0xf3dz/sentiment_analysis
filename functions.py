### File with necessary functions for the demo

import pandas as pd
import numpy as np
import random
import os
import re
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import asyncio


def predict_ensemble (text):

    sentiment_model = load_model('models/sentiment_model_finetuned.keras', compile=False)
    sentiment_model.name = 'sentiment_model'
    sentiment_model.trainable = False
    emotion_model = load_model('models/emotion_model_finetuned.keras', compile=False)
    emotion_model.name = 'emotion_model'
    emotion_model.trainable = False

    # Define new input layers
    sentiment_input = Input(shape=sentiment_model.input_shape[1:], name="sentiment_input")
    emotion_input = Input(shape=emotion_model.input_shape[1:], name="emotion_input")

    # Pass the inputs through the respective models
    sentiment_output = sentiment_model(sentiment_input)
    emotion_output = emotion_model(emotion_input)

    # Create the joint model
    joint_model = Model(
        inputs=[sentiment_input, emotion_input],
        outputs=[sentiment_output, emotion_output],
        name='ensemble_model'
    )

    def data_preprocessing(text):

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Load the tokeniizer
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer = tokenizer_from_json(f.read())

        sequences = tokenizer.texts_to_sequences([text])
        encoded_text = pad_sequences(sequences, maxlen=500, padding='post')
        
        return encoded_text

    input = data_preprocessing(text)

    # Use the correct input layer names
    predictions = joint_model.predict({
        'sentiment_input': input,  
        'emotion_input': input    
    })

    if predictions[0][0] >= 0.5:
        pred_sent = 'Positive'
    else:
        pred_sent = 'Negative'

    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    with open('optimal_thresholds.json') as f:
        optimal_thresholds = json.load(f)

    pred_emo = np.zeros_like(predictions[1][0], dtype=int)
    for i, emo in enumerate(emotion_columns):
        pred_emo[i] = (predictions[1][0][i] >= optimal_thresholds[emo]).astype(int)
    if np.sum(pred_emo) == 0:
        max_indices = np.argmax(predictions[1][0])
        pred_emo[max_indices] = 1
    
    pred_emotion = []
    i = 0
    for value in pred_emo:
        if value == 1:
            pred_emotion.append(emotion_columns[i])
        i += 1

    return {
        'sentiment': pred_sent,
        'emotion': pred_emotion
    }

def predict_transformer (text):
    transformer_model = load_model('models/transformer_model_finetuned.keras', compile=False)
    transformer_model.trainable = False

    def data_preprocessing(text):

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Load the tokeniizer
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer = tokenizer_from_json(f.read())

        sequences = tokenizer.texts_to_sequences([text])
        encoded_text = pad_sequences(sequences, maxlen=500, padding='post')
        
        return encoded_text

    input = data_preprocessing(text)
    predictions = transformer_model.predict(input)
    pred_sent_t = np.argmax(predictions[0])
    if pred_sent_t == 0:
        pred_sent = 'Positive'
    else:
        pred_sent = 'Negative'

    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    with open('optimal_thresholds_transformer.json') as f:
        optimal_thresholds = json.load(f)

    # Initialize predictions array with correct shape
    pred_emo = np.zeros_like(predictions[1][0], dtype=int)  # Take first sample since predictions[1] has shape (1,28)

    # Apply thresholds to predictions
    for i, emo in enumerate(emotion_columns):
        pred_emo[i] = (predictions[1][0][i] >= optimal_thresholds[emo]).astype(int)

    # Handle case where no emotion is predicted
    if np.sum(pred_emo) == 0:
        max_indices = np.argmax(predictions[1][0])  # Get index of highest probability
        pred_emo[max_indices] = 1

    # Convert binary predictions to emotion labels
    pred_emotion = [emotion_columns[i] for i, value in enumerate(pred_emo) if value == 1]

    return {
        'sentiment': pred_sent,
        'emotion': pred_emotion
    }

def predict_claude(text, api_key):
    ANTHROPIC_API_KEY = api_key
    claude = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model="claude-3-5-sonnet-20241022")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analysis expert. Be concise and direct in your responses so your output is just the observed sentiment and primary emotion."),
        ("user", "Analyze the sentiment of the following text and classify it as positive or negative. Also identify the primary emotion expressed between the following emotions: 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring','confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'. Text: {text}")
    ])
    
    chain = LLMChain(llm=claude, prompt=chat_prompt)
    
    # Fixed: chain.run() instead of chain.arun() for synchronous execution
    response = chain.run(text=text)
    
    # Fixed: removed asyncio.gather which was incorrectly used
    # Fixed: response variable was already defined above
    
    response = response.lower()
    
    pred_sent = 'Positive' if 'positive' in response else 'Negative'

    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # Extract emotion by finding matches with emotion list
    pred_emotion = 'neutral'  # default
    for emotion in emotion_columns:
        if emotion.lower() in response:
            pred_emotion = emotion
            break
            
    return {
        'sentiment': pred_sent,
        'emotion': pred_emotion
    }
