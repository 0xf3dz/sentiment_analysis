#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IMDb Movie Review Sentiment Analysis - Model Implementation
This script defines different model architectures for sentiment classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, 
    BatchNormalization, Bidirectional, GlobalMaxPooling1D, SpatialDropout1D,
    Input, concatenate, GRU
)
from tensorflow.keras.regularizers import l2

embedding_vector_length = 100
top_words = 10000              
max_review_length = 600 

def create_cnn_lstm_model(config):
    """
    Create a CNN-LSTM model for sentiment analysis.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        keras.Model: Compiled Keras model
    """
        
    
    model = Sequential([
        # Embedding layer
    Embedding(input_dim=top_words, output_dim=embedding_vector_length),
    
    # Single convolutional layer with moderate number of filters
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    # Global pooling to reduce dimensionality
    MaxPooling1D(pool_size=2),
    LSTM(100)),
    # Output layer
    Dense(1, activation='sigmoid')

    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(config):
    """
    Create a basic LSTM model for sentiment analysis.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        keras.Model: Compiled Keras model
    """
    embedding_vector_length = 32
    
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=top_words, output_dim=embedding_vector_length),
        # LSTM layer
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        # Output layer
        Dense(1, activation='sigmoid')
        ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Dictionary of available models
MODEL_REGISTRY = {
    'cnn_lstm': create_cnn_lstm_model,
    'lstm': create_lstm_model
}
