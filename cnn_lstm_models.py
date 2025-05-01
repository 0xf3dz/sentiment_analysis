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
    Input, concatenate
)
from tensorflow.keras.regularizers import l2

embedding_vector_length = 100
top_words = 10000              
max_review_length = 600 

def create_cnn_model(config):
    """
    Create a basic CNN model for sentiment analysis.
    
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
    BatchNormalization(),
    # Global pooling to reduce dimensionality
    GlobalMaxPooling1D(),
    
    # Strong dropout for regularization
    Dropout(0.5),
    
    # Single hidden layer
    Dense(32, activation='relu'),
    
    # More dropout
    Dropout(0.5),
    
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
    Create an LSTM model for sentiment analysis.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        keras.Model: Compiled Keras model
    """
    embedding_vector_length = 100  # Increased from 32
    
    model = Sequential([
      # Embedding layer
      Embedding(input_dim=top_words, output_dim=embedding_vector_length),
    
    # Apply spatial dropout to prevent overfitting on the embedding layer
      SpatialDropout1D(0.3),
    
    # LSTM layer
      Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)),
    
      BatchNormalization(),
    
      # Dense layer with regularization
      Dense(32, activation='relu'),
      Dropout(0.5),
    
    # Output layer
      Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Dictionary of available models
MODEL_REGISTRY = {
    'cnn': create_cnn_model,
    'lstm': create_lstm_model
}
