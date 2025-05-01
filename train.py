#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IMDb Movie Review Sentiment Analysis - Model Training
This script handles the training of sentiment analysis models.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
from models import MODEL_REGISTRY

def load_data():
    """
    Load preprocessed dataset.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, config)
    """
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_val = np.load('processed_data/X_val.npy')
    y_val = np.load('processed_data/y_val.npy')
    
    with open('processed_data/config.pickle', 'rb') as handle:
        config = pickle.load(handle)
    
    return X_train, y_train, X_val, y_val, config

def create_callbacks(model_name):
    """
    Create training callbacks.
    
    Args:
        model_name (str): Name of the model for saving checkpoints
        
    Returns:
        list: List of Keras callbacks
    """
    # Create directories for model checkpoints and logs
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    filepath = f"models/{model_name}_best.h5"
    
    callbacks = [
        # Model checkpoint to save best weights
        ModelCheckpoint(
            filepath=filepath,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when performance plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logs for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}_{time.strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callbacks

def plot_training_history(history, model_name):
    """
    Plot training and validation metrics.
    
    Args:
        history (keras.callbacks.History): Training history object
        model_name (str): Name of the model for plot titles and saving
    """
    os.makedirs('plots', exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png')
    plt.close()

def save_training_results(history, model_name):
    """
    Save training history to file.
    
    Args:
        history (keras.callbacks.History): Training history object
        model_name (str): Name of the model
    """
    os.makedirs('results', exist_ok=True)
    
    with open(f'results/{model_name}_history.pickle', 'wb') as file:
        pickle.dump(history.history, file)

def train_model(model_name, batch_size=64, epochs=15):
    """
    Train a model with the specified architecture.
    
    Args:
        model_name (str): Name of the model architecture to use
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        
    Returns:
        tuple: (trained_model, history)
    """
    print(f"Loading data...")
    X_train, y_train, X_val, y_val, config = load_data()
    
    print(f"Creating {model_name} model...")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(config)
    
    print(model.summary())
    
    print(f"Training {model_name} model...")
    callbacks = create_callbacks(model_name)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot and save training history
    plot_training_history(history, model_name)
    save_training_results(history, model_name)
    
    # Save the final model
    model.save(f'models/{model_name}_final.h5')
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train IMDb sentiment analysis model')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Maximum number of epochs')
    
    args = parser.parse_args()
    
    train_model(args.model, args.batch_size, args.epochs)
