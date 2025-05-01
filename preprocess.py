#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IMDb Movie Review Sentiment Analysis - Data Preprocessing
This script handles loading and preprocessing of the IMDb movie review dataset.
"""

import os
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

def load_reviews(directory):
    """
    Load movie reviews from the specified directory.
    
    Args:
        directory (str): Path to directory containing 'pos' and 'neg' subdirectories
        
    Returns:
        tuple: (reviews, labels) lists
    """
    reviews = []
    labels = []
    for label in ['pos', 'neg']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
                reviews.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    
    return reviews, labels

def preprocess_text(text):
    """
    Perform basic text preprocessing operations.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def main(datadir, max_words=10000, max_review_length=600, validation_split=0.2, seed=42):
    """
    Main preprocessing function that loads, processes, and saves the IMDb dataset.
    
    Args:
        datadir (str): Path to the aclImdb directory
        max_words (int): Maximum vocabulary size
        max_review_length (int): Maximum sequence length for padding/truncating
        validation_split (float): Proportion of training data to use for validation
        seed (int): Random seed for reproducibility
    
    Returns:
        None: Saves preprocessed data to files
    """
    # Create output directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    print("Loading raw data...")
    # Load training and test data
    X_train_raw, y_train = load_reviews(os.path.join(datadir, 'train'))
    X_test_raw, y_test = load_reviews(os.path.join(datadir, 'test'))
    
    print("Preprocessing text...")
    # Apply preprocessing
    X_train_processed = [preprocess_text(review) for review in X_train_raw]
    X_test_processed = [preprocess_text(review) for review in X_test_raw]
    
    print("Tokenizing text...")
    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_processed)
    
    # Convert texts to sequences of token IDs
    X_train_sequences = tokenizer.texts_to_sequences(X_train_processed)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_processed)
    
    # Pad sequences to ensure uniform length
    X_train = sequence.pad_sequences(X_train_sequences, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test_sequences, maxlen=max_review_length)
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Split training data to create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=seed
    )
    
    print(f"Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/X_val.npy', X_val)
    np.save('processed_data/y_val.npy', y_val)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_test.npy', y_test)
    
    # Save tokenizer for later use
    with open('processed_data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save configuration
    config = {
        'max_words': max_words,
        'max_review_length': max_review_length,
        'vocab_size': min(max_words, len(tokenizer.word_index) + 1)
    }
    
    with open('processed_data/config.pickle', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess IMDb movie reviews dataset')
    parser.add_argument('--datadir', type=str, required=True, 
                        help='Path to the aclImdb directory')
    parser.add_argument('--max_words', type=int, default=10000,
                        help='Maximum vocabulary size')
    parser.add_argument('--max_review_length', type=int, default=600,
                        help='Maximum sequence length')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    
    args = parser.parse_args()
    
    main(args.datadir, args.max_words, args.max_review_length, args.validation_split)
