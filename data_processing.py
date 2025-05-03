#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Dataset Preprocessing Script
This script handles loading and preprocessing of multiple text datasets:
1. IMDb movie review dataset (binary sentiment classification)
2. GoEmotions dataset (multi-label emotion classification)
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_imdb_reviews(directory):
    """
    Load movie reviews from the specified IMDb directory.
    
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

def load_goemotions(directory):
    """
    Load GoEmotions dataset from the specified directory.
    
    Args:
        directory (str): Path to directory containing GoEmotions CSV files
        
    Returns:
        pandas.DataFrame: Combined GoEmotions dataset
    """
    # Define CSV filenames
    emo1_df = 'goemotions_1.csv'
    emo2_df = 'goemotions_2.csv'
    emo3_df = 'goemotions_3.csv'
    
    # Load individual dataframes
    emo1 = pd.read_csv(os.path.join(directory, emo1_df), sep=',')
    emo2 = pd.read_csv(os.path.join(directory, emo2_df), sep=',')
    emo3 = pd.read_csv(os.path.join(directory, emo3_df), sep=',')
    
    # Concatenate all three dataframes vertically
    emotions_df = pd.concat([emo1, emo2, emo3], axis=0, ignore_index=True)
    
    return emotions_df

def preprocess_text(text):
    """
    Perform basic text preprocessing operations.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = str(text).lower()
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_class_weights(y_train, emotion_columns=None):
    """
    Calculate class weights for multi-label classification.
    
    Args:
        y_train (numpy.ndarray): Training labels
        emotion_columns (list): List of emotion category names
        
    Returns:
        dict: Dictionary mapping class indices to weights
    """
    # Sum each column to get positive counts per class
    positive_counts = np.sum(y_train, axis=0)
    total_samples = len(y_train)
    
    # Base weights - inverse frequency
    base_weights = total_samples / (positive_counts + 1e-5)  # Adding small epsilon to avoid division by zero
    
    # Make weights more aggressive by applying a scaling factor
    scaling_factor = 2  # Adjust this value to make weights more/less aggressive
    aggressive_weights = base_weights ** scaling_factor
    
    # Normalize weights to have a reasonable scale
    normalized_weights = aggressive_weights / np.mean(aggressive_weights)
    
    # Create a dictionary for class weights
    class_weight_dict = {i: float(weight) for i, weight in enumerate(normalized_weights)}
    
    # Cap extremely large weights to prevent instability
    max_weight = 10.0  # Set a maximum weight value
    for i in class_weight_dict:
        if class_weight_dict[i] > max_weight:
            class_weight_dict[i] = max_weight
    
    # Print the weights if emotion columns are provided
    if emotion_columns is not None:
        print("Class weights:")
        for i, weight in class_weight_dict.items():
            print(f"{emotion_columns[i]}: {weight:.4f}")
    
    return class_weight_dict

def process_imdb(datadir, max_words=10000, max_review_length=600, validation_split=0.2, seed=42):
    """
    Process IMDb movie review dataset.
    
    Args:
        datadir (str): Path to the aclImdb directory
        max_words (int): Maximum vocabulary size
        max_review_length (int): Maximum sequence length for padding/truncating
        validation_split (float): Proportion of training data to use for validation
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing processed data and configuration
    """
    print("\n--- Processing IMDb Dataset ---")
    
    print("Loading raw data...")
    # Load training and test data
    X_train_raw, y_train = load_imdb_reviews(os.path.join(datadir, 'train'))
    X_test_raw, y_test = load_imdb_reviews(os.path.join(datadir, 'test'))
    
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
    
    # Create configuration
    config = {
        'max_words': max_words,
        'max_review_length': max_review_length,
        'vocab_size': min(max_words, len(tokenizer.word_index) + 1)
    }
    
    processed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'config': config
    }
    
    return processed_data

def process_goemotions(datadir, max_words=5000, max_text_length=100, validation_split=0.2, test_split=0.2, seed=42):
    """
    Process GoEmotions dataset.
    
    Args:
        datadir (str): Path to the GoEmotions directory
        max_words (int): Maximum vocabulary size
        max_text_length (int): Maximum sequence length for padding/truncating
        validation_split (float): Proportion of training data to use for validation
        test_split (float): Proportion of data to use for testing
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing processed data and configuration
    """
    print("\n--- Processing GoEmotions Dataset ---")
    
    print("Loading raw data...")
    # Load GoEmotions data
    emotions_df = load_goemotions(datadir)
    
    # Define emotion columns
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    print("Preprocessing text...")
    # Apply preprocessing to the text
    emotions_df['processed_text'] = emotions_df['text'].apply(preprocess_text)
    
    print("Tokenizing text...")
    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="")
    tokenizer.fit_on_texts(emotions_df['processed_text'])
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(emotions_df['processed_text'])
    
    # Pad sequences
    X_padded = pad_sequences(sequences, maxlen=max_text_length, padding='post')
    
    # Get multi-label outputs
    y_multi = emotions_df[emotion_columns].values
    
    # Split data into train, validation, and test sets
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_padded, y_multi, test_size=test_split, random_state=seed, stratify=np.argmax(y_multi, axis=1)
    )
    
    # Then split the temp set into training and validation
    test_val_ratio = validation_split / (1 - test_split)  # Adjusted validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=test_val_ratio, random_state=seed, stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train, emotion_columns)
    
    # Create configuration
    config = {
        'max_words': max_words,
        'max_text_length': max_text_length,
        'vocab_size': min(max_words, len(tokenizer.word_index) + 1),
        'num_classes': len(emotion_columns),
        'emotion_columns': emotion_columns
    }
    
    processed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'class_weights': class_weights,
        'config': config
    }
    
    return processed_data

def save_processed_data(data_dict, output_dir, dataset_name):
    """
    Save processed data to files.
    
    Args:
        data_dict (dict): Dictionary containing processed data
        output_dir (str): Output directory
        dataset_name (str): Name of the dataset
    """
    # Create dataset-specific directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(dataset_dir, 'X_train.npy'), data_dict['X_train'])
    np.save(os.path.join(dataset_dir, 'y_train.npy'), data_dict['y_train'])
    np.save(os.path.join(dataset_dir, 'X_val.npy'), data_dict['X_val'])
    np.save(os.path.join(dataset_dir, 'y_val.npy'), data_dict['y_val'])
    np.save(os.path.join(dataset_dir, 'X_test.npy'), data_dict['X_test'])
    np.save(os.path.join(dataset_dir, 'y_test.npy'), data_dict['y_test'])
    
    # Save tokenizer
    with open(os.path.join(dataset_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(data_dict['tokenizer'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save configuration
    with open(os.path.join(dataset_dir, 'config.pickle'), 'wb') as handle:
        pickle.dump(data_dict['config'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save class weights if available
    if 'class_weights' in data_dict:
        with open(os.path.join(dataset_dir, 'class_weights.pickle'), 'wb') as handle:
            pickle.dump(data_dict['class_weights'], handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    """
    Main function to process multiple datasets.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess text datasets')
    
    # General arguments
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Output directory for processed data')
    
    # Dataset selection
    parser.add_argument('--process_imdb', action='store_true',
                        help='Process IMDb movie reviews dataset')
    parser.add_argument('--process_goemotions', action='store_true',
                        help='Process GoEmotions dataset')
    
    # IMDb specific arguments
    parser.add_argument('--imdb_dir', type=str, 
                        help='Path to the aclImdb directory')
    parser.add_argument('--imdb_max_words', type=int, default=10000,
                        help='Maximum vocabulary size for IMDb')
    parser.add_argument('--imdb_max_length', type=int, default=600,
                        help='Maximum sequence length for IMDb')
    
    # GoEmotions specific arguments
    parser.add_argument('--goemotions_dir', type=str,
                        help='Path to the GoEmotions directory')
    parser.add_argument('--goemotions_max_words', type=int, default=5000,
                        help='Maximum vocabulary size for GoEmotions')
    parser.add_argument('--goemotions_max_length', type=int, default=100,
                        help='Maximum sequence length for GoEmotions')
    
    # Common arguments
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Proportion of data to use for testing (GoEmotions only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process IMDb dataset if requested
    if args.process_imdb:
        if args.imdb_dir is None:
            print("Error: --imdb_dir is required when --process_imdb is set")
            return
        
        imdb_data = process_imdb(
            args.imdb_dir,
            max_words=args.imdb_max_words,
            max_review_length=args.imdb_max_length,
            validation_split=args.validation_split,
            seed=args.seed
        )
        
        save_processed_data(imdb_data, args.output_dir, 'imdb')
        print("IMDb dataset processing complete!")
    
    # Process GoEmotions dataset if requested
    if args.process_goemotions:
        if args.goemotions_dir is None:
            print("Error: --goemotions_dir is required when --process_goemotions is set")
            return
        
        goemotions_data = process_goemotions(
            args.goemotions_dir,
            max_words=args.goemotions_max_words,
            max_text_length=args.goemotions_max_length,
            validation_split=args.validation_split,
            test_split=args.test_split,
            seed=args.seed
        )
        
        save_processed_data(goemotions_data, args.output_dir, 'goemotions')
        print("GoEmotions dataset processing complete!")
    
    # If no dataset was specified, show help
    if not (args.process_imdb or args.process_goemotions):
        parser.print_help()
        print("\nError: At least one dataset must be specified (--process_imdb or --process_goemotions)")

if __name__ == "__main__":
    main()