#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IMDb Movie Review Sentiment Analysis - Model Evaluation
This script evaluates trained models on the test set and generates performance metrics.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import seaborn as sns
import pandas as pd

def load_test_data():
    """
    Load the test dataset.
    
    Returns:
        tuple: (X_test, y_test, config)
    """
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    with open('processed_data/config.pickle', 'rb') as handle:
        config = pickle.load(handle)
    
    return X_test, y_test, config

def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        model_name (str): Name of the model
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, model_name):
    """
    Plot ROC curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred_prob (numpy.ndarray): Predicted probabilities
        model_name (str): Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.close()

def evaluate_model(model_path, model_name):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_path (str): Path to the saved model
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"Evaluating model: {model_name}")
    
    # Load test data
    X_test, y_test, _ = load_test_data()
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create directories for output
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, model_name)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_prob, model_name)
    
    # Create results dictionary
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save results to file
    with open(f'results/{model_name}_evaluation.pickle', 'wb') as file:
        pickle.dump(results, file)
    
    # Save text version of results
    with open(f'results/{model_name}_evaluation.txt', 'w') as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
    
    return results

def compare_models(model_names):
    """
    Compare multiple models on key metrics.
    
    Args:
        model_names (list): List of model names to compare
    """
    results = []
    
    for model_name in model_names:
        model_path = f'models/{model_name}_best.keras'
        if os.path.exists(model_path):
            try:
                # Load previously saved evaluation results if available
                eval_path = f'results/{model_name}_evaluation.pickle'
                if os.path.exists(eval_path):
                    with open(eval_path, 'rb') as file:
                        model_results = pickle.load(file)
                else:
                    model_results = evaluate_model(model_path, model_name)
                
                results.append({
                    'Model': model_name,
                    'Accuracy': model_results['accuracy'],
                    'Precision': model_results['precision'],
                    'Recall': model_results['recall'],
                    'F1 Score': model_results['f1_score']
                })
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    # Create comparison dataframe
    if results:
        df = pd.DataFrame(results)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
        plt.title('Model Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        plt.close()
        
        # Save comparison to CSV
        df.to_csv('results/model_comparison.csv', index=False)
        
        print("\nModel Comparison:")
        print(df.to_string(index=False))
    else:
        print("No models to compare")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate IMDb sentiment analysis models')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to evaluate (default: evaluate all available models)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all evaluated models')
    
    args = parser.parse_args()
    
    # Find available models
    available_models = []
    if os.path.exists('models'):
        for file in os.listdir('models'):
            if file.endswith('_best.keras'):
                model_name = file.replace('_best.keras', '')
                available_models.append(model_name)
    
    if not available_models:
        print("No trained models found in 'models' directory.")
        exit(1)
    
    # Evaluate specific model or all models
    if args.model:
        model_path = f'models/{args.model}_best.keras'
        if os.path.exists(model_path):
            evaluate_model(model_path, args.model)
        else:
            print(f"Model not found: {model_path}")
            print(f"Available models: {', '.join(available_models)}")
    else:
        # Evaluate all models
        for model_name in available_models:
            model_path = f'models/{model_name}_best.keras'
            evaluate_model(model_path, model_name)
    
    # Compare models if requested
    if args.compare or (not args.model):  # Compare by default if no specific model is specified
        compare_models(available_models)
    
    print("\nEvaluation completed!")
