# sentiment_analysis


1) python preprocess.py
2) python models.py
3) python train.py
command-line arguments:
--model: Choose which model architecture to train
  Default: 'cnn'
  Available options: Any model defined in the MODEL_REGISTRY in the models.py file

--batch_size: Set the batch size for training
  Default: 64
  
--epochs: Set the maximum number of training epochs
  Default: 15   
5) python evaluate.py
command-line arguments:
no argument to evaluate and compare all models
--model cnn to evaluate only the CNN model
--model lstm --compare to evaluate the LSTM model and compare it with any previously evaluated models


# Sentiment Analysis

A machine learning project for sentiment analysis on IMDb movie reviews.

## Overview

This project uses deep learning models to classify movie reviews as positive or negative. It includes scripts for data preprocessing, model definition, training, and evaluation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment_analysis.git
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the Data

```bash
python preprocess.py
```

This step processes the raw IMDb review data and prepares it for model training.

### 2. View Available Models

```bash
python models.py
```

Lists available model architectures defined in the MODEL_REGISTRY.

### 3. Train Models

```bash
python train_model.py [options]
```

#### Command-line Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture to use (options from MODEL_REGISTRY) | `cnn` |
| `--batch_size` | Batch size for training | `64` |
| `--epochs` | Maximum number of training epochs | `15` |

#### Examples:

```bash
# Train default CNN model
python train_model.py

# Train LSTM model
python train_model.py --model lstm

# Train with custom settings
python train_model.py --model transformer --batch_size 32 --epochs 20
```

### 4. Evaluate Models

```bash
python evaluate_model.py [options]
```

#### Command-line Arguments:

| Argument | Description |
|----------|-------------|
| `--model` | Specific model to evaluate (omit to evaluate all models) |
| `--compare` | Compare all evaluated models |

#### Examples:

```bash
# Evaluate and compare all models
python evaluate_model.py

# Evaluate only the CNN model
python evaluate_model.py --model cnn  

# Evaluate LSTM model and compare with previously evaluated models
python evaluate_model.py --model lstm --compare
```

## Model Outputs

After training and evaluation, the following artifacts will be generated:

- **Models**: Saved in the `models/` directory
- **Training History**: Plots and data in `plots/` and `results/` directories
- **Evaluation Results**: Performance metrics and comparisons in `results/` directory

## Project Structure

```
sentiment_analysis/
├── preprocess.py         # Data preprocessing
├── models.py             # Model architecture definitions
├── train_model.py        # Model training script
├── evaluate_model.py     # Model evaluation script
├── processed_data/       # Preprocessed datasets
├── models/               # Saved model files
├── results/              # Evaluation results and metrics
├── plots/                # Performance visualizations
└── logs/                 # Training logs for TensorBoard
```

## License

[MIT License](LICENSE)
