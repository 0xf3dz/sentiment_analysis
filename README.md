# sentiment_analysis


1) python preprocess.py
2) python models.py
3) python train.py
   --model: Choose which model architecture to train
  Default: 'cnn'
  Available options: Any model defined in the MODEL_REGISTRY in the models.py file

  --batch_size: Set the batch size for training
  Default: 64
  
  --epochs: Set the maximum number of training epochs
  Default: 15   
5) python evaluate.py
  python evaluate_model.py to evaluate and compare all models
  python evaluate_model.py --model cnn to evaluate only the CNN model
  python evaluate_model.py --model lstm --compare to evaluate the LSTM model and compare it with any previously evaluated models
