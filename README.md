# Neural Network Classifier on Bank Marketing Dataset

## Overview
This project demonstrates a simple neural network classifier applied to the Bank Marketing dataset. Using a multi-layer perceptron (MLP), the code classifies whether a client will subscribe to a term deposit based on various features. The classifier is trained on preprocessed data, evaluated using different hyperparameters, and visualizes the model's loss over epochs.

## Requirements
To run this project, you'll need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install them using the following command:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset
The code utilizes a bank marketing dataset available at the following URL: https://raw.githubusercontent.com/Shodformula/NeuralNet/main/bank.csv
The dataset includes various features (both categorical and numerical) related to client details and their interactions with the bank. The target variable, y, indicates if a client has subscribed to a term deposit (yes or no).

## Code Structure
The code is organized into a class `NeuralNet` with the following methods:

- `__init__`: Initializes the object, loads the dataset, and sets up data structures.
- `preprocess`: Preprocesses the dataset by encoding categorical variables using one-hot encoding, normalizing numerical features, and scaling them. This method also converts the target column into binary format.
- `train_evaluate`: Trains and evaluates the neural network using a grid search approach for different hyperparameters. It plots the loss curve for each configuration and outputs a summary table of results.

## Hyperparameters
The following hyperparameters are varied in the `train_evaluate` method:

- **Activation Functions**: `logistic`, `tanh`, `relu`
- **Learning Rates**: `0.01`, `0.1`
- **Max Iterations (Epochs)**: `100`, `200`
- **Number of Hidden Layers**: `2`

## Output
After training, the code outputs a table of results containing:

- Train and test accuracy
- Train and test mean squared error (MSE)
- Loss curves for each model configuration

## How to Use

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### Run the Script
```bash
python NeuralNet.py
