# Syed Ahmed
# CS 4375.001 
# Assignment 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

class NeuralNet:
    def __init__(self, dataFile):
        self.raw_input = pd.read_csv(dataFile, sep=';', encoding='utf-8')
        self.processed_data = None
        self.model_history = []  

    def preprocess(self):
        if self.raw_input['y'].dtype == 'object':
            self.raw_input['y'] = self.raw_input['y'].apply(lambda x: 1 if x == 'yes' else 0)

        categorical_columns = self.raw_input.select_dtypes(include=['object']).columns.tolist()
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(self.raw_input[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        self.raw_input.drop(categorical_columns, axis=1, inplace=True)
        self.processed_data = pd.concat([self.raw_input, encoded_df], axis=1)
        features = self.processed_data.drop('y', axis=1)
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        self.processed_data = pd.concat([scaled_features, self.processed_data['y']], axis=1)

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]  # Features
        y = self.processed_data.iloc[:, (ncols-1)]  # Target 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameters
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [100, 200]          
        num_hidden_layers = [2]         

        results = []
        fig, ax = plt.subplots(figsize=(12, 8))  

        for activation in activations:
            for lr in learning_rates:
                for epochs in max_iterations:
                    for layers in num_hidden_layers:
                        hidden_layer_sizes = (64,) * layers 
                        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                              activation=activation, 
                                              learning_rate_init=lr, 
                                              max_iter=epochs, 
                                              random_state=42)

                        history = model.fit(X_train, y_train)

                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)

                        train_accuracy = model.score(X_train, y_train)
                        test_accuracy = model.score(X_test, y_test)
                        train_mse = mean_squared_error(y_train, y_train_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)

                        label = f"Activation={activation}, LR={lr}, Epochs={epochs}, Layers={layers}"
                        self.model_history.append((label, train_accuracy, test_accuracy, train_mse, test_mse))

                        results.append([activation, lr, epochs, layers, train_accuracy, test_accuracy, train_mse, test_mse])

                        ax.plot(model.loss_curve_, label=label)

        results_df = pd.DataFrame(results, columns=['Activation', 'Learning Rate', 'Epochs', 'Layers', 'Train Accuracy', 'Test Accuracy', 'Train MSE', 'Test MSE'])
        print("\nResults Table:")
        print(results_df)

        ax.set_title('Model Loss Curve (Loss vs Epochs)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/Shodformula/NeuralNet/main/bank.csv"
    neural_network = NeuralNet(url)
    neural_network.preprocess()
    neural_network.train_evaluate()
