#!/usr/bin/env python
"""
Example comparing mlcpp NeuralNetwork with scikit-learn's MLPClassifier
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Import the NeuralNetwork model from mlcpp
try:
    # Option 1: Import from mlcpp package (if installed with pip install -e .)
    from mlcpp import NeuralNetwork, ActivationFunction
    print("Successfully imported from mlcpp package!")
except ImportError:
    try:
        # Option 2: Import directly from pymlcpp module
        from pymlcpp import NeuralNetwork, ActivationFunction
        print("Successfully imported from pymlcpp module!")
    except ImportError:
        try:
            # Option 3: Add the lib/mlcpp/python directory to sys.path and try again
            module_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                    'lib', 'mlcpp', 'python')
            sys.path.append(module_dir)
            print(f"Added to Python path: {module_dir}")
            from pymlcpp import NeuralNetwork, ActivationFunction
            print("Successfully imported pymlcpp after adding to path!")
        except ImportError:
            print("ERROR: Could not import NeuralNetwork. Make sure you've built the project correctly.")
            print("Try running the setup.sh script or follow the manual build instructions in README.md")
            sys.exit(1)

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary for a binary classification model"""
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    h = 0.02  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions for each point in the mesh
    Z = np.array([])
    
    # Check if we're using sklearn or mlcpp model
    if isinstance(model, MLPClassifier):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        # Using our mlcpp NeuralNetwork
        mesh_points = np.c_[xx.ravel(), yy.ravel()].tolist()
        Z = np.array(model.predict_binary(mesh_points))
    
    # Reshape to match xx shape
    Z = Z.reshape(xx.shape)
    
    # Plot the contour
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    return plt

def main():
    print("MLCPP Neural Network vs. Scikit-learn Comparison")
    print("===============================================")
    
    # Generate a non-linearly separable dataset (moons)
    print("\nGenerating moon dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Normalize the features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert numpy arrays to lists for mlcpp
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()
    
    # Train MLCPP Neural Network
    print("\nTraining mlcpp Neural Network...")
    start_time = time.time()
    
    # Create a neural network with 2 input neurons, 8 hidden neurons in each hidden layer, and 1 output neuron
    nn_mlcpp = NeuralNetwork(
        layers=[2, 8, 8, 1],
        activation=ActivationFunction.TANH,
        learning_rate=0.01,
        max_iterations=1000,
        batch_size=32
    )
    
    # Train the model
    nn_mlcpp.fit(X_train_list, y_train_list)
    mlcpp_training_time = time.time() - start_time
    
    # Get predictions
    mlcpp_predictions = nn_mlcpp.predict_binary(X_test_list)
    mlcpp_accuracy = accuracy_score(y_test, mlcpp_predictions)
    
    print(f"MLCPP Neural Network training time: {mlcpp_training_time:.4f} seconds")
    print(f"MLCPP Neural Network accuracy: {mlcpp_accuracy:.4f}")
    
    # Train scikit-learn MLPClassifier
    print("\nTraining scikit-learn MLPClassifier...")
    start_time = time.time()
    
    # Create a comparable MLP model
    nn_sklearn = MLPClassifier(
        hidden_layer_sizes=(8, 8),
        activation='tanh',
        solver='sgd',
        learning_rate_init=0.01,
        max_iter=1000,
        batch_size=32,
        random_state=42
    )
    
    # Train the model
    nn_sklearn.fit(X_train, y_train)
    sklearn_training_time = time.time() - start_time
    
    # Get predictions
    sklearn_predictions = nn_sklearn.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    
    print(f"scikit-learn MLP training time: {sklearn_training_time:.4f} seconds")
    print(f"scikit-learn MLP accuracy: {sklearn_accuracy:.4f}")
    
    # Compare the results
    print("\nComparison Results:")
    print("------------------")
    print(f"MLCPP Neural Network accuracy: {mlcpp_accuracy:.4f}, Training time: {mlcpp_training_time:.4f}s")
    print(f"scikit-learn MLP accuracy:     {sklearn_accuracy:.4f}, Training time: {sklearn_training_time:.4f}s")
    
    time_ratio = mlcpp_training_time / sklearn_training_time
    accuracy_ratio = mlcpp_accuracy / sklearn_accuracy
    
    print(f"MLCPP/scikit-learn time ratio: {time_ratio:.2f}x")
    print(f"MLCPP/scikit-learn accuracy ratio: {accuracy_ratio:.2f}x")
    
    # Plot decision boundaries
    mlcpp_plot = plot_decision_boundary(nn_mlcpp, X_scaled, y, "MLCPP Neural Network Decision Boundary")
    mlcpp_plot.savefig("mlcpp_neural_network_boundary.png")
    
    sklearn_plot = plot_decision_boundary(nn_sklearn, X_scaled, y, "scikit-learn MLP Decision Boundary")
    sklearn_plot.savefig("sklearn_mlp_boundary.png")
    
    print("\nDecision boundary plots saved to:")
    print("- mlcpp_neural_network_boundary.png")
    print("- sklearn_mlp_boundary.png")
    
    try:
        mlcpp_plot.show()
        sklearn_plot.show()
    except:
        print("Note: Couldn't display plots interactively. Please check the saved PNG files.")

if __name__ == "__main__":
    main()