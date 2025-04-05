#!/usr/bin/env python
"""
Simple example of using mlcpp LinearRegression with the California Housing dataset
"""
import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression as SKLinearRegression
import sys
import os

# Import the LinearRegression model from mlcpp
try:
    # Option 1: Import from mlcpp package (if installed with pip install -e .)
    from mlcpp import LinearRegression as MLCPPLinearRegression
    print("Successfully imported from mlcpp package!")
except ImportError:
    try:
        # Option 2: Import directly from pymlcpp module
        from pymlcpp import LinearRegression as MLCPPLinearRegression
        print("Successfully imported from pymlcpp module!")
    except ImportError:
        try:
            # Option 3: Add the lib/mlcpp/python directory to sys.path and try again
            module_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                    'lib', 'mlcpp', 'python')
            sys.path.append(module_dir)
            print(f"Added to Python path: {module_dir}")
            from pymlcpp import LinearRegression as MLCPPLinearRegression
            print("Successfully imported pymlcpp after adding to path!")
        except ImportError as e:
            print(f"Error importing module: {e}")
            print("\nPlease install the package using one of these methods:")
            print("1. Run the install script: ./install.sh")
            print("2. Install with pip: pip install -e .")
            print("3. Build manually: mkdir -p build && cd build && cmake .. && make")
            sys.exit(1)

try:
    # Load the California Housing dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to lists for mlcpp
    X_train_list = X_train_scaled.tolist()
    y_train_list = y_train.tolist()
    X_test_list = X_test_scaled.tolist()
    
    print("\n--- Training and evaluating MLCPP model ---\n")
    
    # Train MLCPP model
    print("Training mlcpp LinearRegression...")
    start_time = time.time()
    mlcpp_model = MLCPPLinearRegression(learning_rate=0.02, max_iterations=10000, tolerance=1e-6)
    mlcpp_model.fit(X_train_list, y_train_list)
    mlcpp_train_time = time.time() - start_time
    print(f"Training time: {mlcpp_train_time:.4f} seconds")
    
    # Get model coefficients
    print("Model coefficients:")
    coefs = mlcpp_model.get_coefficients()
    for i, (feature, coef) in enumerate(zip(feature_names, coefs)):
        print(f"  {feature}: {coef:.6f}")
    print(f"  intercept: {mlcpp_model.get_intercept():.6f}")
    
    # Make predictions and evaluate
    start_time = time.time()
    y_pred_mlcpp = mlcpp_model.predict(X_test_list)
    mlcpp_pred_time = time.time() - start_time
    
    # Convert predictions to numpy array for evaluation
    y_pred_mlcpp_np = np.array(y_pred_mlcpp)
    
    # Calculate metrics
    mse_mlcpp = mean_squared_error(y_test, y_pred_mlcpp_np)
    r2_mlcpp = r2_score(y_test, y_pred_mlcpp_np)
    
    print(f"Prediction time: {mlcpp_pred_time:.4f} seconds")
    print(f"Mean Squared Error: {mse_mlcpp:.6f}")
    print(f"R² Score: {r2_mlcpp:.6f}")
    
    # Print a few sample predictions
    print("\nSample predictions:")
    print("Actual value | Predicted value")
    print("--------------------------")
    for i in range(5):  # Show first 5 predictions
        print(f"{y_test[i]:.4f} | {y_pred_mlcpp[i]:.4f}")
    
    # Now train and evaluate sklearn's LinearRegression model
    print("\n\n--- Training and evaluating scikit-learn model ---\n")
    print("Training scikit-learn LinearRegression...")
    start_time = time.time()
    sk_model = SKLinearRegression()
    sk_model.fit(X_train_scaled, y_train)
    sk_train_time = time.time() - start_time
    print(f"Training time: {sk_train_time:.4f} seconds")
    
    # Get model coefficients
    print("Model coefficients:")
    for i, (feature, coef) in enumerate(zip(feature_names, sk_model.coef_)):
        print(f"  {feature}: {coef:.6f}")
    print(f"  intercept: {sk_model.intercept_:.6f}")
    
    # Make predictions and evaluate
    start_time = time.time()
    y_pred_sk = sk_model.predict(X_test_scaled)
    sk_pred_time = time.time() - start_time
    
    # Calculate metrics
    mse_sk = mean_squared_error(y_test, y_pred_sk)
    r2_sk = r2_score(y_test, y_pred_sk)
    
    print(f"Prediction time: {sk_pred_time:.4f} seconds")
    print(f"Mean Squared Error: {mse_sk:.6f}")
    print(f"R² Score: {r2_sk:.6f}")
    
    # Print a few sample predictions
    print("\nSample predictions:")
    print("Actual value | Predicted value")
    print("--------------------------")
    for i in range(5):  # Show first 5 predictions
        print(f"{y_test[i]:.4f} | {y_pred_sk[i]:.4f}")
    
    # Print comparison summary
    print("\n--- Performance Comparison ---")
    print(f"Training time: MLCPP: {mlcpp_train_time:.4f}s, scikit-learn: {sk_train_time:.4f}s (ratio: {mlcpp_train_time/sk_train_time:.2f}x)")
    print(f"Prediction time: MLCPP: {mlcpp_pred_time:.4f}s, scikit-learn: {sk_pred_time:.4f}s (ratio: {mlcpp_pred_time/sk_pred_time:.2f}x)")
    print(f"MSE: MLCPP: {mse_mlcpp:.6f}, scikit-learn: {mse_sk:.6f}")
    print(f"R²: MLCPP: {r2_mlcpp:.6f}, scikit-learn: {r2_sk:.6f}")

except Exception as e:
    print(f"Error running the example: {e}")
    print("If you're experiencing NumPy compatibility issues, you might need to:")
    print("1. Downgrade NumPy: pip install numpy<2")
    print("2. Or rebuild the mlcpp module with your current NumPy version")