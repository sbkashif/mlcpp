#!/usr/bin/env python
"""
Ultra simple example of using mlcpp LinearRegression with a synthetic housing dataset
"""
import sys
import os
import time
import random

# Import the LinearRegression model from mlcpp
try:
    # Option 1: Import from mlcpp package (if installed with pip install -e .)
    from mlcpp import LinearRegression
    print("Successfully imported from mlcpp package!")
except ImportError:
    try:
        # Option 2: Import directly from pymlcpp module
        from pymlcpp import LinearRegression
        print("Successfully imported from pymlcpp module!")
    except ImportError:
        try:
            # Option 3: Add the lib/mlcpp/python directory to sys.path and try again
            module_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                    'lib', 'mlcpp', 'python')
            sys.path.append(module_dir)
            print(f"Added to Python path: {module_dir}")
            from pymlcpp import LinearRegression
            print("Successfully imported pymlcpp after adding to path!")
        except ImportError as e:
            print(f"Error importing module: {e}")
            print("\nPlease install the package using one of these methods:")
            print("1. Run the install script: ./install.sh")
            print("2. Install with pip: pip install -e .")
            print("3. Build manually: mkdir -p build && cd build && cmake .. && make")
            sys.exit(1)

# Create a synthetic housing dataset
# Features: sqft, bedrooms, bathrooms, age_years, distance_to_city_center
print("Creating synthetic housing dataset...")
random.seed(42)  # For reproducibility

# Generate 1000 samples
n_samples = 1000
feature_names = ["sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_center"]

# Feature data
X = []
for _ in range(n_samples):
    sqft = random.randint(500, 3500)
    bedrooms = random.randint(1, 6)
    bathrooms = random.randint(1, 4)
    age = random.randint(0, 100)
    distance = random.uniform(0.1, 25.0)
    X.append([sqft, bedrooms, bathrooms, age, distance])

# Target: house price = 100*sqft + 50000*bedrooms + 75000*bathrooms - 1000*age - 15000*distance + noise
y = []
for sample in X:
    price = 100 * sample[0] + 50000 * sample[1] + 75000 * sample[2] - 1000 * sample[3] - 15000 * sample[4]
    # Add some noise
    price += random.uniform(-50000, 50000)
    y.append(price)

print(f"Dataset created with {n_samples} samples and {len(feature_names)} features")
print(f"Features: {feature_names}")

# Split the data into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * n_samples)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Standardize features manually
print("\nStandardizing features...")

# Calculate mean and std for each feature
means = [0] * len(feature_names)
stds = [0] * len(feature_names)

for j in range(len(feature_names)):
    values = [X_train[i][j] for i in range(len(X_train))]
    means[j] = sum(values) / len(values)
    stds[j] = (sum((x - means[j]) ** 2 for x in values) / len(values)) ** 0.5
    stds[j] = 1.0 if stds[j] == 0 else stds[j]  # Avoid division by zero

# Apply standardization
X_train_scaled = []
for sample in X_train:
    scaled = [(sample[j] - means[j]) / stds[j] for j in range(len(feature_names))]
    X_train_scaled.append(scaled)

X_test_scaled = []
for sample in X_test:
    scaled = [(sample[j] - means[j]) / stds[j] for j in range(len(feature_names))]
    X_test_scaled.append(scaled)

print("--- Training and evaluating MLCPP model ---")

# Train MLCPP model
print("Training mlcpp LinearRegression...")
start_time = time.time()
model = LinearRegression(learning_rate=0.01, max_iterations=1000, tolerance=1e-6)
model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.4f} seconds")

# Get model coefficients
print("Model coefficients:")
coefs = model.get_coefficients()
for i, (feature, coef) in enumerate(zip(feature_names, coefs)):
    print(f"  {feature}: {coef:.6f}")
print(f"  intercept: {model.get_intercept():.6f}")

# Make predictions
start_time = time.time()
y_pred = model.predict(X_test_scaled)
pred_time = time.time() - start_time
print(f"Prediction time: {pred_time:.4f} seconds")

# Calculate Mean Squared Error manually
mse = sum([(p - a) ** 2 for p, a in zip(y_pred, y_test)]) / len(y_pred)
print(f"Mean Squared Error: {mse:.6f}")

# Calculate R² manually
y_mean = sum(y_test) / len(y_test)
ss_total = sum([(y - y_mean) ** 2 for y in y_test])
ss_residual = sum([(y_true - y_pred) ** 2 for y_true, y_pred in zip(y_test, y_pred)])
r2 = 1 - (ss_residual / ss_total)
print(f"R² Score: {r2:.6f}")

# Print a few sample predictions
print("\nSample predictions:")
print("Actual value | Predicted value | Difference")
print("---------------------------------------------")
for i in range(5):  # Show first 5 predictions
    diff = y_test[i] - y_pred[i]
    print(f"{y_test[i]:.2f} | {y_pred[i]:.2f} | {diff:.2f}")