#!/usr/bin/env python
"""
Minimal example of using the mlcpp LinearRegression with minimal dependencies
"""

import sys
import os
import time

# Try different import options
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

# Create a simple dataset: y = 2*x1 + 3*x2 + 1
X = [
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0],
    [5.0, 5.0]
]
y = [6.0, 11.0, 16.0, 21.0, 26.0]  # 2*x1 + 3*x2 + 1

# Train the model
print("Training LinearRegression model...")
start_time = time.time()
model = LinearRegression(learning_rate=0.01, max_iterations=1000, tolerance=1e-6)
model.fit(X, y)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.4f} seconds")

# Get model parameters
intercept = model.get_intercept()
coefficients = model.get_coefficients()
print(f"Learned model: y = {intercept:.4f} + {coefficients[0]:.4f}*x1 + {coefficients[1]:.4f}*x2")
print(f"Expected model: y = 1.0000 + 2.0000*x1 + 3.0000*x2")

# Make predictions
X_test = [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]
y_expected = [8.0, 13.0, 18.0]  # True values (2*x1 + 3*x2 + 1)

print("\nMaking predictions:")
predictions = model.predict(X_test)

print("Test data | Prediction | Expected")
print("---------------------------------")
for i, (test, pred, exp) in enumerate(zip(X_test, predictions, y_expected)):
    print(f"[{test[0]}, {test[1]}] | {pred:.4f} | {exp:.4f}")

# Calculate mean squared error manually
mse = sum([(p - e)**2 for p, e in zip(predictions, y_expected)]) / len(predictions)
print(f"\nMean Squared Error: {mse:.6f}")