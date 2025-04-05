# MLCPP Examples

This directory contains examples demonstrating how to use the MLCPP library in both C++ and Python applications.

## C++ Examples

The `cpp/` directory contains examples showing how to use MLCPP directly from C++ applications:

- **[Linear Regression Example](cpp/linear_regression_example.cpp)**: A comprehensive example showing how to use linear regression with both synthetic and realistic data.
- **[Neural Network Example](cpp/neural_network_example.cpp)**: Demonstrates how to create, train, and evaluate neural networks with different architectures and activation functions.

### Building and Running C++ Examples

To build and run the C++ examples:

```bash
# Navigate to the cpp examples directory
cd examples/cpp

# Create a build directory and navigate to it
mkdir -p build && cd build

# Configure and build the examples
cmake ..
make

# Run the linear regression example
./linear_regression_example

# Run the neural network example
./neural_network_example
```

## Python Examples

The `python/` directory contains examples showing how to use MLCPP through its Python bindings:

- **[Minimal Example](python/minimal_example.py)**: A simple example showing basic usage with synthetic data.

### Linear Regression Examples
- **[Simple Housing Example](python/simple_housing_example.py)**: Using MLCPP with the California Housing dataset.
- **[Housing Comparison Example](python/housing_comparison_example.py)**: Comparison of MLCPP vs scikit-learn for linear regression.
- For detailed performance metrics, see the [Linear Regression Comparison](python/COMPARISON_README.md).

### Neural Network Examples
- **[Neural Network Comparison Example](python/neural_network_comparison_example.py)**: Comparison of MLCPP's neural network implementation vs scikit-learn's MLPClassifier.
- For detailed performance metrics, see the [Neural Network Comparison](python/NEURAL_NETWORK_COMPARISON.md).

### Running Python Examples

To run the Python examples:

```bash
# Make sure you have activated your conda environment and built the project
conda activate your_env_name  # Replace with your environment name

# Run the minimal example
python python/minimal_example.py

# Run the housing example
python python/simple_housing_example.py

# Run the comparison examples
python python/housing_comparison_example.py
python python/neural_network_comparison_example.py
```

## Performance Highlights

### Linear Regression Performance

- Comparable accuracy to scikit-learn with R² scores around 0.7-0.8 on the housing dataset
- Small memory footprint ideal for embedded applications
- For full details, see the [Linear Regression Comparison](python/COMPARISON_README.md).

### Neural Network Performance

- 95.67% accuracy on the moons dataset (99% of scikit-learn's performance)
- Best results using TANH activation (100% accuracy on XOR problem)
- Excellent performance on small datasets
- For full details, see the [Neural Network Comparison](python/NEURAL_NETWORK_COMPARISON.md).
