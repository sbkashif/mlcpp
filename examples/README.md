# MLCPP Examples

This directory contains examples demonstrating how to use the MLCPP library in both C++ and Python applications.

## C++ Examples

The `cpp/` directory contains examples showing how to use MLCPP directly from C++ applications:

- **[Linear Regression Example](cpp/linear_regression_example.cpp)**: A comprehensive example showing how to use linear regression with both synthetic and realistic data.

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
```

## Python Examples

The `python/` directory contains examples showing how to use MLCPP through its Python bindings:

- **[Minimal Example](python/minimal_example.py)**: A simple example showing basic usage with synthetic data.
- **[Simple Housing Example](python/simple_housing_example.py)**: Using MLCPP with the California Housing dataset.
- **[Housing Comparison Example](python/housing_comparison_example.py)**: Comparison of MLCPP vs scikit-learn.

### Running Python Examples

To run the Python examples:

```bash
# Make sure you have activated your conda environment and built the project
conda activate your_env_name  # Replace with your environment name

# Run the minimal example
python python/minimal_example.py

# Run the housing example
python python/simple_housing_example.py

# Run the comparison example
python python/housing_comparison_example.py
```

## Performance Comparison

For a detailed performance comparison between MLCPP and scikit-learn, see the [Comparison Documentation](python/COMPARISON_README.md).
