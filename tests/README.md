# ML-CPP Tests

This directory contains the unit tests for the ML-CPP library.

## Test Coverage

The following components are tested:

### C++ Tests

#### Linear Regression (`test_linear_regression.cpp`)

Tests the `LinearRegression` implementation with the following validations:
- Model training on synthetic data (y = 2*x1 + 3*x2 + 1)
- Parameter recovery (coefficient and intercept estimation)
- Model prediction accuracy on test examples
- Gradient descent convergence

## Running the Tests

### From Command Line

From the project root directory:

```bash
# Build the project (if not already built)
mkdir -p build
cd build
cmake ..
make

# Run the tests
ctest
```

Or run an individual test:

```bash
./tests/test_linear_regression
```

### Test Results

#### Linear Regression Test

The latest test execution showed:

```
Learned model: y = 1 + 2*x1 + 3*x2
All tests passed!
```

This confirms that:
1. The gradient descent algorithm correctly converges to the expected parameters
2. The model can accurately predict on unseen data
3. The implementation handles multi-dimensional feature vectors correctly

## Integration with Python API

For tests of the Python bindings, refer to the Python examples in the `python/` directory:
- `minimal_example.py` - Minimal test of the Python API
- `example.py` - More comprehensive example
- `housing_comparison_example.py` - Benchmarking against scikit-learn

These Python examples serve as both demonstrations and validation tests for the Python API.