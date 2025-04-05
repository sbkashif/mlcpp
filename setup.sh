#!/bin/bash

# Exit on error
set -e

echo "=========================================================="
echo "ML-CPP Library Setup Script"
echo "=========================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get conda environment name from user or use default
read -p "Enter conda environment name (default: mlcpp): " ENV_NAME
ENV_NAME=${ENV_NAME:-mlcpp}

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to use the existing environment? (y/n): " USE_EXISTING
    if [[ $USE_EXISTING =~ ^[Yy]$ ]]; then
        echo "Using existing environment: $ENV_NAME"
    else
        echo "Please run this script again with a different environment name."
        exit 1
    fi
else
    # Create conda environment
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.9 numpy
    echo "Installing additional packages..."
    conda install -y -n $ENV_NAME -c conda-forge jupyter matplotlib scikit-learn
fi

# Activate conda environment
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir -p build
fi

# Build the project
echo "Building the project..."
cd build
cmake ..
make -j4

# Return to the project root
cd ..

# Test the Python bindings
echo "Testing Python bindings..."
python -c "try:
    from pymlcpp import LinearRegression
    print('Success: Python bindings loaded correctly!')
    model = LinearRegression(0.01, 1000, 1e-6)
    print('LinearRegression model created successfully.')
except Exception as e:
    print(f'Error loading Python bindings: {e}')
    import sys
    sys.exit(1)"

echo "=========================================================="
echo "Setup complete! Here's how to use the library:"
echo ""
echo "1. Always activate the conda environment before using the library:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Run the examples:"
echo "   python python/minimal_example.py"
echo ""
echo "3. For Jupyter notebook examples:"
echo "   jupyter notebook python/example_notebook.ipynb"
echo "=========================================================="