#!/bin/bash
# Installation script for mlcpp

# Print step information with formatting
print_step() {
    echo -e "\n\033[1;34m===> $1\033[0m"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python first."
    exit 1
fi

print_step "Building C++ library and Python module"
mkdir -p build
cd build
cmake ..
make

print_step "Installing as Python package (development mode)"
cd ..
pip install -e .

print_step "Installation complete!"
echo "You can now import the library in Python using:"
echo "    import mlcpp               # For top-level package"
echo "    from mlcpp import LinearRegression"
echo ""
echo "Or directly import the module:"
echo "    import pymlcpp             # If it's in your Python path"
echo "    from pymlcpp import LinearRegression"
echo ""
echo "To test the installation, run the minimal example:"
echo "    python examples/python/minimal_example.py"