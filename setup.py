#!/usr/bin/env python
"""
Setup script for mlcpp Python bindings
"""
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import platform

class CustomInstallCommand(install):
    """Custom install command to build the C++ library and Python bindings."""
    def run(self):
        # Build the C++ library and Python bindings
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        
        # Get the current directory
        current_dir = os.getcwd()
        
        try:
            # Change to the build directory
            os.chdir(build_dir)
            
            # Run CMake
            subprocess.check_call(["cmake", ".."])
            
            # Run make
            subprocess.check_call(["make"])
            
        finally:
            # Change back to the original directory
            os.chdir(current_dir)
        
        # Run the standard install
        install.run(self)

# Define the package data
package_data = {
    'mlcpp': ['python/*.so', 'python/*.dylib', 'python/*.dll'],
}

setup(
    name="mlcpp",
    version="0.1.0",
    description="C++ Machine Learning Library with Python bindings",
    author="Salman Kashif",
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    package_data=package_data,
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    },
    python_requires=">=3.6",
)