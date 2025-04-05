cmake_minimum_required(VERSION 3.14)
project(mlcpp VERSION 0.1.0 LANGUAGES CXX)

# Add helpful message for users about conda environments
message(STATUS "NOTE: This project works best with Anaconda/Miniconda environments.")
message(STATUS "Please ensure you have activated your conda environment before building:")
message(STATUS "    conda activate myenv")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories - Updated to use the standardized structure
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/mlcpp/include)

# Library targets
add_library(mlcpp_core STATIC 
    lib/mlcpp/src/models/linear_regression.cpp
    lib/mlcpp/src/models/neural_network.cpp
)

# Find Python from the conda environment
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
    # Use the conda Python directly for everything
    set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "Python interpreter from conda environment")
    message(STATUS "Using Python from conda: ${PYTHON_EXECUTABLE}")
    
    # Use the simpler FindPythonInterp and FindPythonLibs modules which work better with conda
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    
    message(STATUS "Found Python interpreter: ${PYTHON_EXECUTABLE} (version: ${PYTHON_VERSION_STRING})")
    message(STATUS "Found Python libraries: ${PYTHON_LIBRARIES}")
    message(STATUS "Found Python include dirs: ${PYTHON_INCLUDE_DIRS}")
else()
    # Only use system Python if no conda environment is active
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()

# Find or fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Include pybind11's CMake modules properly
include(${pybind11_SOURCE_DIR}/tools/pybind11Tools.cmake)

# Python module - Updated to use the standardized structure
pybind11_add_module(pymlcpp 
    lib/mlcpp/python/pymlcpp.cpp
)
target_link_libraries(pymlcpp PRIVATE mlcpp_core)

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS mlcpp_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY lib/mlcpp/include/
    DESTINATION include
)

# Set output directory for the Python module
set_target_properties(pymlcpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/mlcpp/python
)

# Print a helpful message at the end
message(STATUS "")
message(STATUS "===============================================================")
message(STATUS "Build completed. To test the library:")
message(STATUS "1. Activate your conda environment: conda activate myenv")
message(STATUS "2. Run the tests: cd build && ctest")
message(STATUS "3. Try the Python examples in the examples/python directory")
message(STATUS "===============================================================")
message(STATUS "")