cmake_minimum_required(VERSION 3.14)
project(mlcpp_examples CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the main mlcpp library
find_library(MLCPP_LIBRARY mlcpp_core PATHS ${CMAKE_SOURCE_DIR}/../../build)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../../lib/mlcpp/include)

# Create executables for each example
add_executable(linear_regression_example linear_regression_example.cpp)
target_link_libraries(linear_regression_example ${MLCPP_LIBRARY})

# Add more examples as they are created
# add_executable(another_example another_example.cpp)
# target_link_libraries(another_example ${MLCPP_LIBRARY})