#include "mlcpp/models/neural_network.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Simple function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-2) {
    return std::abs(a - b) < epsilon;
}

// Function to calculate accuracy
double calculate_accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }
    
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (approx_equal(y_true[i], y_pred[i], 0.5)) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y_true.size();
}

int main() {
    std::cout << "Testing Neural Network implementation..." << std::endl;
    
    // XOR problem - a classic non-linearly separable problem
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<double> y = {
        0.0,  // XOR: 0 ^ 0 = 0
        1.0,  // XOR: 0 ^ 1 = 1
        1.0,  // XOR: 1 ^ 0 = 1
        0.0   // XOR: 1 ^ 1 = 0
    };
    
    std::cout << "XOR Problem Test:" << std::endl;
    std::cout << "Training data:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "], Target: " << y[i] << std::endl;
    }
    
    // Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    std::cout << "\nCreating neural network with architecture [2, 4, 1]..." << std::endl;
    mlcpp::models::NeuralNetwork model({2, 4, 1}, mlcpp::models::ActivationFunction::SIGMOID, 0.1, 10000, 0, 1e-6);
    
    // Train the model
    std::cout << "Training model..." << std::endl;
    model.fit(X, y);
    
    // Test predictions
    std::cout << "\nTesting predictions:" << std::endl;
    std::vector<double> y_pred = model.predictBinary(X);
    
    std::cout << std::left;
    std::cout << std::setw(20) << "Input" << std::setw(15) << "Expected" << std::setw(15) << "Predicted" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << std::setw(20) << ("[" + std::to_string(X[i][0]) + ", " + std::to_string(X[i][1]) + "]");
        std::cout << std::setw(15) << y[i];
        std::cout << std::setw(15) << y_pred[i] << std::endl;
    }
    
    // Calculate accuracy
    double accuracy = calculate_accuracy(y, y_pred);
    std::cout << "\nModel accuracy: " << (accuracy * 100.0) << "%" << std::endl;
    
    // The XOR problem should be well-solved by a neural network with a hidden layer,
    // so we expect high accuracy (at least 75%, which would be 3/4 correct)
    assert(accuracy >= 0.75);
    
    std::cout << "\nTesting more complex network architecture..." << std::endl;
    // Test different activation functions - ReLU
    mlcpp::models::NeuralNetwork relu_model({2, 8, 8, 1}, mlcpp::models::ActivationFunction::RELU, 0.01, 20000, 0, 1e-6);
    relu_model.fit(X, y);
    
    std::vector<double> relu_pred = relu_model.predictBinary(X);
    double relu_accuracy = calculate_accuracy(y, relu_pred);
    std::cout << "ReLU Network accuracy: " << (relu_accuracy * 100.0) << "%" << std::endl;
    assert(relu_accuracy >= 0.75);
    
    // Test TANH activation
    mlcpp::models::NeuralNetwork tanh_model({2, 4, 1}, mlcpp::models::ActivationFunction::TANH, 0.05, 10000, 0, 1e-6);
    tanh_model.fit(X, y);
    
    std::vector<double> tanh_pred = tanh_model.predictBinary(X);
    double tanh_accuracy = calculate_accuracy(y, tanh_pred);
    std::cout << "TANH Network accuracy: " << (tanh_accuracy * 100.0) << "%" << std::endl;
    assert(tanh_accuracy >= 0.75);
    
    std::cout << "All neural network tests passed successfully!" << std::endl;
    return 0;
}