#include "mlcpp/models/neural_network.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>
#include <sstream>

// Helper functions for evaluation and visualization
double calculate_accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred, double threshold = 0.5) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }
    
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double pred_class = y_pred[i] >= threshold ? 1.0 : 0.0;
        if (std::abs(y_true[i] - pred_class) < 0.01) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y_true.size();
}

// Generate a synthetic dataset for binary classification
void generate_classification_data(
    std::vector<std::vector<double>>& X, 
    std::vector<double>& y,
    size_t n_samples = 500,
    double noise_level = 0.1
) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> radius_dist(0.0, 5.0);
    std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<> noise(0.0, noise_level);
    
    X.resize(n_samples, std::vector<double>(2));
    y.resize(n_samples);
    
    // Generate two intertwined spirals - a classic dataset for neural networks
    for (size_t i = 0; i < n_samples; ++i) {
        double label = (i % 2 == 0) ? 0.0 : 1.0;
        double radius = radius_dist(gen);
        double angle = angle_dist(gen) + 3.0 * radius * (label == 0.0 ? 1.0 : -1.0);
        
        X[i][0] = radius * std::cos(angle) + noise(gen);
        X[i][1] = radius * std::sin(angle) + noise(gen);
        y[i] = label;
    }
}

// Split data into training and testing sets
void split_data(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    double test_ratio = 0.2
) {
    size_t n_samples = X.size();
    size_t test_size = static_cast<size_t>(test_ratio * n_samples);
    size_t train_size = n_samples - test_size;
    
    // Create a random permutation of indices
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Split the data
    X_train.resize(train_size);
    y_train.resize(train_size);
    X_test.resize(test_size);
    y_test.resize(test_size);
    
    for (size_t i = 0; i < train_size; ++i) {
        X_train[i] = X[indices[i]];
        y_train[i] = y[indices[i]];
    }
    
    for (size_t i = 0; i < test_size; ++i) {
        X_test[i] = X[indices[train_size + i]];
        y_test[i] = y[indices[train_size + i]];
    }
}

// Simple ASCII visualization of the data and predictions
void visualize_predictions(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& predictions,
    int grid_size = 25
) {
    // Find min/max x and y values for grid
    double min_x = X[0][0], max_x = X[0][0];
    double min_y = X[0][1], max_y = X[0][1];
    
    for (const auto& point : X) {
        min_x = std::min(min_x, point[0]);
        max_x = std::max(max_x, point[0]);
        min_y = std::min(min_y, point[1]);
        max_y = std::max(max_y, point[1]);
    }
    
    // Add some margin
    double margin_x = (max_x - min_x) * 0.1;
    double margin_y = (max_y - min_y) * 0.1;
    min_x -= margin_x;
    max_x += margin_x;
    min_y -= margin_y;
    max_y += margin_y;
    
    std::vector<std::vector<char>> grid(grid_size, std::vector<char>(grid_size, ' '));
    
    // Plot predictions as grid background
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            double x = min_x + (max_x - min_x) * i / (grid_size - 1);
            double y = min_y + (max_y - min_y) * j / (grid_size - 1);
            
            // Map to grid coordinates
            grid[j][i] = '.';  // Default background
        }
    }
    
    // Plot points
    for (size_t i = 0; i < X.size(); ++i) {
        int grid_x = static_cast<int>((X[i][0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int grid_y = static_cast<int>((X[i][1] - min_y) / (max_y - min_y) * (grid_size - 1));
        
        // Ensure in bounds
        grid_x = std::max(0, std::min(grid_x, grid_size - 1));
        grid_y = std::max(0, std::min(grid_y, grid_size - 1));
        
        if (y[i] < 0.5) {
            grid[grid_y][grid_x] = 'o';  // Class 0
        } else {
            grid[grid_y][grid_x] = 'x';  // Class 1
        }
    }
    
    // Print the grid
    std::cout << "Data Visualization (o = Class 0, x = Class 1):" << std::endl;
    for (const auto& row : grid) {
        for (char c : row) {
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

// Example showing the XOR problem
void run_xor_example() {
    std::cout << "\n==== Neural Network with XOR Problem ====" << std::endl;
    
    // Create XOR dataset
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<double> y = {
        0.0,  // 0 XOR 0 = 0
        1.0,  // 0 XOR 1 = 1
        1.0,  // 1 XOR 0 = 1
        0.0   // 1 XOR 1 = 0
    };
    
    std::cout << "XOR Truth Table:" << std::endl;
    std::cout << "----------------" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] 
                  << "], Output: " << y[i] << std::endl;
    }
    
    // Create and train model
    std::cout << "\nTraining neural network..." << std::endl;
    mlcpp::models::NeuralNetwork model({2, 4, 1}, mlcpp::models::ActivationFunction::SIGMOID, 0.1, 5000);
    
    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Predict and evaluate
    std::vector<std::vector<double>> raw_predictions = model.predict(X);
    std::vector<double> predictions = model.predictBinary(X);
    
    std::cout << "\nModel predictions:" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << std::left << std::fixed << std::setprecision(4);
    std::cout << std::setw(15) << "Input" << std::setw(15) << "Target" 
              << std::setw(15) << "Raw Output" << std::setw(15) << "Prediction" << std::endl;
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::stringstream ss;
        ss << "[" << X[i][0] << ", " << X[i][1] << "]";
        std::cout << std::setw(15) << ss.str()
                  << std::setw(15) << y[i]
                  << std::setw(15) << raw_predictions[i][0]
                  << std::setw(15) << predictions[i] << std::endl;
    }
    
    double accuracy = calculate_accuracy(y, predictions);
    std::cout << "\nAccuracy: " << (accuracy * 100.0) << "%" << std::endl;
    
    // Get loss history
    std::vector<double> loss_history = model.getLossHistory();
    std::cout << "\nLoss history (first 5 and last 5 iterations):" << std::endl;
    
    // First 5 iterations
    size_t to_show = std::min(loss_history.size(), size_t(5));
    for (size_t i = 0; i < to_show; ++i) {
        std::cout << "Iteration " << i << ": " << loss_history[i] << std::endl;
    }
    
    // Last 5 iterations
    if (loss_history.size() > 10) {
        std::cout << "..." << std::endl;
        for (size_t i = loss_history.size() - 5; i < loss_history.size(); ++i) {
            std::cout << "Iteration " << i << ": " << loss_history[i] << std::endl;
        }
    }
}

// Example with a more complex classification task
void run_spiral_example() {
    std::cout << "\n==== Neural Network with Spiral Classification ====" << std::endl;
    
    // Generate spiral dataset
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    std::cout << "Generating spiral dataset..." << std::endl;
    generate_classification_data(X, y, 500, 0.2);
    
    // Split into training and testing sets
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    
    std::cout << "Splitting data into training/testing sets..." << std::endl;
    split_data(X, y, X_train, y_train, X_test, y_test, 0.3);
    
    std::cout << "Training set: " << X_train.size() << " samples" << std::endl;
    std::cout << "Testing set: " << X_test.size() << " samples" << std::endl;
    
    // Visualize the data
    visualize_predictions(X, y, y, 30);
    
    // Create a neural network with multiple hidden layers
    std::cout << "\nTraining neural network with architecture [2, 16, 8, 1]..." << std::endl;
    mlcpp::models::NeuralNetwork model({2, 16, 8, 1}, mlcpp::models::ActivationFunction::TANH, 0.01, 3000, 32);
    
    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X_train, y_train);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Evaluate on test set
    std::vector<double> train_pred = model.predictBinary(X_train);
    std::vector<double> test_pred = model.predictBinary(X_test);
    
    double train_accuracy = calculate_accuracy(y_train, train_pred);
    double test_accuracy = calculate_accuracy(y_test, test_pred);
    
    std::cout << "\nModel evaluation:" << std::endl;
    std::cout << "Training accuracy: " << (train_accuracy * 100.0) << "%" << std::endl;
    std::cout << "Testing accuracy: " << (test_accuracy * 100.0) << "%" << std::endl;
    
    // Get final loss
    std::vector<double> loss_history = model.getLossHistory();
    if (!loss_history.empty()) {
        std::cout << "Final loss: " << loss_history.back() << std::endl;
    }
}

int main() {
    std::cout << "MLCPP Neural Network C++ Example" << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        // Run example with XOR problem
        run_xor_example();
        
        // Run example with spiral classification
        run_spiral_example();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}