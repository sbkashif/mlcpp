/**
 * MLCPP Linear Regression Example
 * 
 * This example demonstrates how to use the MLCPP library for linear regression
 * in a C++ application. It covers data preparation, model training, evaluation,
 * and prediction with both synthetic and real-world-like data.
 */

#include "mlcpp/models/linear_regression.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>

// Helper function to calculate Mean Squared Error
double calculate_mse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double error = y_true[i] - y_pred[i];
        sum_squared_error += error * error;
    }
    
    return sum_squared_error / y_true.size();
}

// Helper function to calculate R-squared
double calculate_r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Calculate mean of y_true
    double y_mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
    
    // Calculate total sum of squares
    double ss_total = 0.0;
    for (const auto& y : y_true) {
        ss_total += (y - y_mean) * (y - y_mean);
    }
    
    // Calculate residual sum of squares
    double ss_residual = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_residual += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    
    return 1.0 - (ss_residual / ss_total);
}

// Helper function to generate synthetic data
void generate_synthetic_data(
    std::vector<std::vector<double>>& X, 
    std::vector<double>& y,
    size_t n_samples,
    size_t n_features,
    const std::vector<double>& coefficients,
    double intercept,
    double noise_level = 0.5
) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);  // For feature values
    std::normal_distribution<> noise(0.0, noise_level); // For noise
    
    X.resize(n_samples, std::vector<double>(n_features));
    y.resize(n_samples);
    
    // Generate random feature values and calculate target
    for (size_t i = 0; i < n_samples; ++i) {
        double target = intercept;
        for (size_t j = 0; j < n_features; ++j) {
            X[i][j] = dis(gen);
            target += X[i][j] * coefficients[j];
        }
        y[i] = target + noise(gen);
    }
}

// Example with synthetic data
void run_synthetic_example() {
    std::cout << "\n==== Linear Regression with Synthetic Data ====" << std::endl;
    
    // Create synthetic data
    const size_t n_samples = 1000;
    const size_t n_features = 3;
    std::vector<double> true_coefficients = {2.5, -1.3, 0.5};
    double true_intercept = 4.2;
    
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    generate_synthetic_data(X, y, n_samples, n_features, true_coefficients, true_intercept);
    
    std::cout << "Generated " << n_samples << " samples with " << n_features << " features" << std::endl;
    std::cout << "True model: y = " << true_intercept << " + " 
              << true_coefficients[0] << "*x1 + "
              << true_coefficients[1] << "*x2 + "
              << true_coefficients[2] << "*x3" << std::endl;
    
    // Split data into training and test sets (80/20 split)
    const size_t train_size = 0.8 * n_samples;
    
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());
    
    std::cout << "Training set size: " << X_train.size() << std::endl;
    std::cout << "Test set size: " << X_test.size() << std::endl;
    
    // Create and train the model
    std::cout << "Training model..." << std::endl;
    mlcpp::models::LinearRegression model(0.01, 10000, 1e-6);
    
    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X_train, y_train);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Get model parameters
    double intercept = model.getIntercept();
    std::vector<double> coefficients = model.getCoefficients();
    
    std::cout << "Learned model: y = " << intercept << " + ";
    for (size_t i = 0; i < coefficients.size(); ++i) {
        std::cout << coefficients[i] << "*x" << (i+1);
        if (i < coefficients.size() - 1) {
            std::cout << " + ";
        }
    }
    std::cout << std::endl;
    
    // Make predictions
    std::cout << "Making predictions on test data..." << std::endl;
    std::vector<double> y_pred = model.predict(X_test);
    
    // Evaluate model performance
    double mse = calculate_mse(y_test, y_pred);
    double r2 = calculate_r2_score(y_test, y_pred);
    
    std::cout << "Model evaluation:" << std::endl;
    std::cout << "  Mean Squared Error: " << mse << std::endl;
    std::cout << "  R-squared: " << r2 << std::endl;
    
    // Display some predictions
    std::cout << "\nSample predictions:" << std::endl;
    std::cout << std::setw(15) << "Actual" << std::setw(15) << "Predicted" << std::setw(15) << "Error" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    size_t num_samples = std::min(size_t(5), y_test.size());
    for (size_t i = 0; i < num_samples; ++i) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(15) << y_test[i] << std::setw(15) << y_pred[i] << std::setw(15) << (y_test[i] - y_pred[i]) << std::endl;
    }
}

// Example with realistic dataset (home price prediction)
void run_home_price_example() {
    std::cout << "\n==== Home Price Prediction Example ====" << std::endl;
    
    // Create synthetic housing data
    // Features: [size_sqft, bedrooms, bathrooms, age_years]
    std::vector<std::vector<double>> housing_data = {
        {1200, 2, 1, 50},   // Small older house
        {1500, 3, 2, 40},
        {1800, 3, 2, 30},
        {2200, 4, 2.5, 20},
        {2500, 4, 3, 10},   // Large newer house
        {1300, 3, 1, 45},
        {1600, 3, 1.5, 35},
        {1900, 3, 2, 25},
        {2300, 4, 2.5, 15},
        {2600, 5, 3, 5},    // Very large new house
        {1250, 2, 1, 48},
        {1550, 3, 1.5, 38},
        {1850, 3, 2, 28},
        {2250, 4, 2.5, 18},
        {2550, 4, 3, 8},
        {1350, 2, 1, 42},
        {1650, 3, 2, 32},
        {1950, 3, 2, 22},
        {2350, 4, 2.5, 12},
        {2650, 5, 3.5, 2}   // Newest house
    };
    
    // Home prices in thousands (e.g., 250 = $250,000)
    std::vector<double> home_prices = {
        150, 180, 210, 250, 320,
        155, 185, 215, 260, 340,
        153, 182, 212, 255, 325,
        158, 188, 218, 265, 350
    };
    
    // Split data for training and testing (80/20 split)
    const size_t train_size = 0.8 * housing_data.size();
    
    std::vector<std::vector<double>> X_train(housing_data.begin(), housing_data.begin() + train_size);
    std::vector<double> y_train(home_prices.begin(), home_prices.begin() + train_size);
    
    std::vector<std::vector<double>> X_test(housing_data.begin() + train_size, housing_data.end());
    std::vector<double> y_test(home_prices.begin() + train_size, home_prices.end());
    
    std::cout << "Training on " << X_train.size() << " homes, testing on " << X_test.size() << " homes" << std::endl;
    std::cout << "Features: [size_sqft, bedrooms, bathrooms, age_years]" << std::endl;
    
    // Train the model
    std::cout << "Training model..." << std::endl;
    mlcpp::models::LinearRegression model(0.0001, 20000, 1e-8);
    
    auto start = std::chrono::high_resolution_clock::now();
    model.fit(X_train, y_train);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Get model parameters
    double intercept = model.getIntercept();
    std::vector<double> coefficients = model.getCoefficients();
    
    std::cout << "\nPrice prediction formula:" << std::endl;
    std::cout << "Price = $" << std::fixed << std::setprecision(2) << intercept * 1000;
    std::cout << " + " << coefficients[0] * 1000 << " * size_sqft";
    std::cout << " + " << coefficients[1] * 1000 << " * bedrooms";
    std::cout << " + " << coefficients[2] * 1000 << " * bathrooms";
    std::cout << " + " << coefficients[3] * 1000 << " * age_years" << std::endl;
    
    // Make predictions
    std::vector<double> y_pred = model.predict(X_test);
    
    // Evaluate model performance
    double mse = calculate_mse(y_test, y_pred);
    double rmse = std::sqrt(mse); // Root mean squared error
    double r2 = calculate_r2_score(y_test, y_pred);
    
    std::cout << "\nModel evaluation:" << std::endl;
    std::cout << "  Mean Squared Error: " << mse << " ($" << std::fixed << std::setprecision(2) << std::sqrt(mse) * 1000 << ")" << std::endl;
    std::cout << "  Root Mean Squared Error: " << rmse << " ($" << rmse * 1000 << ")" << std::endl;
    std::cout << "  R-squared: " << r2 << std::endl;
    
    // Display predictions
    std::cout << "\nPredictions for test homes:" << std::endl;
    std::cout << std::setw(15) << "Features" << std::setw(15) << "Actual Price" << std::setw(15) << "Predicted" << std::setw(15) << "Error" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t i = 0; i < X_test.size(); ++i) {
        std::cout << "[" << X_test[i][0] << " sqft, " 
                  << X_test[i][1] << "bd, " 
                  << X_test[i][2] << "ba, " 
                  << X_test[i][3] << "yr]";
                  
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(15) << "$" << y_test[i] * 1000;
        std::cout << std::setw(15) << "$" << y_pred[i] * 1000;
        std::cout << std::setw(15) << "$" << (y_test[i] - y_pred[i]) * 1000 << std::endl;
    }
    
    // Predict a new house price
    std::vector<std::vector<double>> new_house = {
        {2100, 3, 2.5, 15}  // 2100 sqft, 3 bed, 2.5 bath, 15 years old
    };
    
    std::vector<double> predicted_price = model.predict(new_house);
    
    std::cout << "\nPrediction for a new house:" << std::endl;
    std::cout << "  House details: 2100 sqft, 3 bedrooms, 2.5 bathrooms, 15 years old" << std::endl;
    std::cout << "  Predicted price: $" << predicted_price[0] * 1000 << std::endl;
}

int main() {
    std::cout << "MLCPP Linear Regression C++ Example" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        // Run example with synthetic data
        run_synthetic_example();
        
        // Run example with realistic home price data
        run_home_price_example();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}