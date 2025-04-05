#include "mlcpp/models/linear_regression.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace mlcpp {
namespace models {

LinearRegression::LinearRegression(double learning_rate, int max_iterations, double tolerance)
    : m_learning_rate(learning_rate),
      m_max_iterations(max_iterations),
      m_tolerance(tolerance),
      m_intercept(0.0) {
}

LinearRegression::~LinearRegression() = default;

void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }

    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    
    // Initialize parameters
    m_intercept = 0.0;
    m_coefficients = std::vector<double>(n_features, 0.0);
    
    // Perform gradient descent
    for (int iter = 0; iter < m_max_iterations; ++iter) {
        double intercept_gradient = 0.0;
        std::vector<double> coefficients_gradient(n_features, 0.0);
        
        // Calculate gradients
        for (size_t i = 0; i < n_samples; ++i) {
            // Calculate prediction for this sample
            double prediction = m_intercept;
            for (size_t j = 0; j < n_features; ++j) {
                prediction += m_coefficients[j] * X[i][j];
            }
            
            // Calculate error
            double error = prediction - y[i];
            
            // Update intercept gradient
            intercept_gradient += error;
            
            // Update coefficient gradients
            for (size_t j = 0; j < n_features; ++j) {
                coefficients_gradient[j] += error * X[i][j];
            }
        }
        
        // Average gradients
        intercept_gradient /= n_samples;
        for (size_t j = 0; j < n_features; ++j) {
            coefficients_gradient[j] /= n_samples;
        }
        
        // Update parameters
        m_intercept -= m_learning_rate * intercept_gradient;
        for (size_t j = 0; j < n_features; ++j) {
            m_coefficients[j] -= m_learning_rate * coefficients_gradient[j];
        }
        
        // Check convergence
        double gradient_norm = std::abs(intercept_gradient);
        for (size_t j = 0; j < n_features; ++j) {
            gradient_norm += std::abs(coefficients_gradient[j]);
        }
        
        if (gradient_norm < m_tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const {
    if (X.empty()) {
        throw std::invalid_argument("Empty input for prediction");
    }
    
    if (!X[0].empty() && X[0].size() != m_coefficients.size()) {
        throw std::invalid_argument("Input features don't match model parameters");
    }
    
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    std::vector<double> predictions(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        double prediction = m_intercept;
        for (size_t j = 0; j < n_features; ++j) {
            prediction += m_coefficients[j] * X[i][j];
        }
        predictions[i] = prediction;
    }
    
    return predictions;
}

std::vector<double> LinearRegression::getCoefficients() const {
    return m_coefficients;
}

double LinearRegression::getIntercept() const {
    return m_intercept;
}

void LinearRegression::setLearningRate(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    m_learning_rate = learning_rate;
}

void LinearRegression::setMaxIterations(int max_iterations) {
    if (max_iterations <= 0) {
        throw std::invalid_argument("Max iterations must be positive");
    }
    m_max_iterations = max_iterations;
}

void LinearRegression::setTolerance(double tolerance) {
    if (tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    m_tolerance = tolerance;
}

} // namespace models
} // namespace mlcpp