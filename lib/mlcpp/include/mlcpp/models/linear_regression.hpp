#pragma once

#include <vector>
#include <utility>
#include <memory>

namespace mlcpp {
namespace models {

class LinearRegression {
public:
    LinearRegression(double learning_rate = 0.01, int max_iterations = 1000, double tolerance = 1e-6);
    ~LinearRegression();

    // Fit the model to training data using gradient descent
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    // Predict using the trained model
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    // Get model parameters
    std::vector<double> getCoefficients() const;
    double getIntercept() const;

    // Set hyperparameters
    void setLearningRate(double learning_rate);
    void setMaxIterations(int max_iterations);
    void setTolerance(double tolerance);

private:
    double m_learning_rate;
    int m_max_iterations;
    double m_tolerance;
    double m_intercept;                // b0 (bias term)
    std::vector<double> m_coefficients; // b1, b2, ..., bn
};

} // namespace models
} // namespace mlcpp