#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <functional>

namespace mlcpp {
namespace models {

// Available activation functions
enum class ActivationFunction {
    SIGMOID,
    RELU,
    TANH
};

class NeuralNetwork {
public:
    // Constructor with customizable network architecture
    // layers: vector of integers where each element represents the number of neurons in a layer
    // activation: activation function to use in hidden layers
    // learning_rate: learning rate for gradient descent
    // max_iterations: maximum number of iterations for training
    // batch_size: size of mini-batches for stochastic gradient descent (0 = full batch)
    // tolerance: convergence tolerance
    NeuralNetwork(
        const std::vector<int>& layers,
        ActivationFunction activation = ActivationFunction::SIGMOID,
        double learning_rate = 0.01,
        int max_iterations = 1000,
        int batch_size = 32,
        double tolerance = 1e-6
    );
    
    ~NeuralNetwork();
    
    // Train the neural network on the given data
    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y);
    
    // Binary classification convenience method
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    
    // Predict using the trained model (raw outputs)
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& X) const;
    
    // Convenience method for binary classification
    std::vector<double> predictBinary(const std::vector<std::vector<double>>& X, double threshold = 0.5) const;
    
    // Get loss history during training
    std::vector<double> getLossHistory() const;
    
    // Get weights for a specific layer
    std::vector<std::vector<double>> getLayerWeights(int layer) const;
    
    // Get biases for a specific layer
    std::vector<double> getLayerBiases(int layer) const;
    
    // Set hyperparameters
    void setLearningRate(double learning_rate);
    void setMaxIterations(int max_iterations);
    void setBatchSize(int batch_size);
    void setTolerance(double tolerance);
    
private:
    // Network architecture parameters
    std::vector<int> m_layers;                 // Number of neurons in each layer
    ActivationFunction m_activation;           // Activation function for hidden layers
    
    // Training hyperparameters
    double m_learning_rate;                    // Learning rate for gradient descent
    int m_max_iterations;                      // Maximum number of training iterations
    int m_batch_size;                          // Mini-batch size (0 = full batch)
    double m_tolerance;                        // Convergence tolerance
    
    // Network parameters
    std::vector<std::vector<std::vector<double>>> m_weights;  // Weights for each layer
    std::vector<std::vector<double>> m_biases;                // Biases for each layer
    
    // Training history
    std::vector<double> m_loss_history;                       // Loss history during training
    
    // Helper methods
    void initializeParameters();                // Initialize weights and biases
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& X) const;  // Forward propagation
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<double>>& X,
        const std::vector<std::vector<double>>& y,
        const std::vector<std::vector<double>>& activations
    );  // Backward propagation
    
    // Activation functions and their derivatives
    double activate(double x) const;
    double activateDerivative(double x) const;
    
    // Random number generator for weight initialization
    std::mt19937 m_rng;
};

} // namespace models
} // namespace mlcpp