#include "mlcpp/models/neural_network.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

namespace mlcpp {
namespace models {

// Constructor
NeuralNetwork::NeuralNetwork(
    const std::vector<int>& layers,
    ActivationFunction activation,
    double learning_rate,
    int max_iterations,
    int batch_size,
    double tolerance
) : m_layers(layers),
    m_activation(activation),
    m_learning_rate(learning_rate),
    m_max_iterations(max_iterations),
    m_batch_size(batch_size),
    m_tolerance(tolerance),
    m_rng(std::chrono::system_clock::now().time_since_epoch().count()) {
    
    // Validate network architecture
    if (layers.size() < 2) {
        throw std::invalid_argument("Neural network must have at least input and output layers");
    }
    
    for (int layer_size : layers) {
        if (layer_size <= 0) {
            throw std::invalid_argument("Each layer must have at least one neuron");
        }
    }
    
    // Validate hyperparameters
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    
    if (max_iterations <= 0) {
        throw std::invalid_argument("Max iterations must be positive");
    }
    
    if (batch_size < 0) {
        throw std::invalid_argument("Batch size must be non-negative");
    }
    
    if (tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    
    // Initialize network parameters
    initializeParameters();
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::initializeParameters() {
    m_weights.clear();
    m_biases.clear();
    
    // Xavier/Glorot initialization for weights
    // For each layer (except the input layer)
    for (size_t l = 1; l < m_layers.size(); ++l) {
        int fan_in = m_layers[l-1];
        int fan_out = m_layers[l];
        
        // Initialize weights for this layer
        double weight_scale = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<double> weight_distribution(0.0, weight_scale);
        
        std::vector<std::vector<double>> layer_weights;
        for (int j = 0; j < fan_out; ++j) {
            std::vector<double> neuron_weights;
            for (int i = 0; i < fan_in; ++i) {
                neuron_weights.push_back(weight_distribution(m_rng));
            }
            layer_weights.push_back(neuron_weights);
        }
        m_weights.push_back(layer_weights);
        
        // Initialize biases for this layer
        std::vector<double> layer_biases(fan_out, 0.0);
        m_biases.push_back(layer_biases);
    }
}

// Activation function
double NeuralNetwork::activate(double x) const {
    switch (m_activation) {
        case ActivationFunction::SIGMOID:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationFunction::RELU:
            return std::max(0.0, x);
        case ActivationFunction::TANH:
            return std::tanh(x);
        default:
            throw std::invalid_argument("Unsupported activation function");
    }
}

// Derivative of activation function
double NeuralNetwork::activateDerivative(double x) const {
    switch (m_activation) {
        case ActivationFunction::SIGMOID: {
            double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            return sigmoid_x * (1.0 - sigmoid_x);
        }
        case ActivationFunction::RELU:
            return x > 0.0 ? 1.0 : 0.0;
        case ActivationFunction::TANH: {
            double tanh_x = std::tanh(x);
            return 1.0 - tanh_x * tanh_x;
        }
        default:
            throw std::invalid_argument("Unsupported activation function");
    }
}

// Forward pass through the network
std::vector<std::vector<double>> NeuralNetwork::forward(const std::vector<std::vector<double>>& X) const {
    if (X.empty()) {
        throw std::invalid_argument("Empty input for forward propagation");
    }
    
    if (X[0].size() != static_cast<size_t>(m_layers[0])) {
        throw std::invalid_argument("Input features don't match input layer size");
    }
    
    const size_t batch_size = X.size();
    std::vector<std::vector<double>> activations = X;  // Start with input layer activations
    
    // For each layer (except input)
    for (size_t l = 0; l < m_weights.size(); ++l) {
        std::vector<std::vector<double>> layer_activations(batch_size, std::vector<double>(m_layers[l+1], 0.0));
        
        // For each sample in batch
        for (size_t i = 0; i < batch_size; ++i) {
            // For each neuron in the current layer
            for (size_t j = 0; j < m_weights[l].size(); ++j) {
                double z = m_biases[l][j];  // Start with bias
                
                // Add weighted sum of previous activations
                for (size_t k = 0; k < m_weights[l][j].size(); ++k) {
                    z += m_weights[l][j][k] * activations[i][k];
                }
                
                // Apply activation function (sigmoid for output layer, specified activation for hidden)
                if (l == m_weights.size() - 1) {
                    // Output layer uses sigmoid for binary classification or identity for regression
                    layer_activations[i][j] = 1.0 / (1.0 + std::exp(-z));  // Sigmoid for output
                } else {
                    layer_activations[i][j] = activate(z);
                }
            }
        }
        
        activations = std::move(layer_activations);
    }
    
    return activations;
}

// Backward pass for computing gradients
std::vector<std::vector<std::vector<double>>> NeuralNetwork::backward(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& y,
    const std::vector<std::vector<double>>& predictions
) {
    const size_t batch_size = X.size();
    const size_t num_layers = m_weights.size();
    
    // Store activations for all layers
    std::vector<std::vector<std::vector<double>>> activations(num_layers + 1);
    activations[0] = X;  // Input layer
    
    // Forward pass to compute all activations
    std::vector<std::vector<double>> current_activations = X;
    for (size_t l = 0; l < num_layers; ++l) {
        std::vector<std::vector<double>> next_activations(batch_size, std::vector<double>(m_layers[l+1], 0.0));
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < m_weights[l].size(); ++j) {
                double z = m_biases[l][j];
                
                for (size_t k = 0; k < m_weights[l][j].size(); ++k) {
                    z += m_weights[l][j][k] * current_activations[i][k];
                }
                
                if (l == num_layers - 1) {
                    next_activations[i][j] = 1.0 / (1.0 + std::exp(-z));  // Sigmoid for output
                } else {
                    next_activations[i][j] = activate(z);
                }
            }
        }
        
        activations[l+1] = next_activations;
        current_activations = next_activations;
    }
    
    // Calculate error at the output layer
    std::vector<std::vector<double>> deltas(num_layers);
    deltas[num_layers-1].resize(batch_size * m_layers[num_layers]);
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < m_layers[num_layers]; ++j) {
            double error = predictions[i][j] - y[i][j];
            double derivative = predictions[i][j] * (1.0 - predictions[i][j]);  // Sigmoid derivative
            deltas[num_layers-1][i * m_layers[num_layers] + j] = error * derivative;
        }
    }
    
    // Backpropagate the error
    for (size_t l = num_layers - 2; l < num_layers; --l) {
        deltas[l].resize(batch_size * m_layers[l+1]);
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < m_layers[l+1]; ++j) {
                double error = 0.0;
                
                for (size_t k = 0; k < m_layers[l+2]; ++k) {
                    error += deltas[l+1][i * m_layers[l+2] + k] * m_weights[l+1][k][j];
                }
                
                double z = 0.0;
                for (size_t k = 0; k < m_weights[l][j].size(); ++k) {
                    z += m_weights[l][j][k] * activations[l][i][k];
                }
                z += m_biases[l][j];
                
                double derivative = (l > 0) ? activateDerivative(z) : activations[l+1][i][j] * (1.0 - activations[l+1][i][j]);
                deltas[l][i * m_layers[l+1] + j] = error * derivative;
            }
        }
    }
    
    // Calculate gradients
    std::vector<std::vector<std::vector<double>>> gradients(num_layers);
    
    for (size_t l = 0; l < num_layers; ++l) {
        gradients[l].resize(m_layers[l+1]);
        
        for (size_t j = 0; j < m_layers[l+1]; ++j) {
            gradients[l][j].resize(m_layers[l] + 1);  // +1 for bias
            
            for (size_t k = 0; k < m_layers[l]; ++k) {
                double gradient = 0.0;
                
                for (size_t i = 0; i < batch_size; ++i) {
                    gradient += deltas[l][i * m_layers[l+1] + j] * activations[l][i][k];
                }
                
                gradients[l][j][k] = gradient / batch_size;
            }
            
            // Bias gradient
            double bias_gradient = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                bias_gradient += deltas[l][i * m_layers[l+1] + j];
            }
            
            gradients[l][j][m_layers[l]] = bias_gradient / batch_size;
        }
    }
    
    return gradients;
}

// Train on data with multiple output values
void NeuralNetwork::fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    const size_t n_outputs = y[0].size();
    
    if (n_features != static_cast<size_t>(m_layers[0])) {
        throw std::invalid_argument("Input features don't match input layer size");
    }
    
    if (n_outputs != static_cast<size_t>(m_layers.back())) {
        throw std::invalid_argument("Output dimensions don't match output layer size");
    }
    
    // Reset loss history
    m_loss_history.clear();
    
    // Use either full batch or mini-batch
    int effective_batch_size = (m_batch_size == 0) ? n_samples : m_batch_size;
    
    // Training iterations
    for (int iter = 0; iter < m_max_iterations; ++iter) {
        double total_loss = 0.0;
        
        // Shuffle indices for stochastic training
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), m_rng);
        
        // Process mini-batches
        for (size_t batch_start = 0; batch_start < n_samples; batch_start += effective_batch_size) {
            size_t batch_end = std::min(batch_start + effective_batch_size, n_samples);
            size_t current_batch_size = batch_end - batch_start;
            
            // Create current batch
            std::vector<std::vector<double>> X_batch(current_batch_size);
            std::vector<std::vector<double>> y_batch(current_batch_size);
            
            for (size_t i = 0; i < current_batch_size; ++i) {
                X_batch[i] = X[indices[batch_start + i]];
                y_batch[i] = y[indices[batch_start + i]];
            }
            
            // Forward pass
            std::vector<std::vector<double>> predictions = forward(X_batch);
            
            // Compute loss (mean squared error)
            double batch_loss = 0.0;
            for (size_t i = 0; i < current_batch_size; ++i) {
                for (size_t j = 0; j < n_outputs; ++j) {
                    double error = predictions[i][j] - y_batch[i][j];
                    batch_loss += error * error;
                }
            }
            batch_loss /= (current_batch_size * n_outputs);
            total_loss += batch_loss * current_batch_size / n_samples;
            
            // Backward pass
            auto gradients = backward(X_batch, y_batch, predictions);
            
            // Update weights and biases
            for (size_t l = 0; l < m_weights.size(); ++l) {
                for (size_t j = 0; j < m_weights[l].size(); ++j) {
                    for (size_t k = 0; k < m_weights[l][j].size(); ++k) {
                        m_weights[l][j][k] -= m_learning_rate * gradients[l][j][k];
                    }
                    m_biases[l][j] -= m_learning_rate * gradients[l][j][m_weights[l][j].size()];
                }
            }
        }
        
        // Record loss
        m_loss_history.push_back(total_loss);
        
        // Check for convergence
        if (iter > 0 && std::abs(m_loss_history[iter] - m_loss_history[iter-1]) < m_tolerance) {
            std::cout << "Converged after " << (iter + 1) << " iterations." << std::endl;
            break;
        }
        
        // Print progress
        if (iter % 100 == 0 || iter == m_max_iterations - 1) {
            std::cout << "Iteration " << iter << ", Loss: " << total_loss << std::endl;
        }
    }
}

// Convenience method for binary classification
void NeuralNetwork::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    // Convert y to one-hot encoding
    std::vector<std::vector<double>> y_encoded(y.size(), std::vector<double>(1, 0.0));
    for (size_t i = 0; i < y.size(); ++i) {
        y_encoded[i][0] = y[i];
    }
    
    fit(X, y_encoded);
}

// Predict outputs for new data
std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& X) const {
    return forward(X);
}

// Convenience method for binary classification
std::vector<double> NeuralNetwork::predictBinary(const std::vector<std::vector<double>>& X, double threshold) const {
    std::vector<std::vector<double>> raw_predictions = predict(X);
    std::vector<double> binary_predictions(raw_predictions.size());
    
    for (size_t i = 0; i < raw_predictions.size(); ++i) {
        binary_predictions[i] = raw_predictions[i][0] >= threshold ? 1.0 : 0.0;
    }
    
    return binary_predictions;
}

std::vector<double> NeuralNetwork::getLossHistory() const {
    return m_loss_history;
}

std::vector<std::vector<double>> NeuralNetwork::getLayerWeights(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= m_weights.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    return m_weights[layer];
}

std::vector<double> NeuralNetwork::getLayerBiases(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= m_biases.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    return m_biases[layer];
}

void NeuralNetwork::setLearningRate(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    m_learning_rate = learning_rate;
}

void NeuralNetwork::setMaxIterations(int max_iterations) {
    if (max_iterations <= 0) {
        throw std::invalid_argument("Max iterations must be positive");
    }
    m_max_iterations = max_iterations;
}

void NeuralNetwork::setBatchSize(int batch_size) {
    if (batch_size < 0) {
        throw std::invalid_argument("Batch size must be non-negative");
    }
    m_batch_size = batch_size;
}

void NeuralNetwork::setTolerance(double tolerance) {
    if (tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    m_tolerance = tolerance;
}

} // namespace models
} // namespace mlcpp