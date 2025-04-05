#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mlcpp/models/linear_regression.hpp"
#include "mlcpp/models/neural_network.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pymlcpp, m) {
    m.doc() = "ML-CPP: Machine Learning library with C++ backend";

    py::class_<mlcpp::models::LinearRegression>(m, "LinearRegression")
        .def(py::init<double, int, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iterations") = 1000,
             py::arg("tolerance") = 1e-6)
        .def("fit", &mlcpp::models::LinearRegression::fit,
             "Fit the model using gradient descent",
             py::arg("X"), py::arg("y"))
        .def("predict", &mlcpp::models::LinearRegression::predict,
             "Predict using the trained model", py::arg("X"))
        .def("get_coefficients", &mlcpp::models::LinearRegression::getCoefficients,
             "Get the model coefficients")
        .def("get_intercept", &mlcpp::models::LinearRegression::getIntercept,
             "Get the model intercept")
        .def("set_learning_rate", &mlcpp::models::LinearRegression::setLearningRate,
             "Set the learning rate", py::arg("learning_rate"))
        .def("set_max_iterations", &mlcpp::models::LinearRegression::setMaxIterations,
             "Set the maximum iterations", py::arg("max_iterations"))
        .def("set_tolerance", &mlcpp::models::LinearRegression::setTolerance,
             "Set the convergence tolerance", py::arg("tolerance"));

    // Enum for activation functions
    py::enum_<mlcpp::models::ActivationFunction>(m, "ActivationFunction")
        .value("SIGMOID", mlcpp::models::ActivationFunction::SIGMOID)
        .value("RELU", mlcpp::models::ActivationFunction::RELU)
        .value("TANH", mlcpp::models::ActivationFunction::TANH)
        .export_values();

    // Neural Network class binding
    py::class_<mlcpp::models::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int>&, mlcpp::models::ActivationFunction, double, int, int, double>(),
             py::arg("layers"),
             py::arg("activation") = mlcpp::models::ActivationFunction::SIGMOID,
             py::arg("learning_rate") = 0.01,
             py::arg("max_iterations") = 1000,
             py::arg("batch_size") = 32,
             py::arg("tolerance") = 1e-6)
        .def("fit", static_cast<void (mlcpp::models::NeuralNetwork::*)(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&)>(&mlcpp::models::NeuralNetwork::fit),
             "Train the neural network with multi-dimensional output",
             py::arg("X"), py::arg("y"))
        .def("fit", static_cast<void (mlcpp::models::NeuralNetwork::*)(const std::vector<std::vector<double>>&, const std::vector<double>&)>(&mlcpp::models::NeuralNetwork::fit),
             "Train the neural network for binary classification",
             py::arg("X"), py::arg("y"))
        .def("predict", &mlcpp::models::NeuralNetwork::predict,
             "Get raw predictions from the neural network", py::arg("X"))
        .def("predict_binary", &mlcpp::models::NeuralNetwork::predictBinary,
             "Get binary predictions for classification", py::arg("X"), py::arg("threshold") = 0.5)
        .def("get_loss_history", &mlcpp::models::NeuralNetwork::getLossHistory,
             "Get the loss history during training")
        .def("get_layer_weights", &mlcpp::models::NeuralNetwork::getLayerWeights,
             "Get the weights for a specific layer", py::arg("layer"))
        .def("get_layer_biases", &mlcpp::models::NeuralNetwork::getLayerBiases,
             "Get the biases for a specific layer", py::arg("layer"))
        .def("set_learning_rate", &mlcpp::models::NeuralNetwork::setLearningRate,
             "Set the learning rate", py::arg("learning_rate"))
        .def("set_max_iterations", &mlcpp::models::NeuralNetwork::setMaxIterations,
             "Set the maximum number of iterations", py::arg("max_iterations"))
        .def("set_batch_size", &mlcpp::models::NeuralNetwork::setBatchSize,
             "Set the batch size", py::arg("batch_size"))
        .def("set_tolerance", &mlcpp::models::NeuralNetwork::setTolerance,
             "Set the convergence tolerance", py::arg("tolerance"));
}