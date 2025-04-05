#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mlcpp/models/linear_regression.hpp"

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
}