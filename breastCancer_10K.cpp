#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <thread>
#endif

// Sigmoid function for logistic regression
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Logistic regression function with batch gradient descent
std::vector<double> logistic_regression(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double learning_rate, int epochs) {
    size_t num_features = x[0].size();
    std::vector<double> weights(num_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> gradients(num_features, 0.0);

        for (size_t i = 0; i < x.size(); ++i) {
            double y_pred = 0.0;
            for (size_t j = 0; j < num_features; ++j) {
                y_pred += weights[j] * x[i][j];
            }
            y_pred = sigmoid(y_pred);
            double error = y_pred - y[i];

            for (size_t j = 0; j < num_features; ++j) {
                gradients[j] += error * x[i][j];
            }
        }

        for (size_t j = 0; j < num_features; ++j) {
            weights[j] -= learning_rate * gradients[j] / static_cast<double>(x.size());
        }
    }

    return weights;
}

// Function to calculate accuracy
double calculate_accuracy(const std::vector<std::vector<double>>& x, const std::vector<double>& y, const std::vector<double>& weights) {
    int correct = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double y_pred = 0.0;
        for (size_t j = 0; j < weights.size(); ++j) {
            y_pred += weights[j] * x[i][j];
        }
        y_pred = sigmoid(y_pred);
        if ((y_pred >= 0.5 && y[i] == 1.0) || (y_pred < 0.5 && y[i] == 0.0)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / x.size() * 100.0;
}

// Generate synthetic dataset
void generate_dataset(std::vector<std::vector<double>>& x, std::vector<double>& y, size_t num_samples, size_t num_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    x.resize(num_samples, std::vector<double>(num_features));
    y.resize(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            x[i][j] = dis(gen);
        }
        y[i] = (x[i][0] + x[i][1] > 50.0) ? 1.0 : 0.0;
    }
}

// Perform Student's t-test for precision control
bool check_precision(const std::vector<double>& times, double precision) {
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / times.size() - mean * mean);

    double margin_of_error = 1.96 * (stddev / std::sqrt(times.size()));
    return (margin_of_error / mean) <= precision;
}

int main() {
    size_t num_samples = 10000;  // Adjust this for different data sizes
    size_t num_features = 30;
    double learning_rate = 0.1;
    int epochs = 1000;
    double precision = 0.025;  // 2.5% precision

    // Generate dataset
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    generate_dataset(x, y, num_samples, num_features);

    // Split dataset into 80% training and 20% testing
    size_t train_size = static_cast<size_t>(0.8 * num_samples);
    std::vector<std::vector<double>> x_train(x.begin(), x.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> x_test(x.begin() + train_size, x.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    std::vector<double> execution_times;
    double total_accuracy = 0.0;

    do {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<double> weights = logistic_regression(x_train, y_train, learning_rate, epochs);
        auto duration = std::chrono::high_resolution_clock::now() - start_time;

        execution_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1e6);

        double accuracy = calculate_accuracy(x_test, y_test, weights);
        total_accuracy += accuracy;

#ifndef __EMSCRIPTEN__
        // Introduce delay to prevent cache effects (native only)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif

    } while (!check_precision(execution_times, precision));

    double mean_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
    double mean_accuracy = total_accuracy / execution_times.size();

    std::cout << "Mean execution time: " << mean_time << " seconds\n";
    std::cout << "Mean accuracy: " << mean_accuracy << " %\n";

    return 0;
}

