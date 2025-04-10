#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <numeric>

// Generate synthetic dataset
std::vector<std::vector<double>> generate_data(size_t num_samples, size_t dimensions) {
    std::vector<std::vector<double>> data(num_samples, std::vector<double>(dimensions));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < dimensions; ++j) {
            data[i][j] = dis(gen);
        }
    }
    return data;
}

// Euclidean distance function
double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// K-means clustering
std::vector<std::vector<double>> kmeans(const std::vector<std::vector<double>>& x, size_t num_clusters, size_t max_iters, std::vector<int>& assignments) {
    size_t num_points = x.size();
    std::vector<std::vector<double>> centroids(x.begin(), x.begin() + num_clusters);
    assignments.resize(num_points);

    for (size_t iter = 0; iter < max_iters; ++iter) {
        for (size_t i = 0; i < num_points; ++i) {
            size_t closest_centroid = 0;
            double min_distance = euclidean_distance(centroids[0], x[i]);

            for (size_t j = 1; j < num_clusters; ++j) {
                double dist = euclidean_distance(centroids[j], x[i]);
                if (dist < min_distance) {
                    min_distance = dist;
                    closest_centroid = j;
                }
            }
            assignments[i] = closest_centroid;
        }

        std::vector<std::vector<double>> new_centroids(num_clusters, std::vector<double>(x[0].size(), 0.0));
        std::vector<int> counts(num_clusters, 0);

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < x[i].size(); ++j) {
                new_centroids[assignments[i]][j] += x[i][j];
            }
            counts[assignments[i]]++;
        }

        for (size_t i = 0; i < num_clusters; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < new_centroids[i].size(); ++j) {
                    new_centroids[i][j] /= counts[i];
                }
            }
        }

        centroids = std::move(new_centroids);
    }
    return centroids;
}

// Compute intra-cluster variance (accuracy measure)
double compute_intra_cluster_variance(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& centroids, const std::vector<int>& assignments) {
    double variance = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        variance += euclidean_distance(x[i], centroids[assignments[i]]);
    }
    return variance / x.size();
}

// Convert variance to accuracy percentage
double variance_to_accuracy(double variance, double max_possible_variance = 100.0) {
    return 100.0 * (1.0 - (variance / max_possible_variance));
}

// Confidence interval computation
double compute_confidence_interval(const std::vector<double>& times, double& mean) {
    size_t n = times.size();
    mean = std::accumulate(times.begin(), times.end(), 0.0) / n;
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    variance /= n;
    double stddev = std::sqrt(variance);
    double t_value = 2.045; // Approximate for 95% CI with large sample size
    return t_value * (stddev / std::sqrt(n));
}

int main() {
    size_t num_samples = 10000; // Adjust as needed
    size_t dimensions = 4;
    size_t num_clusters = 5;
    size_t max_iters = 100;

    std::vector<std::vector<double>> data = generate_data(num_samples, dimensions);

    // Split data into training (80%) and testing (20%)
    size_t train_size = (size_t)(0.8 * num_samples);
    std::vector<std::vector<double>> train_data(data.begin(), data.begin() + train_size);
    std::vector<std::vector<double>> test_data(data.begin() + train_size, data.end());

    std::vector<double> execution_times;
    double mean_time;
    double mean_variance = 0.0;

    do {
        std::vector<int> assignments;
        auto start_time = std::chrono::steady_clock::now();
        auto centroids = kmeans(train_data, num_clusters, max_iters, assignments);
        auto end_time = std::chrono::steady_clock::now();

        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0; // Convert ms to seconds
        execution_times.push_back(duration);

        double variance = compute_intra_cluster_variance(test_data, centroids, assignments);
        mean_variance += variance;

        // Ensure time elapses to mitigate cache effects
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (compute_confidence_interval(execution_times, mean_time) / mean_time > 0.025); // Precision within 2.5%

    mean_variance /= execution_times.size(); // Average variance over runs
    double accuracy_percentage = variance_to_accuracy(mean_variance);

    std::cout << "Mean Execution Time: " << mean_time << " seconds" << std::endl;
    std::cout << "Clustering Accuracy: " << accuracy_percentage << " %" << std::endl;

    return 0;
}
