import math
import random
import time

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Mini-Batch Gradient Descent for Faster Training
def logistic_regression(x, y, learning_rate, epochs, batch_size=500):
    num_samples = len(x)
    num_features = len(x[0])
    weights = [0.0] * num_features

    for _ in range(epochs):
        for i in range(0, num_samples, batch_size):  # Process in chunks
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            gradients = [0.0] * num_features
            batch_size_actual = len(x_batch)

            for j in range(batch_size_actual):
                y_pred = sigmoid(sum(weights[k] * x_batch[j][k] for k in range(num_features)))
                error = y_pred - y_batch[j]
                for k in range(num_features):
                    gradients[k] += error * x_batch[j][k]

            # Update weights for the batch
            weights = [weights[k] - (learning_rate * gradients[k] / batch_size_actual) for k in range(num_features)]

    return weights

# Optimized synthetic dataset generation
def generate_data(num_samples=10000, num_features=30):
    return (
        [[random.uniform(0, 1) for _ in range(num_features)] for _ in range(num_samples)],
        [1.0 if sum(sample) > (num_features * 0.5) else 0.0 for sample in [[random.uniform(0, 1) for _ in range(num_features)] for _ in range(num_samples)]]
    )

# Measure execution time and accuracy with statistical precision control
def benchmark(num_samples=10000, num_features=30):
    learning_rate = 0.01
    epochs = 200  # Keep 200 for now
    batch_size = 500  # Mini-Batch Size

    execution_times = []
    accuracies = []

    # Statistical precision control (95% CI with 2.5% precision)
    precision_target = 0.025  # 2.5% precision
    confidence_level = 0.95
    sample_size = 30
    mean_time = 0

    while True:
        for run in range(sample_size):
            x, y = generate_data(num_samples, num_features)
            start_time = time.time()
            weights = logistic_regression(x, y, learning_rate, epochs, batch_size)
            exec_time = time.time() - start_time
            execution_times.append(exec_time)

            # Compute accuracy efficiently
            correct_predictions = sum(
                (sigmoid(sum(weights[j] * x[i][j] for j in range(num_features))) > 0.5) == y[i]
                for i in range(num_samples)
            )
            accuracy = correct_predictions / num_samples
            accuracies.append(accuracy)

        # Calculate mean and standard deviation
        mean_time = sum(execution_times) / len(execution_times)
        std_dev = (sum((x - mean_time) ** 2 for x in execution_times) / len(execution_times)) ** 0.5
        
        # Calculate margin of error for 95% CI using a simplified method (Student's t-distribution assumption)
        margin_of_error = 1.96 * (std_dev / (len(execution_times) ** 0.5))

        # Check if the margin of error is within the target precision
        if margin_of_error / mean_time <= precision_target:
            break

    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_time, mean_accuracy

if __name__ == "__main__":
    num_samples = 10000  # Default value
    mean_time, mean_accuracy = benchmark(num_samples)
    print(f"Mean Execution Time: {mean_time:.4f} seconds")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")

