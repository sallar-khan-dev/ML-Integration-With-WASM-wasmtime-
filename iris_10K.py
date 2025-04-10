import random
import math
import time
from collections import defaultdict

def run_kmeans(data, num_clusters, max_iters):
    if not data or num_clusters == 0:
        return []

    centroids = initialize_centroids(data, num_clusters)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        centroids = update_centroids(data, clusters, num_clusters)

    return centroids

def initialize_centroids(data, k):
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = defaultdict(list)
    for i, point in enumerate(data):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        centroid_idx = distances.index(min(distances))
        clusters[centroid_idx].append(i)
    return clusters

def update_centroids(data, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = clusters.get(i, [])
        if cluster_points:
            cluster_data = [data[idx] for idx in cluster_points]
            centroids.append(mean_point(cluster_data))
    return centroids

def mean_point(cluster_data):
    dimension = len(cluster_data[0])
    return [sum(point[dim] for point in cluster_data) / len(cluster_data) for dim in range(dimension)]

def euclidean_distance(point_a, point_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))

def calculate_confidence_interval(data, confidence=0.95):
    mean = sum(data) / len(data)
    n = len(data)
    stddev = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
    error_margin = 1.96 * (stddev / math.sqrt(n))
    return mean, error_margin

def is_within_precision(mean, error_margin, precision=0.025):
    return (error_margin / mean) <= precision

# Dataset size and split
dataset_size = 10000 
data = [[random.random(), random.random()] for _ in range(dataset_size)]
split_ratio = 0.8
split_index = int(split_ratio * dataset_size)
train_data = data[:split_index]  # 80% Training
test_data = data[split_index:]  # 20% Testing

# Max inertia for normalization
max_inertia = sum(euclidean_distance(point, [0.5, 0.5]) ** 2 for point in data)

execution_times = []
inertia_list = []

while True:
    start_time = time.time()
    
    # Train the model on training data
    centroids = run_kmeans(train_data, 3, 10)
    
    # Evaluate the model on test data
    clusters = assign_clusters(test_data, centroids)
    inertia = 0
    for cluster_idx, points in clusters.items():
        centroid = centroids[cluster_idx]
        inertia += sum(euclidean_distance(test_data[i], centroid) ** 2 for i in points)
    
    execution_time = time.time() - start_time  # Measure only ML implementation time
    execution_times.append(execution_time)
    inertia_list.append(inertia)

    # Check statistical precision
    mean_execution, error_margin_execution = calculate_confidence_interval(execution_times)
    mean_inertia, error_margin_inertia = calculate_confidence_interval(inertia_list)

    if is_within_precision(mean_execution, error_margin_execution) and is_within_precision(mean_inertia, error_margin_inertia):
        break

accuracy = 100 * (1 - (mean_inertia / max_inertia))
result = {
    "execution_time": mean_execution,
    "accuracy": accuracy
}

print(result)

