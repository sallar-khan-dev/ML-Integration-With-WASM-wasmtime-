use rand::distributions::{Distribution, Uniform};
use std::time::{Duration, Instant};

// Generate synthetic dataset
fn generate_data(num_samples: usize, dimensions: usize) -> Vec<Vec<f64>> {
    let mut data = vec![vec![0.0; dimensions]; num_samples];
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(0.0, 10.0);

    for i in 0..num_samples {
        for j in 0..dimensions {
            data[i][j] = uniform.sample(&mut rng);
        }
    }
    data
}

// Euclidean distance function
fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// K-means clustering
fn kmeans(
    x: &[Vec<f64>],  // Use a slice reference instead of Vec reference
    num_clusters: usize,
    max_iters: usize,
    assignments: &mut Vec<usize>,
) -> Vec<Vec<f64>> {
    let num_points = x.len();
    let mut centroids = x.iter().take(num_clusters).cloned().collect::<Vec<Vec<f64>>>();
    assignments.clear();
    assignments.resize(num_points, 0);

    for _ in 0..max_iters {
        for i in 0..num_points {
            let mut closest_centroid = 0;
            let mut min_distance = euclidean_distance(&centroids[0], &x[i]);

            for j in 1..num_clusters {
                let dist = euclidean_distance(&centroids[j], &x[i]);
                if dist < min_distance {
                    min_distance = dist;
                    closest_centroid = j;
                }
            }
            assignments[i] = closest_centroid;
        }

        let mut new_centroids = vec![vec![0.0; x[0].len()]; num_clusters];
        let mut counts = vec![0; num_clusters];

        for i in 0..num_points {
            for j in 0..x[i].len() {
                new_centroids[assignments[i]][j] += x[i][j];
            }
            counts[assignments[i]] += 1;
        }

        for i in 0..num_clusters {
            if counts[i] > 0 {
                for j in 0..new_centroids[i].len() {
                    new_centroids[i][j] /= counts[i] as f64;
                }
            }
        }

        centroids = new_centroids;
    }

    centroids
}

// Compute intra-cluster variance (accuracy measure)
fn compute_intra_cluster_variance(
    x: &[Vec<f64>],  // Use a slice reference instead of Vec reference
    centroids: &[Vec<f64>],  // Use a slice reference instead of Vec reference
    assignments: &[usize],  // Use a slice reference instead of Vec reference
) -> f64 {
    let variance: f64 = x
        .iter()
        .zip(assignments.iter())
        .map(|(point, &assignment)| euclidean_distance(&point, &centroids[assignment]))
        .sum();
    variance / x.len() as f64
}

// Convert variance to accuracy percentage
fn variance_to_accuracy(variance: f64, max_possible_variance: f64) -> f64 {
    100.0 * (1.0 - (variance / max_possible_variance))
}

// Confidence interval computation
fn compute_confidence_interval(times: &[f64], mean: &mut f64) -> f64 {
    let n = times.len();
    *mean = times.iter().copied().sum::<f64>() / n as f64;
    let variance = times
        .iter()
        .map(|&t| (t - *mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let stddev = variance.sqrt();
    let t_value = 2.045; // Approximate for 95% CI with large sample size
    t_value * (stddev / (n as f64).sqrt())
}

fn main() {
    let num_samples = 10000; // Adjust as needed
    let dimensions = 4;
    let num_clusters = 5;
    let max_iters = 100;

    let data = generate_data(num_samples, dimensions);

    // Split data into training (80%) and testing (20%)
    let train_size = (0.8 * num_samples as f64) as usize;
    let train_data = &data[0..train_size];
    let test_data = &data[train_size..];

    let mut execution_times = Vec::new();
    let mut mean_time = 0.0;
    let mut mean_variance = 0.0;

    loop {
        let mut assignments = Vec::new();
        let start_time = Instant::now();
        let centroids = kmeans(train_data, num_clusters, max_iters, &mut assignments);
        let duration = start_time.elapsed().as_secs_f64();
        execution_times.push(duration);

        let variance = compute_intra_cluster_variance(test_data, &centroids, &assignments);
        mean_variance += variance;

        // Sleep for 100ms to mitigate cache effects
        std::thread::sleep(Duration::from_millis(100));

        // Ensure the confidence interval is within the precision limit
        let confidence_interval = compute_confidence_interval(&execution_times, &mut mean_time);
        if confidence_interval / mean_time <= 0.025 {
            break;
        }
    }

    mean_variance /= execution_times.len() as f64; // Average variance over runs
    let accuracy_percentage = variance_to_accuracy(mean_variance, 100.0);

    println!("Mean Execution Time: {} seconds", mean_time);
    println!("Clustering Accuracy: {} %", accuracy_percentage);
}

