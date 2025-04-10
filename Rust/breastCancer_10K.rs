use rand::distributions::{Distribution, Uniform};
use std::f64;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logistic_regression(
    x: &[Vec<f64>],
    y: &[f64],
    learning_rate: f64,
    epochs: usize,
) -> Vec<f64> {
    let num_features = x[0].len();
    let mut weights = vec![0.0; num_features];

    for _ in 0..epochs {
        let mut gradients = vec![0.0; num_features];

        for i in 0..x.len() {
            let y_pred = sigmoid(x[i].iter().zip(weights.iter()).map(|(xi, w)| xi * w).sum());
            let error = y_pred - y[i];

            for j in 0..num_features {
                gradients[j] += error * x[i][j];
            }
        }

        for j in 0..num_features {
            weights[j] -= learning_rate * gradients[j] / x.len() as f64;
        }
    }

    weights
}

fn calculate_accuracy(
    x: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
) -> f64 {
    let mut correct = 0;

    for i in 0..x.len() {
        let y_pred = sigmoid(x[i].iter().zip(weights.iter()).map(|(xi, w)| xi * w).sum());
        let predicted = if y_pred >= 0.5 { 1.0 } else { 0.0 };

        if (predicted - y[i]).abs() < f64::EPSILON {
            correct += 1;
        }
    }

    correct as f64 / x.len() as f64 * 100.0
}

fn generate_dataset(
    x: &mut Vec<Vec<f64>>,
    y: &mut Vec<f64>,
    num_samples: usize,
    num_features: usize,
) {
    let mut rng = rand::thread_rng();
    let range = Uniform::from(0.0..100.0);

    for _ in 0..num_samples {
        let mut sample = vec![0.0; num_features];
        for j in 0..num_features {
            sample[j] = range.sample(&mut rng);
        }
        x.push(sample.clone());
        y.push(if sample[0] + sample[1] > 50.0 { 1.0 } else { 0.0 });
    }
}

fn main() {
    let num_samples = 10_000;
    let num_features = 30;
    let learning_rate = 0.1;
    let epochs = 1000;

    let mut x = vec![];
    let mut y = vec![];
    generate_dataset(&mut x, &mut y, num_samples, num_features);

    let train_size = (0.8 * num_samples as f64) as usize;
    let x_train = &x[..train_size];
    let y_train = &y[..train_size];
    let x_test = &x[train_size..];
    let y_test = &y[train_size..];

    let start_time = std::time::Instant::now();
    let weights = logistic_regression(x_train, y_train, learning_rate, epochs);
    let duration = start_time.elapsed().as_secs_f64();

    let accuracy = calculate_accuracy(x_test, y_test, &weights);

    println!("Execution time: {:.4} seconds", duration);
    println!("Accuracy: {:.2} %", accuracy);
}
