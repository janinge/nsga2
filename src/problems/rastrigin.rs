use rand::prelude::*;
use std::f64::consts::PI;
use crate::solution::Solution;

#[derive(Clone, Debug)]
pub(crate) struct Rastrigin {
    pub(crate) data: Vec<f64>, // Decision variables x_i, where i = 1..n
}

impl Rastrigin {
    fn random_n(n: usize) -> Self {
        let mut rng = thread_rng();
        let data = (0..n).map(|_| rng.gen_range(-5.12..=5.12)).collect();
        Self { data }
    }
}

impl Solution for Rastrigin {
    type Data = Vec<f64>;

    /// Initialize a new random solution
    fn random() -> Self {
        let mut rng = thread_rng();
        let n = 20;
        let data = (0..n).map(|_| rng.gen_range(-5.12..=5.12)).collect();
        Self { data }
    }
    
    /// Add/subtract small random values to the decision variables
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let mutation_strength = 0.2;
        let n = self.data.len();

        // Decide how many variables should be mutated, up to a maximum of 25%
        let num_mutations = rng.gen_range(1..=n / 4);

        for _ in 0..num_mutations {
            let idx = rng.gen_range(0..n);
            let delta = rng.gen_range(-mutation_strength..=mutation_strength);
            self.data[idx] = (self.data[idx] + delta).clamp(-5.12, 5.12);
        }
    }

    /// Blend crossover (BLX-α)
    fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let n = self.data.len();
        let alpha = 0.5;
        let mut blended = vec![0.0; n];

        for i in 0..n {
            let c_min = self.data[i].min(other.data[i]);
            let c_max = self.data[i].max(other.data[i]);
            let range = c_max - c_min;
            let lower = c_min - alpha * range;
            let upper = c_max + alpha * range;
            blended[i] = rng.gen_range(lower..=upper).clamp(-5.12, 5.12);
        }

        Self { data: blended }
    }
    
    fn fitness(&self) -> Vec<f64> {
        let n = self.data.len() as f64;
        let sum: f64 = self
            .data
            .iter()
            .map(|&x| x.powi(2) - 10.0 * (2.0 * PI * x).cos())
            .sum();
        let f = 10.0 * n + sum;
        vec![f]
    }
    
    fn dominates(&self, other: &Self) -> bool {
        let self_fitness = self.fitness()[0];
        let other_fitness = other.fitness()[0];
        self_fitness < other_fitness
    }
    
    fn feasible(&self) -> bool {
        self.data.iter().all(|&x| (-5.12..=5.12).contains(&x))
    }
}
