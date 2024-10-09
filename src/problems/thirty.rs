use crate::solution::Solution;

use rand::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub(crate) struct Thirty {
    pub(crate) data: Vec<f64>, // Decision variables x_i, where i = 1..30
}

impl Solution for Thirty {
    type Data = Vec<f64>;
    
    fn random() -> Self {
        let mut rng = thread_rng();
        let n = 30;
        let data = (0..n).map(|_| rng.gen_range(0.0..=1.0)).collect();
        Self { data }
    }

    /// Add/subtract small random values to/from a random selection of variables
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let mutation_strength = 0.1;
        let n = self.data.len();

        // Decide how many variables should be mutated, up to a maximum of 25%
        let num_mutations = rng.gen_range(1..=n / 4);

        for _ in 0..num_mutations {
            let idx = rng.gen_range(0..n);
            let delta = rng.gen_range(-mutation_strength..=mutation_strength);
            self.data[idx] = (self.data[idx] + delta).clamp(0.0, 1.0);
        }
    }

    /// Try simulated binary crossover (SBX)
    fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let n = self.data.len();
        let mut child_data = vec![0.0; n];
        let eta = 2.0; // Crossover distribution index

        for i in 0..n {
            let u = rng.gen::<f64>();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };

            child_data[i] = 0.5 * ((1.0 + beta) * self.data[i] + (1.0 - beta) * other.data[i]);

            // Ensure child variable is within bounds
            child_data[i] = child_data[i].clamp(0.0, 1.0);
        }

        Self { data: child_data }
    }

    /// Return the value of all (?) objective functions
    fn fitness(&self) -> Vec<f64> {
        let x1 = self.data[0];

        let g: f64 = 1.0 + self.data[1..].iter().sum::<f64>();

        let h = 1.0 - (x1 / g).sqrt() - (x1 / g) * (10.0 * PI * x1).sin();

        let f1 = x1;
        let f2 = g * h;

        vec![f1, f2]
    }
    
    fn dominates(&self, other: &Self) -> bool {
        let self_fitness = self.fitness();
        let other_fitness = other.fitness();

        let mut better_in_any = false;

        for (self_value, other_value) in self_fitness.iter().zip(other_fitness.iter()) {
            // If self is worse in any objective, it does not dominate
            if self_value > other_value {
                return false;
            }

            // All objectives should be minimized
            if self_value < other_value {
                better_in_any = true;
            }
        }

        better_in_any
    }

    /// Check if all variables are within [0, 1]
    fn feasible(&self) -> bool {
        self.data.iter().all(|&x| (0.0..=1.0).contains(&x))
    }
}