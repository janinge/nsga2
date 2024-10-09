use rand::prelude::*;

use crate::solution::Solution;

#[derive(Clone, Debug)]
pub(crate) struct Quadratic {
    pub(crate) x1: f64,
    pub(crate) x2: f64,
}

impl Solution for Quadratic {
    type Data = (f64, f64);

    fn random() -> Self {
        let mut rng = thread_rng();
        let x1 = rng.gen_range(0.0..=5.0);
        let x2 = rng.gen_range(0.0..=3.0);
        Self { x1, x2 }
    }

    /// Add/subtract small random values to/from x_1 and x_2
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let mutation_strength = 0.1;

        let mut x1 = self.x1 + rng.gen_range(-mutation_strength..=mutation_strength);
        let mut x2 = self.x2 + rng.gen_range(-mutation_strength..=mutation_strength);

        // Clamp the mutated variables within their constraints
        x1 = x1.clamp(0.0, 5.0);
        x2 = x2.clamp(0.0, 3.0);

        self.x1 = x1;
        self.x2 = x2;
    }

    /// Blend crossover (BLX-α)
    fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let alpha = 0.5;

        let gamma1 = (1.0 + 2.0 * alpha) * rng.gen::<f64>() - alpha;
        let gamma2 = (1.0 + 2.0 * alpha) * rng.gen::<f64>() - alpha;

        let mut x1 = gamma1 * self.x1 + (1.0 - gamma1) * other.x1;
        let mut x2 = gamma2 * self.x2 + (1.0 - gamma2) * other.x2;

        x1 = x1.clamp(0.0, 5.0);
        x2 = x2.clamp(0.0, 3.0);

        Self { x1, x2 }
    }

    fn fitness(&self) -> Vec<f64> {
        let f1 = 4.0 * self.x1.powi(2) + 4.0 * self.x2.powi(2);
        let f2 = (self.x1 - 5.0).powi(2) + (self.x2 - 5.0).powi(2);
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

    fn feasible(&self) -> bool {
        let c1 = (self.x1 - 5.0).powi(2) + self.x2.powi(3) <= 25.0 + 1e-6;
        let c2 = (self.x1 - 8.0).powi(2) + (self.x2 + 3.0).powi(2) >= 7.7 - 1e-6;
        c1 && c2
    }
}