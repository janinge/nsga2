use crate::solution::Solution;

use rand::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct Series {
    pub(crate) r: Vec<f64>,  // Reliabilities – continuous
    pub(crate) n: Vec<u32>,  // Redundancies - discrete
    pub(crate) m: usize,     // Number of components
}

fn random(m: usize) -> Series {
    let mut rng = thread_rng();

    // Initialize reliabilities r_i between 0.01 and 1.0 to avoid ln(0)
    let r = (0..m).map(|_| rng.gen_range(0.01..=1.0)).collect();

    // Initialize redundancies n_i
    let n = (0..m).map(|_| rng.gen_range(1..=m as u32)).collect();

    Series { r, n, m }
}

impl Solution for Series {
    type Data = (Vec<f64>, Vec<u32>);

    fn random() -> Self {
        random(5)
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let m = self.r.len();

        // Mutate reliabilities
        for i in 0..m {
            if rng.gen_bool(0.2) { // 20% chance for mutation of each reliability
                let delta = rng.gen_range(-0.05..=0.05);
                self.r[i] = (self.r[i] + delta).clamp(0.5, 1.0);
            }
        }

        // Mutate redundancies
        for i in 0..m {
            if rng.gen_bool(0.2) { // 20% chance for mutation of each redundancy
                let delta = if rng.gen_bool(0.5) { 1 } else { -1 };
                let new_value = (self.n[i] as i32 + delta).max(1);
                self.n[i] = new_value as u32;
            }
        }
    }
    
    fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();

        let mut cross_r = Vec::with_capacity(self.r.len());
        let mut cross_n = Vec::with_capacity(self.n.len());

        // Binary crossover (BLX-α) for reliabilities
        let alpha = 0.5;
        for i in 0..self.r.len() {
            let c_min = self.r[i].min(other.r[i]);
            let c_max = self.r[i].max(other.r[i]);
            let range = c_max - c_min;
            let lower = (c_min - alpha * range).clamp(0.5, 1.0);
            let upper = (c_max + alpha * range).clamp(0.5, 1.0);
            let value = rng.gen_range(lower..=upper);
            cross_r.push(value);
        }
        
        for i in 0..self.n.len() {
            let value = if rng.gen_bool(0.5) { self.n[i] } else { other.n[i] };
            cross_n.push(value);
        }

        Self { r: cross_r, n: cross_n, m: self.m }
    }
    
    fn fitness(&self) -> Vec<f64> {
        let alpha = [2.33, 1.45, 0.541, 8.05, 1.95];
        let beta = [1.5; 5];
        let C = 200.0;

        let mut f1 = 1.0; // System reliability
        let mut f2 = 0.0; // System cost

        // Calculate R_i(n_i) and cost for each component
        for i in 0..self.m {
            let r_i = self.r[i];
            let n_i = self.n[i] as f64;
            let alpha_i = alpha[i];
            let beta_i = beta[i];

            // Calculate R_i(n_i)
            let R_i = 1.0 - (1.0 - r_i).powf(n_i);
            f1 *= R_i;

            // Calculate cost
            let cost = alpha_i * ((-1000.0 / r_i.ln()).powf(beta_i)) * (n_i + (0.25 * n_i).exp());
            f2 += cost;
        }
        f2 -= C;

        // We want to maximize system reliability, so invert f1 and turn it into a minimization problem
        let f1_min = -f1;

        vec![f1_min, f2]
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
        let v = [7.0, 8.0, 8.0, 6.0, 9.0];
        let w_i = [110.0; 5];
        let V = 175.0;
        let W = 200.0;
        
        let mut g1 = 0.0;
        let mut g2 = 0.0;

        for i in 0..self.m {
            let n_i = self.n[i] as f64;
            let w_i_i = w_i[i];
            let v_i = v[i];

            let term1 = w_i_i * v_i * 2.0_f64.powf(n_i);
            g1 += term1 * term1;

            let term2 = w_i_i * n_i * (0.25 * n_i).exp();
            g2 += term2;
        }

        g1 -= V;
        g2 -= W;

        // Decision variable constraints
        let r_bounds = self.r.iter().all(|&r_i| (0.5..=1.0).contains(&r_i));
        let n_bounds = self.n.iter().all(|&n_i| n_i >= 1);
        
        g1 <= 0.0 && g2 <= 0.0 && r_bounds && n_bounds
    }
}
