
use rand::prelude::*;
use rand::distributions::Uniform;
use crate::solution::Solution;

pub struct NSGA2<S: Solution> {
    population_size: usize,
    max_generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    population: Vec<S>,
}

impl<S: Solution> NSGA2<S> {
    pub fn new(population_size: usize, max_generations: usize, mutation_rate: f64, crossover_rate: f64) -> Self {
        let mut population = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            population.push(S::random());
        }

        Self {
            population_size,
            max_generations,
            mutation_rate,
            crossover_rate,
            population,
        }
    }

    pub fn run(&mut self) {
        let objective_count = self.population[0].fitness().len();

        for _ in 0..self.max_generations {
            let fronts = non_dominated_sort(&self.population);

            let distances = fronts.iter().flat_map(|front| {
                let front_solutions = front.iter().map(|&index| self.population[index].clone()).collect::<Vec<_>>();
                crowding_distance(&front_solutions, objective_count)
            }).collect::<Vec<_>>();

            let mating_pool = self.tournament_selection(&fronts, &distances);
            let offspring = self.crossover_and_mutate(&mating_pool);

            let mut combined_population = self.population.clone();
            combined_population.extend(offspring);

            let fronts = non_dominated_sort(&combined_population);

            let mut new_population = Vec::with_capacity(self.population_size);

            for front in fronts {
                let front_solutions = front.iter().map(|&i| combined_population[i].clone()).collect::<Vec<_>>();
                let distance = crowding_distance(&front_solutions, objective_count);

                if new_population.len() + front.len() <= self.population_size {
                    new_population.extend(front_solutions);
                    continue;
                }

                // Sort by crowding distance if this front is larger than remaining space for population
                let mut front_with_distances = front_solutions.iter().zip(distance).collect::<Vec<_>>();
                front_with_distances.sort_by(|a, b| b.1.total_cmp(&a.1));

                for (solution, _) in front_with_distances {
                    if new_population.len() < self.population_size {
                        new_population.push(solution.clone());
                    } else {
                        break;
                    }
                }
            }

            self.population = new_population;
        }
    }

    pub fn current_population(&self) -> &Vec<S> {
        &self.population
    }

    fn tournament_selection(&self, fronts: &[Vec<usize>], distances: &Vec<f64>) -> Vec<S> {
        let mut mating_pool = Vec::with_capacity(self.population_size);
        let mut rng = thread_rng();

        while mating_pool.len() < self.population_size {
            let idx1 = rng.gen_range(0..self.population_size);
            let idx2 = rng.gen_range(0..self.population_size);

            let ind1 = &self.population[idx1];
            let ind2 = &self.population[idx2];

            // Determine the ranks of the solutions by finding which front they belong to
            let rank1 = fronts.iter().position(|front| front.contains(&idx1)).unwrap();
            let rank2 = fronts.iter().position(|front| front.contains(&idx2)).unwrap();

            // Crowding distances
            let dist1 = distances[idx1];
            let dist2 = distances[idx2];

            // Feasible solutions dominate infeasible ones so constraints are satisfied
            let feasible1 = ind1.feasible();
            let feasible2 = ind2.feasible();

            let selected = if feasible1 && !feasible2 {
                ind1.clone()
            } else if !feasible1 && feasible2 {
                ind2.clone()
            } else if rank1 < rank2 {
                ind1.clone()
            } else if rank1 > rank2 {
                ind2.clone()
            } else if dist1 > dist2 {
                ind1.clone()
            } else {
                ind2.clone()
            };

            mating_pool.push(selected);
        }
        
        mating_pool
    }

    fn crossover_and_mutate(&self, mating_pool: &[S]) -> Vec<S> {
        let mut offspring = Vec::with_capacity(self.population_size);
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0);

        for i in (0..mating_pool.len()).step_by(2) {
            let parent1 = &mating_pool[i];
            let parent2 = &mating_pool[(i + 1) % mating_pool.len()];

            let mut child = if rng.sample(dist) < self.crossover_rate {
                parent1.crossover(parent2)
            } else {
                parent1.clone()
            };

            if rng.sample(dist) < self.mutation_rate {
                child.mutate();
            }

            offspring.push(child);
        }

        offspring
    }
}

fn non_dominated_sort<S: Solution>(population: &[S]) -> Vec<Vec<usize>> {
    let population_size = population.len();
    let mut domination_counts = vec![0; population_size];
    let mut dominated_solutions = vec![Vec::new(); population_size];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front = Vec::new();
    
    for p in 0..population_size {
        for q in 0..population_size {
            if p == q {
                continue;
            }

            if population[p].dominates(&population[q]) {
                dominated_solutions[p].push(q);
            } else if population[q].dominates(&population[p]) {
                domination_counts[p] += 1;
            }
        }
        
        if domination_counts[p] == 0 {
            first_front.push(p);
        }
    }

    fronts.push(first_front);
    
    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut next_front = Vec::new();

        for &p in &fronts[i] {
            for &q in &dominated_solutions[p] {
                domination_counts[q] -= 1;
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }

        i += 1;
        if !next_front.is_empty() {
            fronts.push(next_front.clone());
        } else {
            break;
        }
    }

    fronts
}

fn crowding_distance<S: Solution>(front: &[S], objective_count: usize) -> Vec<f64> {
    let mut distances = vec![0.0; front.len()];

    for m in 0..objective_count {
        let mut sorted = front
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.fitness()[m]))
            .collect::<Vec<_>>();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances[sorted[0].0] = f64::INFINITY;
        distances[sorted[sorted.len() - 1].0] = f64::INFINITY;

        let min = sorted[0].1;
        let max = sorted[sorted.len() - 1].1;

        if (max - min).abs() < f64::EPSILON {
            continue; // Avoid division by zero
        }

        for j in 1..sorted.len() - 1 {
            distances[sorted[j].0] += (sorted[j + 1].1 - sorted[j - 1].1) / (max - min);
        }
    }

    distances
}
