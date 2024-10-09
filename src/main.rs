use crate::solution::Solution;
use crate::nsga2::NSGA2;
use crate::problems::*;

mod solution;
mod nsga2;
mod problems;

fn main() {
    // Problem 1.2
    let mut quadratic = NSGA2::<quadratic::Quadratic>::new(100, 1_000, 0.1, 0.9);

    quadratic.run();

    let final_population = quadratic.current_population();

    // Filter feasible solutions
    let feasible_solutions: Vec<&quadratic::Quadratic> = final_population
        .iter()
        .filter(|s| s.feasible())
        .collect();

    println!("{} of {} are feasible solutions to Problem 1.2:", feasible_solutions.len(), final_population.len());

    // Display the final Pareto front
    for solution in feasible_solutions {
        println!(
            "x1: {:.4}, x2: {:.4} == {:.4?}",
            solution.x1,
            solution.x2,
            solution.fitness()
        );
    }

    // Problem 1.3
    let mut thirty = NSGA2::<thirty::Thirty>::new(100, 1_000, 0.1, 0.9);
    
    thirty.run();
    
    let final_population = thirty.current_population();
    
    // Filter feasible solutions
    let feasible_solutions: Vec<&thirty::Thirty> = final_population
        .iter()
        .filter(|s| s.feasible())
        .collect();

    println!("\n{} of {} are feasible solutions to Problem 1.3:", feasible_solutions.len(), final_population.len());
    
    for solution in feasible_solutions {
        for (i, &val) in solution.data.iter().enumerate() {
            if i > 0 {
                print!("\t");
            }
            print!("{:.4}", val);
        }
        println!(
            " == {:.4?}",
            solution.fitness()
        );
    }
    
    // Problem 3.1
    let mut series = NSGA2::<series::Series>::new(100, 1_000, 0.1, 0.9);
    
    series.run();
    
    let final_population = series.current_population();
    
    // Filter feasible solutions
    let feasible_solutions: Vec<&series::Series> = final_population
        .iter()
        .filter(|s| s.feasible())
        .collect();
    
    println!("\n{} of {} are feasible solutions to Problem 3.1:", feasible_solutions.len(), final_population.len());

    for solution in feasible_solutions {
        println!(
            "r: {:.4?}, n: {:.4?} == {:.4?}",
            solution.r,
            solution.n,
            solution.fitness()
        );
    }

}
