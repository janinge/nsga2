pub trait Solution: Clone {
    type Data;

    fn random() -> Self;

    fn mutate(&mut self);

    fn crossover(&self, other: &Self) -> Self;

    fn fitness(&self) -> Vec<f64>;

    fn dominates(&self, other: &Self) -> bool;

    fn feasible(&self) -> bool;
}
