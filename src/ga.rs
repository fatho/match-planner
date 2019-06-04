#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Fitness(f64);

impl Fitness {
    pub fn new(fitness: f64) -> Self {
        assert!(! fitness.is_nan(), "fitness must not be Nan");
        Fitness(fitness)
    }
}

impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for Fitness {}

#[derive(Debug)]
pub struct Individual<Genome> {
    /// The fitness of this individual, higher is better.
    fitness: Fitness,
    /// The generation that produced this individual.
    generation: usize,
    /// The defining features of this individual.
    genome: Genome,
}

impl<Genome> Individual<Genome> {
    pub fn new(generation: usize, genome: Genome, fitness: Fitness) -> Self {
        Individual {
            fitness: fitness,
            genome: genome,
            generation: generation,
        }
    }

    pub fn fitness(&self) -> Fitness {
        self.fitness
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }
}
