//! This module contains data structures that are useful for developing evolutionary algorithms.


/// Type for the fitness of an individual, represented as float.
/// The constructor imposes the restriction that the fitness value must not be nan,
/// so that it is possible to implement `Ord` for `Fitness`.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Fitness(f64);

impl Fitness {
    /// Create a new fitness value.
    ///
    /// # Panics
    ///
    /// A panic is raised if `fitness` is not a number, because
    /// fitness values must form a total order. That is, the following
    /// snippet will panic:
    ///
    /// ```should_panic
    /// let a = Fitness::new(f64::NAN);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// let a = Fitness::new(2.0);
    /// let b = Fitness::new(3.0);
    /// assert_eq!(a.cmp(b), std::cmp::Ordering::Less)
    /// ```
    pub fn new(fitness: f64) -> Self {
        assert!(! fitness.is_nan(), "fitness must not be Nan");
        Fitness(fitness)
    }

    /// Return the raw fitness value that was passed to `Fitness::new`.
    pub fn raw(&self) -> f64 {
        self.0
    }
}

impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for Fitness {}

/// An "individual" with a specific genome in a population.
/// Users of this type are responsible for computing the fitness
/// and keeping track of the number of generations.
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
