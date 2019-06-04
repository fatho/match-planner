use std::fs::File;
use std::path::Path;
use rand::seq::SliceRandom;
use itertools::{Itertools, MinMaxResult};

mod input;
mod ga;
use input::PlanningData;

fn main() -> input::Result<()> {
    let input = match std::env::args().skip(1).next() {
        None => PlanningData::load(std::io::stdin())?,
        Some(file_name) => {
            let path = Path::new::<str>(file_name.as_ref());
            let file = File::open(path)?;
            PlanningData::load(file)?
        },
    };

    eprintln!("Number of players: {}", input.players().len());
    eprintln!("Number of matches: {}", input.match_count());

    let rng = rand::thread_rng();

    let mut planner = Planner::new(rng, 100, input);

    eprintln!("Initial quality: {:?}", planner.best().fitness());

    while planner.improve() {
        eprintln!("Generation quality: {:?}", planner.best().fitness());
    }

    eprintln!("Final quality: {:?}", planner.best().fitness());

    // Header
    println!("{}", planner.planning.players().iter().map(|p| p.name()).join(";"));
    // Match rows
    for pair in planner.best().genome().match_parings.iter() {
        let row = planner.planning.players().iter().enumerate()
            .map(|(index, _)| if pair.left.0 == index || pair.right.0 == index { "1" } else { "" })
            .join(";");
        println!("{}", row);
    }

    Ok(())
}

#[derive(Debug)]
pub struct Planner<R> {
    rng: R,
    planning: PlanningData,
    population_size: usize,
    population: Vec<ga::Individual<Assignment>>,
    generation: usize,
}

impl<R: rand::Rng> Planner<R> {
    pub fn new(mut rng: R, population_size: usize, planning: PlanningData) -> Self {
        assert!(population_size > 0);

        let match_count = planning.match_count();
        let player_count = planning.players().len();
        let initial_population: Vec<_> = std::iter::repeat_with(|| {
                let assignment = Assignment::random(&mut rng, match_count, player_count);
                let fitness = Self::fitness(&planning, &assignment);
                ga::Individual::new(0, assignment, fitness)
            })
            .take(population_size)
            .collect();

        Planner {
            rng: rng,
            planning: planning,
            population_size: population_size,
            population: initial_population,
            generation: 0,
        }
    }

    pub fn improve(&mut self) -> bool {
        self.generation += 1;
        // Tweakables
        let tournament_size = 2;

        // Select parents via turnament and generate offspring
        let rng = &mut self.rng;
        self.population.shuffle(rng);
        let offspring: Vec<_> = self.population.as_slice()
            // one tournament for each parent
            .chunks_exact(tournament_size * 2)
            // select best of each half
            .map(|chunk| {
                let parent1 = chunk[..tournament_size].iter()
                    .max_by_key(|individual| individual.fitness())
                    .unwrap();
                let parent2 = chunk[tournament_size..].iter()
                    .max_by_key(|individual| individual.fitness())
                    .unwrap();
                Self::generate_offspring(rng, parent1.genome(), parent2.genome())
            })
            .collect();

        // flatten tuples
        for (child1, child2) in offspring {
            self.population.push(self.make_individual(child1));
            self.population.push(self.make_individual(child2));
        }

        // Select best-of for next generation (with the highest fitness, breaking ties using lowest generation)
        self.population.sort_unstable_by_key(|individual| (
            std::cmp::Reverse(individual.fitness()),
            individual.generation()
        ));
        self.population.truncate(self.population_size);

        // Check if at least one offspring was chosen for the new generation
        self.population.iter().any(|individual| individual.generation() == self.generation)
    }

    pub fn best(&self) -> &ga::Individual<Assignment> {
        self.population.iter()
            .max_by_key(|individual| individual.fitness())
            .expect("population is at least 1")
    }

    fn make_individual(&self, assignment: Assignment) -> ga::Individual<Assignment> {
        let fitness = Self::fitness(&self.planning, &assignment);
        ga::Individual::new(self.generation, assignment, fitness)
    }

    fn generate_offspring(rng: &mut R, parent1: &Assignment, parent2: &Assignment) -> (Assignment, Assignment) {
        let pair_count = parent1.match_parings.len();

        let split_point = rng.gen_range(0, pair_count);

        let mut child1 = Assignment { match_parings: Vec::with_capacity(pair_count) };
        let mut child2 = Assignment { match_parings: Vec::with_capacity(pair_count) };

        child1.match_parings.extend(parent1.match_parings[0..split_point].iter().cloned());
        child1.match_parings.extend(parent2.match_parings[split_point..pair_count].iter().cloned());

        child2.match_parings.extend(parent2.match_parings[0..split_point].iter().cloned());
        child2.match_parings.extend(parent1.match_parings[split_point..pair_count].iter().cloned());

        (child1, child2)
    }

    /// Judge the quality of an assignment.
    /// A good assignment should fulfil the following criteria
    ///
    /// 1. all players should play the same number of matches
    /// 2. all players should have roughly the same distance between their matches
    /// 3. all players should play with as many other players as possible
    /// 4. a player should never be assigned to a date where they cannot play
    fn fitness(planning: &PlanningData, assignment: &Assignment) -> ga::Fitness {
        let player_count = planning.players().len();

        // 1. equal number of plays
        let mut counts: Vec<usize> = vec![0; player_count];

        // 2. equidistant matches
        let mut last_match: Vec<Option<usize>> = vec![None; player_count];
        let mut min_max_match_distance: Vec<MinMaxStat<usize>> = vec![MinMaxStat::new(); player_count];

        // 3. variety of opponents
        let mut variety_matrix = vec![false; player_count * player_count];
        let matrix_index = |pair: &MatchPair| player_count * pair.left.0 + pair.right.0;
        let matrix_index_flip = |pair: &MatchPair| player_count * pair.right.0 + pair.left.0;

        // 4. availability
        let mut unavailability_score = 0;


        for (day, pair) in assignment.match_parings.iter().enumerate() {
            counts[pair.left.0] += 1;
            counts[pair.right.0] += 1;

            variety_matrix[matrix_index(pair)] = true;
            variety_matrix[matrix_index_flip(pair)] = true;

            if ! planning.players()[pair.left.0].availability()[day] {
                unavailability_score += 1;
            }
            if ! planning.players()[pair.right.0].availability()[day] {
                unavailability_score += 1;
            }

            if let Some(last_day) = last_match[pair.left.0].replace(day) {
                min_max_match_distance[pair.left.0].observe(day - last_day);
            }
            if let Some(last_day) = last_match[pair.right.0].replace(day) {
                min_max_match_distance[pair.right.0].observe(day - last_day);
            }
        }

        let played_at_least_once_score = counts.iter().filter(|cnt| **cnt > 0).count();

        let inequality_score = match counts.into_iter().minmax() {
            MinMaxResult::MinMax(min, max) => max - min,
            _ => 0,
        };

        let variety_score = variety_matrix.into_iter()
            .filter(|played| *played).count();

        let equidistance_score: usize = min_max_match_distance.iter()
            .map(|stat| stat.min.and_then(|min| stat.max.map(|max| max - min)))
            .flatten()
            .sum();

        // TODO: equidistant matches

        let weighted_score =
            inequality_score           as f64 * -4.0 +
            played_at_least_once_score as f64 *  2.0 +
            unavailability_score       as f64 * -5.0 +
            equidistance_score         as f64 * -0.5 +
            variety_score              as f64 *  1.0;

        ga::Fitness::new(weighted_score)
    }
}


#[derive(Debug)]
pub struct Assignment {
    match_parings: Vec<MatchPair>,
}

impl Assignment {
    pub fn random(rng: &mut impl rand::Rng, match_count: usize, player_count: usize) -> Self {
        let assignment = std::iter::repeat_with(|| MatchPair::random(rng, player_count))
            .take(match_count)
            .collect();

        Assignment {
            match_parings: assignment,
        }
    }
}

// struct Constraints {
//     pub num_players: usize,
//     pub num_days: usize,
//     pub blocked_days: usize,
//     pub count_bias: Vec<usize>,
// }

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlayerId(usize);

// #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
// pub struct Day(usize);

#[derive(Debug, Clone)]
pub struct MatchPair {
    left: PlayerId,
    right: PlayerId,
}

impl MatchPair {
    /// Create a normalized pair where the lower
    pub fn new(left: PlayerId, right: PlayerId) -> Self {
        assert!(left != right, "two distinct players are required for a match");
        MatchPair {
            left: left.min(right),
            right: left.max(right),
        }
    }

    pub fn random(rng: &mut impl rand::Rng, player_count: usize) -> Self {
        assert!(player_count > 1, "Must have more than one player for making a pair");

        // Generate two disjoint players
        let player1 = rng.gen_range(0, player_count);
        let mut player2 = rng.gen_range(0, player_count - 1);
        if player2 >= player1 {
            player2 += 1;
        }
        MatchPair::new(PlayerId(player1), PlayerId(player2))
    }
}


#[derive(Clone, Copy, Debug)]
struct MinMaxStat<T> {
    min: Option<T>,
    max: Option<T>,
}

impl<T: Ord + Copy> MinMaxStat<T> {
    fn new() -> Self {
        MinMaxStat { min: None, max: None }
    }

    fn observe(&mut self, value: T) {
        if self.min.map_or(true, |the_min| value < the_min) {
            self.min = Some(value)
        }
        if self.max.map_or(true, |the_max| value > the_max) {
            self.max = Some(value)
        }
    }
}