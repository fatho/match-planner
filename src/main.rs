use std::fs::File;
use std::path::PathBuf;
use std::io::Write;
use rand::seq::SliceRandom;
use itertools::{Itertools, MinMaxResult};
use structopt::StructOpt;

// submodules
mod errors;
mod input;
mod ga;
use input::{PlanningData, Player};

/// The command line options that can be given to this application.
#[derive(Debug, StructOpt)]
#[structopt(name = "match-lanner", about = "An evolutionary planning application for assigning players to matches.")]
struct Opt {
    /// Population size used for the evolutionary algorithm.
    #[structopt(short = "n", long = "population", default_value = "100000")]
    population_size: usize,

    /// Input file, stdin if not present
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file, stdout if not present
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,

    /// Previous plannings that should be continued
    #[structopt(short = "p", long = "previous", parse(from_os_str))]
    previous: Vec<PathBuf>,

    /// Quiet mode, do not print anything to stderr
    #[structopt(short = "q", long = "quiet")]
    quiet: bool,
}

/// Implements Write but doesn't write anything.
struct NullWrite;

impl Write for NullWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn main() -> ! {
    match run() {
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(1)
        },
        Ok(_) => {
            std::process::exit(0)
        }
    }
}

fn run() -> errors::Result<()> {
    let opt = Opt::from_args();

    let input = match opt.input {
        None => PlanningData::load(std::io::stdin())?,
        Some(file_name) => {
            let file = File::open(&file_name)?;
            PlanningData::load(file)?
        },
    };

    let mut log_out: Box<dyn Write> = if opt.quiet {
        Box::new(NullWrite)
    } else {
        Box::new(std::io::stderr())
    };

    writeln!(log_out, "Number of players: {}", input.players().len())?;
    writeln!(log_out, "Number of matches: {}", input.match_count())?;

    let rng = rand::thread_rng();

    let mut planner = Planner::new(rng, opt.population_size, input);

    writeln!(log_out, "Beginning evolutionary optimization")?;
    writeln!(log_out, "Population size: {}", opt.population_size)?;

    writeln!(log_out, "Generation | Average quality | Best quality")?;

    let mut print_stats = |planner: &Planner<_>| {
        let fitness_sum = planner.population().iter().map(|ind| ind.fitness().raw()).sum::<f64>();
        let count = planner.population().len();

        writeln!(log_out, "{: >10} | {: >15.3} | {: >10.3}", planner.generation(), fitness_sum / count as f64, planner.best().fitness().raw()).unwrap();
    };

    print_stats(&planner);

    while planner.improve() {
        print_stats(&planner);
    }

    print_stats(&planner);

    writeln!(log_out, "Population converged on a solution")?;

    match opt.output {
        None => print_solution(std::io::stdout(), planner.planning.players(), planner.best().genome())?,
        Some(file_name) => {
            let file = File::create(&file_name)?;
            print_solution(file, planner.planning.players(), planner.best().genome())?;
        },
    }

    Ok(())
}

fn print_solution<W: Write>(out: W, players: &[Player], assignment: &Assignment) -> errors::Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(out);

    // Header, consists of the player names
    writer.write_record(players.iter().map(|p| p.name()))?;

    // Rows, one for each match day. Contains a 1 in the columns of the two players playing on that day.
    for pair in assignment.match_parings.iter() {
        writer.write_record(players.iter().enumerate()
            .map(|(index, _)| if pair.left.0 == index || pair.right.0 == index { "1" } else { "" })
        )?;
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

        // flatten tuples and mutate offspring
        for (mut child1, mut child2) in offspring {
            self.mutate(&mut child1);
            self.mutate(&mut child2);
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

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn population(&self) -> &[ga::Individual<Assignment>] {
        self.population.as_slice()
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

    fn mutate(&mut self, assignment: &mut Assignment) {
        let match_count = assignment.match_parings.len();

        // Approximate exponential distribution of number of mutations
        let mut num_mutations = 1;
        while num_mutations < match_count && self.rng.gen_bool(0.3) {
            num_mutations += 1;
        }


        if self.rng.gen_bool(0.2) {
            // In 10% of cases switch out one player at random per mutation
            let mutation_indexes = rand::seq::index::sample(&mut self.rng, match_count, num_mutations);
            for mut_idx in mutation_indexes.into_iter() {
                let pairing = &mut assignment.match_parings[mut_idx];
                let player1 = if self.rng.gen() { pairing.left } else { pairing.left };

                let mut player2 = self.rng.gen_range(0, self.planning.players().len() - 1);
                if player2 >= player1.0 {
                    player2 += 1
                };
                *pairing = MatchPair::new(player1, PlayerId(player2));
            }
        } else {
            // In the remaining 90% of cases, swap the given number of pairs.
            for _ in 0..num_mutations {
                let i = self.rng.gen_range(0, match_count);
                let j = self.rng.gen_range(0, match_count);
                self.population.swap(i, j);
            }
        }
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
        let match_count = assignment.match_parings.len();

        // Note: when match_count * 2 is evenly divisble by player count,
        // min_expected_matches == max_expected_matches
        let min_expected_matches = match_count * 2 / player_count;
        let max_expected_matches = (match_count * 2 + player_count - 1) / player_count;

        // equal number of plays
        let mut counts: Vec<usize> = vec![0; player_count];

        // variety of opponents
        let mut variety_matrix = vec![false; player_count * player_count];
        let matrix_index = |pair: &MatchPair| player_count * pair.left.0 + pair.right.0;
        let matrix_index_flip = |pair: &MatchPair| player_count * pair.right.0 + pair.left.0;

        // availability
        let mut unavailability_penalty = 0;

        for (day, pair) in assignment.match_parings.iter().enumerate() {
            counts[pair.left.0] += 1;
            counts[pair.right.0] += 1;

            variety_matrix[matrix_index(pair)] = true;
            variety_matrix[matrix_index_flip(pair)] = true;

            if ! planning.players()[pair.left.0].availability()[day] {
                unavailability_penalty += 1;
            }
            if ! planning.players()[pair.right.0].availability()[day] {
                unavailability_penalty += 1;
            }
        }

        let variety_score = variety_matrix.into_iter()
            .filter(|played| *played).count();

        // equidistant matches
        let mut match_distances: Vec<f64> = Vec::new();
        for player_num in 0..player_count {
            let player = PlayerId(player_num);
            let mut last_match = 0;
            for (day, pair) in assignment.match_parings.iter().enumerate() {
                if pair.involves(player) {
                    match_distances.push((day - last_match + 1) as f64);
                    last_match = day + 1;
                }
            }
            match_distances.push((match_count - last_match + 2) as f64);
        }

        let distance_count = match_distances.iter().count() as f64;
        let average_time = match_distances.iter().sum::<f64>() / distance_count;
        let equidistance_penalty = match_distances.iter()
            .map(|dist| dist - average_time)
            .map(|deviation| deviation * deviation)
            .sum::<f64>();

        let inequality_penalty = counts.into_iter()
            // Compute how far away a player is from the expected number of matches
            // everyone should play
            .map(|count| if count < min_expected_matches {
                min_expected_matches - count
            } else if count > max_expected_matches {
                count - max_expected_matches
            } else {
                0
            })
            .sum::<usize>();

        let weighted_score =
            inequality_penalty         as f64 * -6.0 +
            unavailability_penalty     as f64 * -10.0 +
            equidistance_penalty              * -1.0 +
            variety_score              as f64 *  3.0;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlayerId(usize);

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

    pub fn involves(&self, player: PlayerId) -> bool {
        self.left == player || self.right == player
    }
}
