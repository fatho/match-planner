use std::fs::File;
use std::path::PathBuf;
use std::io::Write;
use rand::seq::SliceRandom;
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
    #[structopt(short = "p", long = "past", parse(from_os_str))]
    past: Vec<PathBuf>,

    /// Plan double matches instead of singles.
    #[structopt(short = "d", long = "double")]
    double: bool,

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
    let opt = Opt::from_args();
    let result = if opt.double {
        run::<DoubleMatchPair>(opt)
    } else {
        run::<SingleMatchPair>(opt)
    };
    match result {
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(1)
        },
        Ok(_) => {
            std::process::exit(0)
        }
    }
}

fn run<P: MatchPair + Clone>(opt: Opt) -> errors::Result<()> {
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

    let mut past_matches = Vec::<P>::new();

    for previuos_file in opt.past.iter() {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_path(previuos_file)?;

        // Headers must match
        if ! reader.headers()?.iter().eq(input.players().iter().map(Player::name)) {
            return Err(errors::Error::InvalidTimetable {
                file: previuos_file.clone(),
                line: 1,
                error: errors::TimetableError::PlayerMismatch
            });
        }

        for (record_num, record) in reader.into_records().enumerate() {
            // find the ones in the row
            let ones = record?.iter().enumerate()
                .filter(|(_, column)| *column == "1")
                .map(|(index, _)| PlayerId(index))
                .collect::<Vec<_>>();

            if let Some(match_pair) = P::new(&ones) {
                past_matches.push(match_pair);
            } else {
                return Err(errors::Error::InvalidTimetable {
                    file: previuos_file.clone(),
                    line: record_num + 2,
                    error: errors::TimetableError::InvalidPlayerCount
                });
            }
        }
    }

    writeln!(log_out, "Number of past matches: {}\n", past_matches.len())?;
    writeln!(log_out, "Number of players: {}", input.players().len())?;
    writeln!(log_out, "Number of matches: {}", input.match_count())?;

    let rng = rand::thread_rng();

    let mut planner = Planner::new(rng, opt.population_size, input, past_matches);

    writeln!(log_out, "Beginning evolutionary optimization")?;
    writeln!(log_out, "Population size: {}", opt.population_size)?;

    writeln!(log_out, "Generation | Average quality | Best quality")?;

    let mut print_stats = |planner: &Planner<_, _>| {
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

fn print_solution<W: Write, P: MatchPair>(out: W, players: &[Player], assignment: &Assignment<P>) -> errors::Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(out);

    // Header, consists of the player names
    writer.write_record(players.iter().map(|p| p.name()))?;

    // Rows, one for each match day. Contains a 1 in the columns of the two players playing on that day.
    for pair in assignment.match_parings.iter() {
        writer.write_record(players.iter().enumerate()
            .map(|(index, _)| if pair.involves(PlayerId(index)) { "1" } else { "" })
        )?;
    }
    Ok(())
}


#[derive(Debug)]
pub struct Planner<R, P> {
    rng: R,
    planning: PlanningData,
    past_matches: Vec<P>,
    population_size: usize,
    population: Vec<ga::Individual<Assignment<P>>>,
    generation: usize,
}

impl<R: rand::Rng, P: MatchPair + Clone> Planner<R, P> {
    pub fn new(
        mut rng: R, population_size: usize, planning: PlanningData, past_matches: Vec<P>
    ) -> Self {
        assert!(population_size > 0);

        let match_count = planning.match_count();
        let player_count = planning.players().len();
        let initial_population: Vec<_> = std::iter::repeat_with(|| {
                let assignment = Assignment::random(&mut rng, match_count, player_count);
                let fitness = Self::fitness(&planning, past_matches.as_slice(), &assignment);
                ga::Individual::new(0, assignment, fitness)
            })
            .take(population_size)
            .collect();

        Planner {
            rng: rng,
            planning: planning,
            past_matches: past_matches,
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

    pub fn best(&self) -> &ga::Individual<Assignment<P>> {
        self.population.iter()
            .max_by_key(|individual| individual.fitness())
            .expect("population is at least 1")
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn population(&self) -> &[ga::Individual<Assignment<P>>] {
        self.population.as_slice()
    }

    fn make_individual(&self, assignment: Assignment<P>) -> ga::Individual<Assignment<P>> {
        let fitness = Self::fitness(&self.planning, self.past_matches.as_slice(), &assignment);
        ga::Individual::new(self.generation, assignment, fitness)
    }

    fn generate_offspring(rng: &mut R, parent1: &Assignment<P>, parent2: &Assignment<P>) -> (Assignment<P>, Assignment<P>) {
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

    fn mutate(&mut self, assignment: &mut Assignment<P>) {
        let match_count = assignment.match_parings.len();

        // Approximate exponential distribution of number of mutations
        let mut num_mutations = 1;
        while num_mutations < match_count && self.rng.gen_bool(0.3) {
            num_mutations += 1;
        }


        if self.rng.gen_bool(0.2) {
            // In 10% of cases switch out one player at random per mutation
            // This mutation is more destructive than the other, but it allows
            // us to generate completely new pairs, and therefore the only way
            // of improving the variety.
            let mutation_indexes = rand::seq::index::sample(&mut self.rng, match_count, num_mutations);
            for mut_idx in mutation_indexes.into_iter() {
                assignment.match_parings[mut_idx].mutate(&mut self.rng, self.planning.players().len());
            }
        } else {
            // In the remaining 90% of cases, swap pairs instead.
            // The advantage of this mutation is that is preserves most
            // fitness coponents. It only affects the availability scores
            // and equidistance scores.
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
    fn fitness(
        planning: &PlanningData, past_matches: &[P], assignment: &Assignment<P>
    ) -> ga::Fitness {
        let player_count = planning.players().len();
        let all_matches: Vec<_> = past_matches.iter().chain(assignment.match_parings.iter()).collect();
        let match_count = all_matches.len();

        // Note: when match_count * 2 is evenly divisble by player count,
        // min_expected_matches == max_expected_matches
        let min_expected_matches = match_count * 2 / player_count;
        let max_expected_matches = (match_count * 2 + player_count - 1) / player_count;

        // equal number of plays
        let mut counts: Vec<usize> = vec![0; player_count];

        // variety of opponents
        let mut variety_matrix = vec![0usize; player_count * player_count];

        // availability
        let mut unavailability_penalty = 0;

        for (day, pair) in assignment.match_parings.iter().enumerate() {
            for player in pair.players() {
                counts[player.0] += 1;
            }

            for p1 in pair.players() {
                for p2 in pair.players() {
                    if p1 != p2 {
                        let matrix_index = player_count * p1.0 + p2.0;
                        variety_matrix[matrix_index] += 1;
                    }
                }
            }

            if day >= past_matches.len() {
                let new_day = day - past_matches.len();
                unavailability_penalty += pair.players().iter().filter(|p| ! planning.players()[p.0].availability()[new_day] ).count();
            }
        }

        let mut variety_penalty = 0;

        for i in 0..player_count {
            // How many matches this player should have played at least/most with each opponent.
            let num_opponents = player_count - 1;
            let min_matches_per_opponent = counts[i] / num_opponents;
            let max_matches_per_opponent = (counts[i] + num_opponents - 1) / num_opponents;

            // Count if the number of matches against each opponent deviates from the expected values
            variety_penalty += variety_matrix[i * player_count..(i+1) * player_count].iter()
                .map(|count| if *count < min_matches_per_opponent {
                    min_matches_per_opponent - count
                } else if *count > max_matches_per_opponent {
                    count - max_matches_per_opponent
                } else {
                    0
                })
                .sum::<usize>();
        }

        // equidistant matches
        let mut match_distances: Vec<f64> = Vec::new();
        for player_num in 0..player_count {
            let player = PlayerId(player_num);
            let mut last_match = 0;
            for (day, pair) in all_matches.iter().enumerate() {
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
            inequality_penalty      as f64 * -10.0 * (P::player_count() - 1).pow(3) as f64 +
            unavailability_penalty  as f64 * -10.0 * (P::player_count() - 1).pow(2) as f64 +
            equidistance_penalty           * -1.0 +
            variety_penalty         as f64 * -3.0;

        ga::Fitness::new(weighted_score)
    }
}


#[derive(Debug)]
pub struct Assignment<P> {
    match_parings: Vec<P>,
}

impl<P: MatchPair> Assignment<P> {
    pub fn random(rng: &mut impl rand::Rng, match_count: usize, player_count: usize) -> Self {
        let assignment = std::iter::repeat_with(|| P::random(rng, player_count))
            .take(match_count)
            .collect();

        Assignment {
            match_parings: assignment,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlayerId(usize);

fn sample_players<R: rand::Rng>(rng: &mut R, player_count: usize, sample_size: usize) -> Vec<PlayerId> {
    assert!(player_count >= sample_size, "Must have at least as many players as the sample size");

    // Generate four disjoint players
    let mut players: Vec<PlayerId> = Vec::new();
    for i in 0..sample_size {
        let mut random_player = PlayerId(rng.gen_range(0, player_count - i));
        let insertion_index = players.iter()
            .enumerate()
            .find_map(|(index, value)| {
                if random_player.0 >= value.0 {
                    random_player.0 += 1;
                    None
                } else {
                    Some(index)
                }
            })
            .unwrap_or(players.len());
        players.insert(insertion_index, random_player);
    }
    players
}

fn replace_random_player<R: rand::Rng>(players: &mut [PlayerId], rng: &mut R, player_count: usize) {
    let index = rng.gen_range(0, players.len());
    players.copy_within(index + 1..players.len(), index);
    let mut new_player = PlayerId(rng.gen_range(0, player_count - players.len() + 1));
    let insertion_index = players[..players.len() - 1].iter()
        .enumerate()
        .find_map(|(index, value)| {
            if new_player.0 >= value.0 {
                new_player.0 += 1;
                None
            } else {
                Some(index)
            }
        })
        .unwrap_or(players.len() - 1);
    //if insertion_index < players.len() - 1 {
    players.copy_within(insertion_index..players.len() - 1, insertion_index + 1);
    //}
    players[insertion_index] = new_player;
}

#[derive(Debug, Clone)]
pub struct SingleMatchPair {
    players: [PlayerId; 2],
}

#[derive(Debug, Clone)]
pub struct DoubleMatchPair {
    players: [PlayerId; 4],
}

pub trait MatchPair {
    fn new(players: &[PlayerId]) -> Option<Self> where Self: Sized;

    fn mutate<R: rand::Rng>(&mut self, rng: &mut R, num_players: usize);

    fn random<R: rand::Rng>(rng: &mut R, player_count: usize) -> Self;

    fn players(&self) -> &[PlayerId];

    fn player_count() -> usize;

    fn involves(&self, player: PlayerId) -> bool {
        self.players().iter().any(|p| *p == player)
    }
}

impl MatchPair for SingleMatchPair {
    /// Create a normalized pair where the lower
    fn new(players: &[PlayerId]) -> Option<Self> {
        if players.len() != 2 {
            return None;
        }
        if players[0] == players[1] {
            return None;
        }
        assert!(players[0] != players[1], "two distinct players are required for a match");
        Some(SingleMatchPair {
            players: [players[0].min(players[1]), players[0].max(players[1])]
        })
    }

    fn mutate<R: rand::Rng>(&mut self, rng: &mut R, player_count: usize) {
        replace_random_player(&mut self.players, rng, player_count);
    }

    fn players(&self) -> &[PlayerId] {
        &self.players
    }

    fn player_count() -> usize {
        2
    }

    fn random<R: rand::Rng>(rng: &mut R, player_count: usize) -> Self {
        let players = sample_players(rng, player_count, 2);
        SingleMatchPair::new(&players).unwrap()
    }
}

impl MatchPair for DoubleMatchPair {
    /// Create a normalized pair where the lower
    fn new(players: &[PlayerId]) -> Option<Self> {
        if players.len() != 4 {
            return None;
        }
        let mut players_arr = [PlayerId(0); 4];
        players_arr.copy_from_slice(players);
        players_arr.sort();

        let all_unique = players_arr.iter()
            .zip(players_arr.iter().skip(1))
            .all(|(p1, p2)| p1 != p2);
        if ! all_unique {
            return None;
        }

        Some(DoubleMatchPair {
            players: players_arr,
        })
    }

    fn mutate<R: rand::Rng>(&mut self, rng: &mut R, player_count: usize) {
        replace_random_player(&mut self.players, rng, player_count);
    }

    fn players(&self) -> &[PlayerId] {
        &self.players
    }

    fn player_count() -> usize {
        4
    }

    fn random<R: rand::Rng>(rng: &mut R, player_count: usize) -> Self {
        let players = sample_players(rng, player_count, 4);
        DoubleMatchPair::new(&players).unwrap()
    }
}