use ndarray::{Array, Array1, Array2};
use std::iter;

/// In the match table, we store which players play in which matches.
/// The rows represent the matches, and the columns represent the players.
/// If an entry is true, it means that the player plays the given match.
///
/// This is the representation of one state in the search space.
#[derive(Clone)]
pub struct MatchTable {
    /// A table where the rows are matches and the columns are players.
    playing: Array2<bool>,
    /// Number of players per match. (Row sums)
    players_per_match: Array1<usize>,
    /// Number of matches per player. (Column sums)
    matches_per_player: Array1<usize>,
}

impl MatchTable {
    pub fn new(match_count: usize, player_count: usize) -> MatchTable {
        Self {
            playing: Array::from_elem((match_count, player_count), false),
            players_per_match: Array::zeros(match_count),
            matches_per_player: Array::zeros(player_count),
        }
    }

    pub fn from_table(playing: Array2<bool>) -> MatchTable {
        let players_per_match =
            playing.fold_axis(ndarray::Axis(1), 0, |count, elem| count + (*elem as usize));
        let matches_per_player =
            playing.fold_axis(ndarray::Axis(0), 0, |count, elem| count + (*elem as usize));

        Self {
            playing,
            players_per_match,
            matches_per_player,
        }
    }

    pub fn set_playing(&mut self, match_index: Match, player_index: Player, playing: bool) {
        let previous = self.playing[(match_index.0, player_index.0)];
        let previous_count = previous as usize;
        let new_count = playing as usize;

        self.playing[(match_index.0, player_index.0)] = playing;

        self.matches_per_player[player_index.0] += new_count;
        self.matches_per_player[player_index.0] -= previous_count;

        self.players_per_match[match_index.0] += new_count;
        self.players_per_match[match_index.0] -= previous_count;
    }

    pub fn is_playing(&self, match_index: Match, player_index: Player) -> bool {
        self.playing[(match_index.0, player_index.0)]
    }

    #[allow(unused)]
    pub fn players_per_match(&self, match_index: Match) -> usize {
        self.players_per_match[match_index.0]
    }

    pub fn matches_per_player(&self, player_index: Player) -> usize {
        self.matches_per_player[player_index.0]
    }

    pub fn match_count(&self) -> usize {
        self.playing.dim().0
    }

    pub fn player_count(&self) -> usize {
        self.playing.dim().1
    }

    #[allow(unused)]
    pub fn to_debug_tsv(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let (match_count, player_count) = self.playing.dim();

        for match_index in 0..match_count {
            write!(&mut out, "{:2} |", match_index).unwrap();
            for player_index in 0..player_count {
                write!(
                    &mut out,
                    "{}\t",
                    if self.playing[(match_index, player_index)] {
                        "1"
                    } else {
                        ""
                    }
                )
                .unwrap();
            }
            write!(&mut out, "| {}\n", self.players_per_match[match_index]).unwrap();
        }
        write!(
            &mut out,
            "------------------------------------------------------------------------------\n"
        )
        .unwrap();
        for player_index in 0..player_count {
            write!(&mut out, "{}\t", self.matches_per_player[player_index]).unwrap();
        }
        write!(&mut out, "\n").unwrap();

        out
    }
}

pub struct AvailabilityTable {
    /// A table where the rows are matches and the columns are players.
    /// An entry indicates if a player is a available for the match.
    availability: Array2<bool>,
    /// Number of players available on each match day.
    players_per_match: Array1<usize>,
    /// Number of match days where a player is available.
    matches_per_player: Array1<usize>,
}

impl AvailabilityTable {
    pub fn new(availability: Array2<bool>) -> Self {
        let players_per_match =
            availability.fold_axis(ndarray::Axis(1), 0, |count, elem| count + (*elem as usize));
        let matches_per_player =
            availability.fold_axis(ndarray::Axis(0), 0, |count, elem| count + (*elem as usize));

        Self {
            availability,
            players_per_match,
            matches_per_player,
        }
    }

    pub fn is_available(&self, match_index: Match, player_index: Player) -> bool {
        self.availability[(match_index.0, player_index.0)]
    }

    pub fn players_per_match(&self, match_index: Match) -> usize {
        self.players_per_match[match_index.0]
    }

    pub fn matches_per_player(&self, player_index: Player) -> usize {
        self.matches_per_player[player_index.0]
    }

    #[allow(unused)]
    pub fn to_debug_tsv(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let (match_count, player_count) = self.availability.dim();

        for match_index in 0..match_count {
            for player_index in 0..player_count {
                write!(
                    &mut out,
                    "{}\t",
                    if self.availability[(match_index, player_index)] {
                        "1"
                    } else {
                        ""
                    }
                )
                .unwrap();
            }
            write!(&mut out, "{}\n", self.players_per_match[match_index]).unwrap();
        }
        for player_index in 0..player_count {
            write!(&mut out, "{}\t", self.matches_per_player[player_index]).unwrap();
        }
        write!(&mut out, "\n").unwrap();

        out
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Player(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Match(pub usize);

/// Generate `num_starts` random solutions and then perform local search on each of those.
/// Return the best solution that was found.
#[allow(unused)]
pub fn multi_start_local_search(
    previous_matches: &MatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
    num_starts: usize,
) -> MatchTable {
    iter::repeat_with(|| {
        let mut solution = random_solution(availability, mode);
        let score = local_search(&mut solution, previous_matches, availability, mode);
        (score, solution)
    })
    .take(num_starts)
    .min_by_key(|(score, _solution)| *score)
    .expect("num_starts must be greate than zero")
    .1
}

/// Start with a random solution and perform local search until stuck in a local optimum.
/// Once a local optimum is reached, the solution is randomly perturbed and the search is retried.
/// The previous matches are used for evaluating the quality of a solution.
pub fn iterated_local_search(
    previous_matches: &MatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
) -> MatchTable {
    let (match_count, player_count) = availability.availability.dim();

    let mut solution = random_solution(availability, mode);

    let mut current_best = score(&solution, previous_matches, availability, mode);
    let mut best_solution = solution.clone();

    // The number of matches to perturb in the pertubation step
    let mut perturbation_size = 1;

    let mut matches_to_perturb: Vec<Match> = Vec::new();
    let mut random_players = Vec::with_capacity(mode.max_players());
    let mut rng = rand::thread_rng();

    loop {
        // Descend into local optimum
        let new_score = local_search(&mut solution, previous_matches, availability, mode);

        if new_score < current_best {
            // we found a better local optimum, reset perturbation size
            current_best = new_score;
            best_solution = solution.clone();
            perturbation_size = 1;
        } else if perturbation_size > match_count {
            // We increase the pertubation size several times to no avail, stop the search
            break;
        }

        // Perturb solution
        reservoir_sample_into(
            &mut rng,
            (0..match_count).map(Match),
            perturbation_size,
            &mut matches_to_perturb,
        );
        for match_index in matches_to_perturb.drain(..) {
            // Reset match
            for player_index in (0..player_count).map(Player) {
                solution.set_playing(match_index, player_index, false);
            }

            // Randomize match
            let available_players = (0..player_count)
                .map(Player)
                .filter(|player_index| availability.is_available(match_index, *player_index));

            reservoir_sample_into(
                &mut rng,
                available_players,
                mode.max_players(),
                &mut random_players,
            );

            for player_index in random_players.drain(..) {
                solution.set_playing(match_index, player_index, true);
            }
        }

        perturbation_size += 1;
    }

    best_solution
}

/// Generate a random match assignment within the given constraints.
pub fn random_solution(availability: &AvailabilityTable, mode: Mode) -> MatchTable {
    let (match_count, player_count) = availability.availability.dim();
    let mut table = MatchTable::new(match_count, player_count);
    let mut rng = rand::thread_rng();
    let mut chosen = Vec::with_capacity(mode.max_players());

    for match_index in (0..match_count).map(Match) {
        let available_players = (0..player_count)
            .map(Player)
            .filter(|player_index| availability.is_available(match_index, *player_index));

        reservoir_sample_into(&mut rng, available_players, mode.max_players(), &mut chosen);

        for player_index in chosen.drain(..) {
            table.set_playing(match_index, player_index, true);
        }
    }

    table
}

/// Perform local search on the match table for finding a local optimum.
/// Returns the score of the solution that was found.
pub fn local_search(
    table: &mut MatchTable,
    previous_matches: &MatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
) -> usize {
    use rand::RngCore;

    let mut current_best = score(table, previous_matches, availability, mode);

    let mut neighbour_vec: Vec<Neighbour> = Vec::new();

    let mut rng = rand::thread_rng();

    // This loop terminates because at every step we either decrease `current_best` or we stop the iteration.
    loop {
        // Gather neighbours
        neighbours(&table, &availability, mode, &mut neighbour_vec);

        let best_neighbour = neighbour_vec
            .drain(..)
            .map(|n| {
                let new_score = n.inspect(table, |table| {
                    score(table, previous_matches, availability, mode)
                });
                // whenever two solutions have the same score, pick one at random
                let tie_breaker = rng.next_u64();
                (new_score, tie_breaker, n)
            })
            .min_by_key(|(score, tie_breaker, _n)| (*score, *tie_breaker));

        if let Some((new_score, _, neighbour)) = best_neighbour {
            if new_score < current_best {
                neighbour.apply(table);
                current_best = new_score;
            } else {
                // even the best move was worse than what we had
                return current_best;
            }
        } else {
            // No more valid neighbours to explore
            return current_best;
        }
    }
}

/// Concatenate two match tables along the vertical axis (i.e. combining their matches).
/// Used for evaluating the quality of a solution given some earlier matches.
#[derive(Copy, Clone)]
struct ConcatenatedMatchTable<'a> {
    previous_matches: &'a MatchTable,
    current_matches: &'a MatchTable,
}

impl<'a> ConcatenatedMatchTable<'a> {
    fn dim(&self) -> (usize, usize) {
        let (match_count1, player_count1) = self.previous_matches.playing.dim();
        let (match_count2, player_count2) = self.current_matches.playing.dim();

        assert_eq!(player_count1, player_count2);
        (match_count1 + match_count2, player_count1)
    }

    /// Iterate over all rows (i.e. matches) in the given range.
    fn iter_rows(
        &self,
        range: std::ops::Range<usize>,
    ) -> impl Iterator<Item = ndarray::ArrayView1<bool>> {
        let (match_count1, _player_count1) = self.previous_matches.playing.dim();

        range.map(move |row_index| {
            if row_index < match_count1 {
                self.previous_matches
                    .playing
                    .index_axis(ndarray::Axis(0), row_index)
            } else {
                self.current_matches
                    .playing
                    .index_axis(ndarray::Axis(0), row_index - match_count1)
            }
        })
    }

    pub fn matches_per_player(&self, player_index: Player) -> usize {
        self.previous_matches.matches_per_player(player_index)
            + self.current_matches.matches_per_player(player_index)
    }
}

/// Compute the "score" of a valid state according to the following criteria:
///
/// 1. all players should have approximately equal intervals between their matches
/// 2. all players should play with as many other players as possible
/// 3. all players should play an approximatly equal number of matches
///
/// A lower score is better.
pub fn score(
    current_matches: &MatchTable,
    previous_matches: &MatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
) -> usize {
    score_impl(
        ConcatenatedMatchTable {
            previous_matches,
            current_matches,
        },
        availability,
        mode,
    )
}

fn score_impl(
    table: ConcatenatedMatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
) -> usize {
    let (match_count, player_count) = table.dim();

    // 2. all players should play with as many other players as possible

    // TODO: O(n^2) alert, optimize later

    let duplicate_match_assignments: usize = table
        .iter_rows(0..match_count - 1)
        .enumerate()
        .map(|(index1, row1)| {
            // count how many of the following rows (= matches) are equal to this row
            table
                .iter_rows(index1 + 1..match_count)
                .map(|row2| if row1 == row2 { 1 } else { 0 })
                .sum::<usize>()
        })
        .sum();

    // 3. all players should play an approximatly equal number of matches
    let expected_match_count = (match_count * mode.max_players()) / player_count;

    let match_count_inequality: usize = (0..player_count)
        .map(Player)
        .map(|player_index| {
            let available = availability.matches_per_player(player_index);
            let actual = table.matches_per_player(player_index);

            if actual < available.min(expected_match_count) {
                // Increase score when the player plays fewer times than what we would expect on average
                // By squaring the distance from the average we make sure that we can move a player
                // that is further away from the average closer to the average at the expense of
                // a player that is already closer to the average.
                (expected_match_count - actual).pow(2)
            } else {
                // Don't count it, if
                // - a player plays more often than the average, as there has to be another player that
                //   plays fewer times, so we're already counting this case.
                // - a player is limited by their availability, so it's their own fault
                0
            }
        })
        .sum();

    // 1. all players should have approximately equal intervals between their matches

    // Measure deviation in `match/MATCH_DEVIATION_ACCURACY` matches
    const MATCH_DEVIATION_ACCURACY: usize = 10;

    let match_interval_deviation: usize = (0..player_count)
        .map(Player)
        .map(|player_index| {
            let num_matches = table.matches_per_player(player_index);
            // Given the number of times we expect the player to play, how many days should be between each match
            let expected_interval = match_count * MATCH_DEVIATION_ACCURACY / (num_matches + 1);

            // For the deviation we only look at the new matches, as the old once won't influence it
            table
                .current_matches
                .playing
                .index_axis(ndarray::Axis(1), player_index.0)
                .iter()
                .enumerate()
                .fold(
                    (None, 0),
                    |(last_match, total_deviation), (match_index, playing_current)| {
                        if *playing_current {
                            // If the match is played, count the deviation.
                            let deviation = if let Some(last_match) = last_match {
                                let since_last =
                                    (match_index - last_match) * MATCH_DEVIATION_ACCURACY;
                                if since_last < expected_interval {
                                    expected_interval - since_last
                                } else {
                                    since_last - expected_interval
                                }
                            } else {
                                // If it's the first match played so far, ignore the deviation
                                0
                            };
                            (Some(match_index), total_deviation + deviation)
                        } else {
                            // Skip matches that are not played
                            if availability.is_available(Match(match_index), player_index) {
                                (last_match, total_deviation)
                            } else {
                                // If someone is not available, pretend that day didn't happen
                                (last_match.map(|x| x + 1), total_deviation)
                            }
                        }
                    },
                )
                .1
        })
        .sum::<usize>()
        / MATCH_DEVIATION_ACCURACY;

    // ensure that inequality is optimized first, and then the remaining criteria
    match_count_inequality * 100 + duplicate_match_assignments + match_interval_deviation
}

/// Enumerate the valid neighbours of some state under the given availability constraints into the given vector.
pub fn neighbours(
    table: &MatchTable,
    availability: &AvailabilityTable,
    mode: Mode,
    neighbours: &mut Vec<Neighbour>,
) {
    let (match_count, player_count) = table.playing.dim();

    // Iterate over all matches
    for match_index in (0..match_count).map(Match) {
        // Skip matches where there is no choice
        if availability.players_per_match(match_index) <= mode.max_players() {
            continue;
        }

        // pair each selected player with an available non-selected player
        neighbours.extend(
            (0..player_count)
                .map(Player)
                .filter(|player_index| table.is_playing(match_index, *player_index))
                .flat_map(|selected_index| {
                    (0..player_count)
                        .map(Player)
                        .filter(|player_index| {
                            !table.is_playing(match_index, *player_index)
                                && availability.is_available(match_index, *player_index)
                        })
                        .map(move |available_index| Neighbour {
                            match_index,
                            player_out: selected_index,
                            player_in: available_index,
                        })
                }),
        );
    }
}

/// A neighbour of a state in the search space, represented as the entry that needs to be flipped.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Neighbour {
    /// The match that is changed.
    match_index: Match,
    /// The player being removed from the match.
    player_out: Player,
    /// The player being inserted into the match.
    player_in: Player,
}

impl Neighbour {
    /// Move to the neighbouring state, assuming that `table` is the state from which the neighbour was generated.
    /// If the state was changed in the meantime, the results are undefined.
    pub fn apply(&self, table: &mut MatchTable) {
        table.set_playing(self.match_index, self.player_in, true);
        table.set_playing(self.match_index, self.player_out, false);
    }

    /// Move back to the previous state. May only be called if `table` is in the state that was reached by calling
    /// `apply` first.
    pub fn unapply(&self, table: &mut MatchTable) {
        table.set_playing(self.match_index, self.player_in, false);
        table.set_playing(self.match_index, self.player_out, true);
    }

    /// Apply the neighbour transition to the table, evaluate the callback and undo the change again.
    pub fn inspect<R, F>(&self, table: &mut MatchTable, inner: F) -> R
    where
        F: FnOnce(&mut MatchTable) -> R,
    {
        self.apply(table);
        let result = inner(table);
        self.unapply(table);
        result
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Single,
    Double,
}

impl Mode {
    pub fn max_players(self) -> usize {
        match self {
            Mode::Single => 2,
            Mode::Double => 4,
        }
    }
}

impl std::str::FromStr for Mode {
    type Err = ParseModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single" => Ok(Mode::Single),
            "double" => Ok(Mode::Double),
            _ => Err(ParseModeError),
        }
    }
}

#[derive(Debug)]
pub struct ParseModeError;

impl std::fmt::Display for ParseModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid mode value, must be `single` or `double`")
    }
}

/// Select a uniformly chosen element from the items of the given iterator.
#[allow(unused)]
pub fn reservoir_sampling_single<R: rand::Rng, I: Iterator>(
    rng: &mut R,
    items: I,
) -> Option<I::Item> {
    let mut chosen: Option<I::Item> = None;
    for (index, item) in items.enumerate() {
        if rng.gen_range(0, index + 1) == 0 {
            chosen = Some(item);
        }
    }
    chosen
}

/// Select `k` uniformly chosen elements from the items of the given iterator and push them onto the given vector.
/// Existing elements in the vector are preserved.
pub fn reservoir_sample_into<R: rand::Rng, I: Iterator>(
    rng: &mut R,
    mut items: I,
    k: usize,
    chosen: &mut Vec<I::Item>,
) {
    let base = chosen.len();

    // always pick the first `k` items
    for _ in 0..k {
        if let Some(item) = items.next() {
            chosen.push(item);
        } else {
            return;
        }
    }

    // then pick the rest
    for (index, item) in items.enumerate() {
        let overwrite = rng.gen_range(0, k + 1 + index);
        if overwrite < chosen.len() {
            chosen[base + overwrite] = item;
        }
    }
}
