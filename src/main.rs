use std::fs::File;
use std::path::Path;
use rand::distributions::Uniform;

mod input;
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

    let mut rng = rand::thread_rng();

    let population: Vec<Genome> = (0..1).into_iter()
        .map(|_| Genome::new_random(&mut rng, &input))
        .collect();

    println!("{:?}", population);

    Ok(())
}

#[derive(Debug)]
pub struct Genome {
    match_parings: Vec<MatchPair>,
}

impl Genome {
    pub fn new_random(rng: &mut impl rand::Rng, planning: &PlanningData) -> Self {
        let player_count = planning.players().len();
        let genome = (0..planning.match_count()).into_iter()
            .map(|_| MatchPair::random(rng, player_count))
            .collect();

        Genome {
            match_parings: genome,
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

#[derive(Debug)]
pub struct MatchPair {
    left: PlayerId,
    right: PlayerId,
}

impl MatchPair {
    pub fn random(rng: &mut impl rand::Rng, num_players: usize) -> Self {
        assert!(num_players > 1, "Must have more than one player for making a pair");

        let player1_distribution = Uniform::new(0, num_players);
        let player2_distribution = Uniform::new(0, num_players - 1);

        // Generate two disjoint players
        let player1 = rng.sample(player1_distribution);
        let mut player2 = rng.sample(player2_distribution);
        if player2 >= player1 {
            player2 += 1;
        }
        MatchPair::new(PlayerId(player1), PlayerId(player2))
    }

    /// Create a normalized pair where the lower 
    pub fn new(left: PlayerId, right: PlayerId) -> Self {
        assert!(left != right, "two distinct players are required for a match");
        MatchPair {
            left: left.min(right),
            right: left.max(right),
        }
    }
}
