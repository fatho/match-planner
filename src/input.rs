use std::io::Read;

use crate::errors::Result;

/// A player participating in matches.
#[derive(Debug)]
pub struct Player {
    /// The name of the player, used in the output file.
    name: String,
    /// The availability of a player given consecutively for all match dates.
    /// If true, the player is available, otherwise, the player is unavailable
    /// and cannot play.
    availability: Vec<bool>,
}

impl Player {
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn availability(&self) -> &[bool] {
        self.availability.as_slice()
    }
}

/// The input to the planning algorithm. Defines which players partake in the matches,
/// and how many matches there are. The number of matches must be equal to the length
/// of the availability vectors of all players.
#[derive(Debug)]
pub struct PlanningData {
    players: Vec<Player>,
    match_count: usize,
}

impl PlanningData {
    /// Read the planning data from an input stream.
    pub fn load<In: Read>(stream: In) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(stream);

        // Each column corresponds to a player, the column header is the players name.
        let mut players: Vec<_> = reader.headers()?.iter()
            .map(|name| Player {
                name: name.to_owned(),
                availability: Vec::new(),
            })
            .collect();

        let mut match_count: usize = 0;

        // Each row corresponds to a match day, the column values indicates whether a
        // player is available or not.
        for record in reader.into_records() {
            for (player, availability) in players.iter_mut().zip(record?.iter()) {
                // Player is available if there's no X in the column
                player.availability.push(!(availability == "x" || availability == "X"));
            }
            match_count += 1;
        }

        Ok(PlanningData {
            players: players,
            match_count: match_count,
        })
    }

    pub fn player_index_by_name(&self, name: &str) -> Option<usize> {
        self.players.iter().position(|player| player.name() == name)
    }

    pub fn players(&self) -> &[Player] {
        self.players.as_slice()
    }

    pub fn match_count(&self) -> usize {
        self.match_count
    }
}
