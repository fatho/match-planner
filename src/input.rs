use std::io::{BufReader, BufRead, Read};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    InvalidLine(usize),
    InvalidCall,
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::Io(err)
    }
}

#[derive(Debug)]
pub struct Player {
    name: String,
    availability: Vec<bool>,
}

impl Player {
    /// Parse a player definition such as
    ///
    /// John Doe   | __XX__XX_X_XX___
    pub fn parse(line: &str) -> Option<Self> {
        let mut parts = line.splitn(2, '|');
        let name = parts.next()?.trim();
        parts.next()?
            .trim()
            .chars()
            .map(|ch| match ch {
                '_' => Some(true),
                'X' => Some(false),
                _ => None,
            })
            .collect::<Option<_>>()
            .map(|availability| Player {
                name: name.to_owned(),
                availability: availability,
            })
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn availability(&self) -> &[bool] {
        self.availability.as_slice()
    }
}

#[derive(Debug)]
pub struct PlanningData {
    players: Vec<Player>,
    match_count: usize,
}

impl PlanningData {
    pub fn load<In: Read>(stream: In) -> Result<Self> {
        let mut players = Vec::new();

        let mut previous_match_count: Option<usize> = None;

        let reader = BufReader::new(stream);
        for (index, line_or_err) in reader.lines().enumerate() {
            let line: String = line_or_err?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let row = Player::parse(line.as_ref()).ok_or_else(|| Error::InvalidLine(index))?;
            let current_match_count = row.availability.len();

            let same_count = previous_match_count.map(|count| count == current_match_count).unwrap_or(true);
            previous_match_count = Some(current_match_count);

            if ! same_count {
                return Err(Error::InvalidLine(index));
            }

            players.push(row);
        }
        Ok(PlanningData {
            players: players,
            match_count: previous_match_count.unwrap_or(0),
        })
    }

    pub fn players(&self) -> &[Player] {
        self.players.as_slice()
    }

    pub fn match_count(&self) -> usize {
        self.match_count
    }
}
