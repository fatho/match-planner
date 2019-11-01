use std::fmt;
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

/// The various kinds of errors that can happen while reading input.
#[derive(Debug)]
pub enum Error {
    /// An I/O error (such as not being able to read the input file)
    Io(std::io::Error),
    /// An error while reading or writing CSV.
    Csv(csv::Error),
    /// The timetable file is not valid.
    InvalidTimetable {
        file: PathBuf,
        line: usize,
        error: TimetableError,
    },
}

#[derive(Debug)]
pub enum TimetableError {
    /// There are unknown and/or missing players in the previous timetable.
    PlayerMismatch,
    /// There are too many or too few players assigned on a match.
    InvalidPlayerCount,
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::Io(err)
    }
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Error {
        Error::Csv(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Io(err) => err.fmt(f),
            Error::Csv(err) => err.fmt(f),
            Error::InvalidTimetable { file, line, error } => {
                write!(f, "{}:{}: {}", file.to_string_lossy(), line, error)
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(error) => Some(error),
            Error::Csv(error) => Some(error),
            Error::InvalidTimetable { error, .. } => Some(error),
        }
    }
}

impl fmt::Display for TimetableError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TimetableError::PlayerMismatch => write!(
                f,
                "Players in time table do not match with players in planning data"
            ),
            TimetableError::InvalidPlayerCount => write!(f, "Match has too many players"),
        }
    }
}

impl std::error::Error for TimetableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
