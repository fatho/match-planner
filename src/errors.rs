use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// The various kinds of errors that can happen while reading input.
#[derive(Debug)]
pub enum Error {
    /// An I/O error (such as not being able to read the input file)
    Io(std::io::Error),
    /// An error while reading or writing CSV.
    Csv(csv::Error),
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
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            Error::Csv(err) => Some(err),
        }
    }
}
