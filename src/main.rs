use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

// submodules
mod errors;
mod input;
mod local_search;
use input::{PlanningData, Player};

/// The command line options that can be given to this application.
#[derive(Debug, StructOpt)]
#[structopt(
    name = "match-planner",
    about = "A planning application for assigning players to matches."
)]
struct Opt {
    /// Input file, stdin if not present
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file, stdout if not present
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,

    /// Previous plannings that should be continued
    #[structopt(short = "p", long = "past", parse(from_os_str))]
    past: Vec<PathBuf>,

    /// Whether to plan double or single matches.
    #[structopt(short, long)]
    mode: local_search::Mode,

    /// Quiet mode, do not print anything to stderr
    #[structopt(short = "q", long = "quiet")]
    quiet: bool,
}

fn main() -> ! {
    let opt = Opt::from_args();

    match run(opt) {
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(1)
        }
        Ok(_) => std::process::exit(0),
    }
}

fn run(opt: Opt) -> errors::Result<()> {
    let input = match opt.input {
        None => PlanningData::load(std::io::stdin())?,
        Some(file_name) => {
            let file = File::open(&file_name)?;
            PlanningData::load(file)?
        }
    };

    let mut log_out: Box<dyn Write> = if opt.quiet {
        Box::new(NullWrite)
    } else {
        Box::new(std::io::stderr())
    };

    // Setup planning constraints
    let availability = make_availability_table(&input);
    let mode = opt.mode;
    let past_match_table = load_past_matches(opt.past.iter(), mode, &input.players())?;

    writeln!(log_out, "Number of players: {}", input.players().len())?;
    writeln!(log_out, "Number of matches: {}", input.match_count())?;
    writeln!(
        log_out,
        "Number of past matches: {}",
        past_match_table.match_count()
    )?;
    writeln!(
        log_out,
        "--------------------------------------------------"
    )?;

    let solution = local_search::iterated_local_search(&past_match_table, &availability, mode);

    match opt.output {
        None => print_solution(std::io::stdout(), input.players(), &solution)?,
        Some(file_name) => {
            let file = File::create(&file_name)?;
            print_solution(file, input.players(), &solution)?;
        }
    }

    Ok(())
}

fn load_past_matches<P: AsRef<Path>, I: Iterator<Item = P>>(
    previous_match_files: I,
    mode: local_search::Mode,
    players: &[Player],
) -> errors::Result<local_search::MatchTable> {
    let mut past_matches = Vec::new();
    for previous_file in previous_match_files {
        read_previous_file_into(previous_file.as_ref(), mode, players, &mut past_matches)?;
    }
    let table = if past_matches.is_empty() {
        local_search::MatchTable::new(0, players.len())
    } else {
        let past_match_table = {
            let views: Vec<_> = past_matches
                .iter()
                .map(|row| row.view().into_shape((1, players.len())).unwrap())
                .collect();
            ndarray::stack(ndarray::Axis(0), &views).unwrap()
        };
        local_search::MatchTable::from_table(past_match_table)
    };
    Ok(table)
}

fn read_previous_file_into(
    filename: &Path,
    mode: local_search::Mode,
    players: &[Player],
    rows: &mut Vec<ndarray::Array1<bool>>,
) -> errors::Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(filename)?;

    // Headers must match
    if !reader
        .headers()?
        .iter()
        .eq(players.iter().map(Player::name))
    {
        return Err(errors::Error::InvalidTimetable {
            file: filename.to_path_buf(),
            line: 1,
            error: errors::TimetableError::PlayerMismatch,
        });
    }

    for (record_num, record) in reader.into_records().enumerate() {
        use std::iter::FromIterator;

        // find the ones in the row
        let row = ndarray::Array::from_iter(record?.iter().map(|column| column == "1"));
        let num_players: usize = row.iter().map(|playing| *playing as usize).sum();

        if num_players <= mode.max_players() {
            rows.push(row);
        } else {
            return Err(errors::Error::InvalidTimetable {
                file: filename.to_path_buf(),
                line: record_num + 2,
                error: errors::TimetableError::InvalidPlayerCount,
            });
        }
    }

    Ok(())
}

/// Build the availability table used for local search from the parsed inpput
fn make_availability_table(input: &PlanningData) -> local_search::AvailabilityTable {
    let mut availability =
        ndarray::Array::from_elem((input.match_count(), input.players().len()), false);
    for (player_index, player) in input.players().iter().enumerate() {
        for (match_index, available) in player.availability().iter().enumerate() {
            availability[(match_index, player_index)] = *available;
        }
    }
    local_search::AvailabilityTable::new(availability)
}

fn print_solution<W: Write>(
    out: W,
    players: &[Player],
    assignment: &local_search::MatchTable,
) -> errors::Result<()> {
    let mut writer = csv::WriterBuilder::new().delimiter(b'\t').from_writer(out);

    // Header, consists of the player names
    writer.write_record(players.iter().map(|p| p.name()))?;

    // Rows, one for each match day. Contains a 1 in the columns of the two players playing on that day.
    for match_index in (0..assignment.match_count()).map(local_search::Match) {
        writer.write_record(
            (0..assignment.player_count())
                .map(local_search::Player)
                .map(|player_index| {
                    if assignment.is_playing(match_index, player_index) {
                        "1"
                    } else {
                        ""
                    }
                }),
        )?;
    }
    Ok(())
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
