use std::io;
use web_runtime::supervisor::{self, Config};

fn main() -> io::Result<()> {
    let config = Config::parse(std::env::args_os().skip(1))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    supervisor::run(config)
}
