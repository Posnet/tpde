//! Encoding snippet generator binary.
//!
//! This binary was previously in the tpde-encodegen crate.

use clap::Parser;
use inkwell::context::Context;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use tpde::encodegen;

#[derive(Parser)]
#[command(name = "encodegen")]
#[command(about = "Generate encoding snippets from LLVM IR", long_about = None)]
struct Cli {
    /// Input IR file (reads from stdin if not provided)
    input: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let ir_text = if let Some(input_file) = cli.input {
        fs::read_to_string(&input_file)?
    } else {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    let context = Context::create();
    match encodegen::parse_and_generate(&context, &ir_text) {
        Ok(tokens) => {
            for token in tokens {
                println!("{token}\n");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
