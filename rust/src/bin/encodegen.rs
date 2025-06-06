//! Encoding snippet generator binary.
//!
//! This binary was previously in the tpde-encodegen crate.

use inkwell::context::Context;
use std::env;
use std::fs;
use std::io::{self, Read};
use tpde::encodegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    let ir_text = if args.len() > 1 {
        // Read from file
        fs::read_to_string(&args[1])?
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
                println!("{}\n", token);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}