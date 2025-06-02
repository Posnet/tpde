/// CLI entry point for snippet encoder generation.
///
/// This parses an LLVM IR file and writes the generated Rust snippets
/// to the provided output path. If no output is specified the snippets
/// are printed to stdout.
use std::{env, fs};

use inkwell::context::Context;

fn usage() {
    eprintln!("usage: tpde-encodegen <input.ll> [output.rs]");
}

fn main() {
    let mut args = env::args().skip(1);
    let input = match args.next() {
        Some(p) => p,
        None => {
            usage();
            return;
        }
    };
    let output = args.next();

    let ir = fs::read_to_string(&input).expect("failed to read input");
    let context = Context::create();
    let tokens = match tpde_encodegen::parse_and_generate(&context, &ir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("parse error: {}", e);
            return;
        }
    };

    if let Some(out) = output {
        fs::write(out, tokens.join("\n")).expect("failed to write output");
    } else {
        for t in tokens {
            println!("{}", t);
        }
    }
}
