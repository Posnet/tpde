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
    let module = match tpde_encodegen::parse_module(&context, &ir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse error: {}", e);
            return;
        }
    };

    let output_code = tpde_encodegen::generate(&module);

    if let Some(out) = output {
        fs::write(out, &output_code).expect("failed to write output");
    } else {
        print!("{}", output_code);
    }
}
