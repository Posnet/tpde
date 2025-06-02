/// CLI entry point for snippet encoder generation.
///
/// This parses an LLVM IR file and writes the generated Rust snippets
/// to the provided output path. If no output is specified the snippets
/// are printed to stdout.
use std::{env, fs};

use inkwell::context::Context;

fn main() {
    let mut args = env::args().skip(1);
    let input = args.next().expect("missing input file");
    let output = args.next();

    let ir = fs::read_to_string(&input).expect("failed to read input");
    let context = Context::create();
    let module = tpde_encodegen::parse_module(&context, &ir).expect("parse error");
    let tokens = tpde_encodegen::generate_tokens(&module);

    if let Some(out) = output {
        fs::write(out, tokens.join("\n")).expect("failed to write output");
    } else {
        for t in tokens {
            println!("{}", t);
        }
    }
}
