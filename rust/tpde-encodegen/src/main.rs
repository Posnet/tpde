/// CLI entry point for snippet encoder generation.
///
/// The program expects an input LLVM IR file and optionally an output file
/// passed via `-o` or `--output`.  The generated source is written to stdout
/// when no output file is specified.
use std::{env, fs::File, io::{self, Read, Write}, process::exit};

use inkwell::{context::Context, memory_buffer::MemoryBuffer};

fn print_help(program: &str) {
    println!(
        "Usage: {program} <input> [-o <output>]\n\n\
Generate snippet encoders from LLVM IR.  Output defaults to stdout." );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = args.next();
                if output.is_none() {
                    eprintln!("missing argument for {}", arg);
                    let prog = env::args().next().unwrap_or_default();
                    print_help(&prog);
                    exit(1);
                }
            }
            "-h" | "--help" => {
                let prog = env::args().next().unwrap_or_default();
                print_help(&prog);
                return Ok(());
            }
            _ => {
                if input.is_none() {
                    input = Some(arg);
                } else {
                    eprintln!("unexpected argument: {}", arg);
                    let prog = env::args().next().unwrap_or_default();
                    print_help(&prog);
                    exit(1);
                }
            }
        }
    }

    let prog = env::args().next().unwrap_or_else(|| "tpde-encodegen".into());
    let input_path = match input {
        Some(p) => p,
        None => {
            print_help(&prog);
            exit(1);
        }
    };

    let ir = {
        let mut buf = Vec::new();
        File::open(&input_path)?.read_to_end(&mut buf)?;
        buf
    };

    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(&ir, &input_path);
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let generated = tpde_encodegen::generate(&module);

    match output {
        Some(path) => {
            let mut file = File::create(path)?;
            file.write_all(generated.as_bytes())?;
        }
        None => {
            io::stdout().write_all(generated.as_bytes())?;
        }
    }

    Ok(())
}
