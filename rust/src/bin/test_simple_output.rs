use clap::Parser;
use std::path::PathBuf;
use tpde::test_ir::{TestIR, TestSpec};

#[derive(Parser)]
#[command(name = "test_simple_output")]
#[command(about = "Print TestIR from a .tir file", long_about = None)]
struct Cli {
    /// Input .tir file
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    let content = std::fs::read_to_string(&cli.input)?;
    let spec = TestSpec::parse(&content)?;
    let ir = TestIR::parse(&spec.tir_content)?;
    println!("{}", ir.print());
    
    Ok(())
}