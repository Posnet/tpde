//! Test IR (TIR) parser and data structures for testing TPDE compilation.
//!
//! This module provides a simple IR format for writing compiler tests
//! without depending on full LLVM IR. The format is designed to be:
//! - Human-readable and writable
//! - Easy to parse
//! - Sufficient for testing compilation patterns
//!
//! # TIR Format
//!
//! ```text
//! ; Comments start with semicolon
//! func_name(%arg1, %arg2) {
//! entry:
//!     %val = add %arg1, %arg2
//!     br ^next_block
//! next_block:
//!     terminate
//! }
//! ```

pub mod adaptor;
pub mod check;
pub mod compiler;
pub mod parser;

pub use adaptor::TestIRAdaptor;
pub use check::{CheckDirective, TestRunner, TestSpec};
pub use compiler::{CompilationError, TestIRCompiler};

#[derive(Debug, Clone, PartialEq)]
pub struct TestIR {
    pub functions: Vec<Function>,
    pub blocks: Vec<Block>,
    pub values: Vec<Value>,
    pub value_operands: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub declaration: bool,
    pub local_only: bool,
    pub has_call: bool,
    pub block_begin_idx: u32,
    pub block_end_idx: u32,
    pub arg_begin_idx: u32,
    pub arg_end_idx: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: String,
    pub succ_begin_idx: u32,
    pub succ_end_idx: u32,
    pub inst_begin_idx: u32,
    pub phi_end_idx: u32,
    pub inst_end_idx: u32,
    pub block_info: u32,
    pub block_info2: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    pub value_type: ValueType,
    pub op: Operation,
    pub force_fixed_assignment: bool,
    /// For call only: called function index
    pub call_func_idx: u32,
    /// Number of value operands
    pub op_count: u32,
    /// Operand indices into value_operands array
    pub op_begin_idx: u32,
    pub op_end_idx: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Normal,
    Arg,
    Phi,
    Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    None,
    Any,
    Add,
    Sub,
    Alloca,
    Terminate,
    Ret,
    Br,
    CondBr,
    Tbz,
    Jump,
    Call,
    ZeroFill,
}

impl Operation {
    pub const fn info(self) -> OpInfo {
        use Operation::*;
        match self {
            None => OpInfo {
                name: "<none>",
                is_terminator: false,
                is_def: false,
                op_count: 0,
                succ_count: 0,
                imm_count: 0,
            },
            Any => OpInfo {
                name: "any",
                is_terminator: false,
                is_def: true,
                op_count: !0,
                succ_count: 0,
                imm_count: 0,
            },
            Add => OpInfo {
                name: "add",
                is_terminator: false,
                is_def: true,
                op_count: 2,
                succ_count: 0,
                imm_count: 0,
            },
            Sub => OpInfo {
                name: "sub",
                is_terminator: false,
                is_def: true,
                op_count: 2,
                succ_count: 0,
                imm_count: 0,
            },
            Alloca => OpInfo {
                name: "alloca",
                is_terminator: false,
                is_def: true,
                op_count: 0,
                succ_count: 0,
                imm_count: 2,
            },
            Terminate => OpInfo {
                name: "terminate",
                is_terminator: true,
                is_def: false,
                op_count: 0,
                succ_count: 0,
                imm_count: 0,
            },
            Ret => OpInfo {
                name: "ret",
                is_terminator: true,
                is_def: false,
                op_count: 1,
                succ_count: 0,
                imm_count: 0,
            },
            Br => OpInfo {
                name: "br",
                is_terminator: true,
                is_def: false,
                op_count: 0,
                succ_count: 1,
                imm_count: 0,
            },
            CondBr => OpInfo {
                name: "condbr",
                is_terminator: true,
                is_def: false,
                op_count: 1,
                succ_count: 2,
                imm_count: 0,
            },
            Tbz => OpInfo {
                name: "tbz",
                is_terminator: true,
                is_def: false,
                op_count: 1,
                succ_count: 2,
                imm_count: 1,
            },
            Jump => OpInfo {
                name: "jump",
                is_terminator: true,
                is_def: false,
                op_count: 0,
                succ_count: !0,
                imm_count: 0,
            },
            Call => OpInfo {
                name: "call",
                is_terminator: false,
                is_def: true,
                op_count: !0,
                succ_count: 0,
                imm_count: 0,
            },
            ZeroFill => OpInfo {
                name: "zerofill",
                is_terminator: false,
                is_def: false,
                op_count: 0,
                succ_count: 0,
                imm_count: 1,
            },
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "any" => Some(Operation::Any),
            "add" => Some(Operation::Add),
            "sub" => Some(Operation::Sub),
            "alloca" => Some(Operation::Alloca),
            "terminate" => Some(Operation::Terminate),
            "ret" => Some(Operation::Ret),
            "br" => Some(Operation::Br),
            "condbr" => Some(Operation::CondBr),
            "tbz" => Some(Operation::Tbz),
            "jump" => Some(Operation::Jump),
            "call" => Some(Operation::Call),
            "zerofill" => Some(Operation::ZeroFill),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OpInfo {
    pub name: &'static str,
    pub is_terminator: bool,
    pub is_def: bool,
    pub op_count: u32,
    pub succ_count: u32,
    pub imm_count: u32,
}

impl TestIR {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            blocks: Vec::new(),
            values: Vec::new(),
            value_operands: Vec::new(),
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        parser::parse_ir(text)
    }

    pub fn print(&self) -> String {
        use std::fmt::Write;
        let mut output = String::new();
        writeln!(output, "Printing IR").unwrap();

        for func in &self.functions {
            // Print function header
            let func_type = if func.declaration {
                "Extern function"
            } else if func.local_only {
                "Local function"
            } else {
                "Function"
            };
            writeln!(output, "\n{} {}", func_type, func.name).unwrap();

            // Print arguments
            for arg_idx in func.arg_begin_idx..func.arg_end_idx {
                let arg = &self.values[arg_idx as usize];
                writeln!(output, "Argument {}", arg.name).unwrap();
            }

            // Skip blocks for declarations
            if func.declaration {
                continue;
            }

            // Print blocks
            for block_idx in func.block_begin_idx..func.block_end_idx {
                let block = &self.blocks[block_idx as usize];
                writeln!(output, "Block {}", block.name).unwrap();

                // Print successors
                for succ_idx in block.succ_begin_idx..block.succ_end_idx {
                    let succ_block_idx = self.value_operands[succ_idx as usize];
                    if let Some(succ_block) = self.blocks.get(succ_block_idx as usize) {
                        writeln!(output, "Succ {}", succ_block.name).unwrap();
                    }
                }

                // Print PHIs
                for inst_idx in block.inst_begin_idx..block.phi_end_idx {
                    self.print_phi(&mut output, inst_idx as usize);
                }

                // Print instructions
                for inst_idx in block.phi_end_idx..block.inst_end_idx {
                    self.print_instruction(&mut output, inst_idx as usize);
                }
            }
        }

        output
    }

    fn print_phi(&self, output: &mut String, inst_idx: usize) {
        use std::fmt::Write;
        let phi = &self.values[inst_idx];
        writeln!(output, "PHI {}", phi.name).unwrap();

        // Print incoming values
        let incoming_count = phi.op_count;
        for i in 0..incoming_count {
            let val_idx = self.value_operands[(phi.op_begin_idx + i) as usize];
            let block_idx = self.value_operands[(phi.op_begin_idx + incoming_count + i) as usize];

            if let (Some(val), Some(block)) = (
                self.values.get(val_idx as usize),
                self.blocks.get(block_idx as usize),
            ) {
                writeln!(output, "{} from {}", val.name, block.name).unwrap();
            }
        }
    }

    fn print_instruction(&self, output: &mut String, inst_idx: usize) {
        use std::fmt::Write;
        let inst = &self.values[inst_idx];
        let info = inst.op.info();

        // Print instruction header
        if info.is_def {
            writeln!(output, "Value {} ({})", inst.name, info.name).unwrap();
        } else {
            writeln!(output, "Value ({})", info.name).unwrap();
        }

        // Special handling for call instructions
        if inst.op == Operation::Call {
            // TODO: Once we track function references, print actual function name
            writeln!(output, "Target ext_func2").unwrap(); // Placeholder
        }

        // Print value operands
        for op_idx in 0..inst.op_count {
            let operand_idx = self.value_operands[(inst.op_begin_idx + op_idx) as usize];
            if let Some(operand) = self.values.get(operand_idx as usize) {
                writeln!(output, "Op {}", operand.name).unwrap();
            }
        }

        // Print block operands (for branches)
        let block_op_start = inst.op_begin_idx + inst.op_count;
        let block_op_count = if info.succ_count == !0 {
            // Variable successors for jump
            inst.op_end_idx - block_op_start
        } else {
            info.succ_count
        };

        for i in 0..block_op_count {
            let block_idx = self.value_operands[(block_op_start + i) as usize];
            if let Some(target_block) = self.blocks.get(block_idx as usize) {
                writeln!(output, "Op ^{}", target_block.name).unwrap();
            }
        }

        // Print immediates
        let imm_start = block_op_start
            + if info.succ_count == !0 {
                0
            } else {
                info.succ_count
            };
        for i in 0..info.imm_count {
            if let Some(&imm) = self.value_operands.get((imm_start + i) as usize) {
                writeln!(output, "Op ${}", imm).unwrap();
            }
        }
    }
}

impl Default for TestIR {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TestIR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.print())
    }
}
