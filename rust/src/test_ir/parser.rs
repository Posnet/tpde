//! TIR (Test IR) parser implementation.

use super::*;
use std::collections::HashMap;

pub fn parse_ir(text: &str) -> Result<TestIR, String> {
    let parser = Parser::new(text);
    parser.parse()
}

struct Parser<'a> {
    text: &'a str,
    pos: usize,
    ir: TestIR,

    // Global maps
    funcs: HashMap<&'a str, u32>,
    func_resolves: Vec<Resolve<'a>>,

    // Per-function maps
    blocks: HashMap<&'a str, u32>,
    values: HashMap<&'a str, u32>,
    block_resolves: Vec<Resolve<'a>>,
    value_resolves: Vec<Resolve<'a>>,
}

#[derive(Debug)]
struct Resolve<'a> {
    name: &'a str,
    index: u32,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            pos: 0,
            ir: TestIR::new(),
            funcs: HashMap::new(),
            func_resolves: Vec::new(),
            blocks: HashMap::new(),
            values: HashMap::new(),
            block_resolves: Vec::new(),
            value_resolves: Vec::new(),
        }
    }

    fn parse(mut self) -> Result<TestIR, String> {
        self.skip_whitespace(true);

        while !self.is_eof() {
            match self.parse_function() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error parsing function at position {}: {}", self.pos, e);
                    if self.pos < self.text.len() {
                        let context_start = self.pos.saturating_sub(20);
                        let context_end = (self.pos + 20).min(self.text.len());
                        eprintln!("Context: '{}'", &self.text[context_start..context_end]);
                        eprintln!("         {}^", " ".repeat(self.pos - context_start));
                    }
                    return Err(e);
                }
            }
            self.skip_whitespace(true);
        }

        // Resolve all references
        self.resolve_all_references()?;

        Ok(self.ir)
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn current_char(&self) -> Option<char> {
        self.text.chars().nth(self.pos)
    }

    fn advance(&mut self) {
        if let Some(ch) = self.current_char() {
            self.pos += ch.len_utf8();
        }
    }

    fn skip_whitespace(&mut self, skip_newlines: bool) {
        while let Some(ch) = self.current_char() {
            if ch == ';' {
                // Skip comment line
                while let Some(ch) = self.current_char() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else if ch.is_whitespace() {
                if ch == '\n' && !skip_newlines {
                    break;
                }
                self.advance();
            } else {
                break;
            }
        }
    }

    fn try_read(&mut self, ch: char) -> bool {
        self.skip_whitespace(true);
        if self.current_char() == Some(ch) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        if !self.try_read(ch) {
            return Err(format!(
                "Expected '{}' but found {:?}",
                ch,
                self.current_char()
            ));
        }
        Ok(())
    }

    fn read_identifier(&mut self) -> Result<&'a str, String> {
        self.skip_whitespace(true);
        let start = self.pos;

        if let Some(ch) = self.current_char() {
            if !ch.is_alphabetic() {
                return Err(format!("Expected identifier but found '{}'", ch));
            }
        } else {
            return Err("Expected identifier but found EOF".to_string());
        }

        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let result = &self.text[start..self.pos];
        if result.is_empty() {
            return Err("Expected identifier but got empty string".to_string());
        }
        Ok(result)
    }

    fn read_value_name(&mut self) -> Result<(&'a str, bool), String> {
        self.skip_whitespace(true);
        self.expect('%')?;
        let name = self.read_identifier()?;

        // Check for force fixed assignment marker
        let force_fixed = if self.current_char() == Some('!') {
            self.advance();
            true
        } else {
            false
        };

        Ok((name, force_fixed))
    }

    fn read_block_name(&mut self) -> Result<&'a str, String> {
        self.skip_whitespace(true);
        self.expect('^')?;
        self.read_identifier()
    }

    fn read_number(&mut self) -> Result<u32, String> {
        self.skip_whitespace(true);
        let start = self.pos;

        // Check for hex prefix
        let is_hex = if self.current_char() == Some('0') {
            self.advance();
            if self.current_char() == Some('x') || self.current_char() == Some('X') {
                self.advance();
                true
            } else {
                // Back to the '0' we just consumed
                false
            }
        } else {
            false
        };

        // Read digits
        if is_hex {
            while let Some(ch) = self.current_char() {
                if ch.is_ascii_hexdigit() {
                    self.advance();
                } else {
                    break;
                }
            }
        } else {
            // If we didn't consume a '0' prefix, or if we did but it wasn't hex
            if !is_hex && start == self.pos {
                // Read decimal digits
                while let Some(ch) = self.current_char() {
                    if ch.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        if start == self.pos || (is_hex && self.pos == start + 2) {
            return Err("Expected number".to_string());
        }

        let number_str = &self.text[start..self.pos];
        if is_hex {
            // Parse hex without the "0x" prefix
            u32::from_str_radix(&number_str[2..], 16)
                .map_err(|e| format!("Failed to parse hex number: {}", e))
        } else {
            number_str
                .parse()
                .map_err(|e| format!("Failed to parse number: {}", e))
        }
    }

    fn parse_function(&mut self) -> Result<(), String> {
        let func_name = self.read_identifier()?;
        let func_idx = self.ir.functions.len() as u32;

        // Check for duplicate function names
        if self.funcs.contains_key(func_name) {
            return Err(format!("Duplicate function definition: '{}'", func_name));
        }

        // Reset per-function state
        self.blocks.clear();
        self.values.clear();
        self.block_resolves.clear();
        self.value_resolves.clear();

        // Parse arguments
        self.expect('(')?;
        let arg_begin_idx = self.ir.values.len() as u32;

        while !self.try_read(')') {
            let (arg_name, _) = self.read_value_name()?;
            let arg_idx = self.ir.values.len() as u32;

            self.values.insert(arg_name, arg_idx);
            self.ir.values.push(Value {
                name: arg_name.to_string(),
                value_type: ValueType::Arg,
                op: Operation::None,
                force_fixed_assignment: false,
                call_func_idx: 0,
                op_count: 0,
                op_begin_idx: 0,
                op_end_idx: 0,
            });

            if !self.try_read(',') && self.current_char() != Some(')') {
                return Err("Expected ',' or ')' in argument list".to_string());
            }
        }

        let arg_end_idx = self.ir.values.len() as u32;

        // Check for function declaration (ends with !)
        let is_declaration = self.try_read('!');
        let mut is_local = false;

        // If not a declaration, check for modifiers and body
        if !is_declaration {
            // Check for "local" modifier
            self.skip_whitespace(true);
            let saved_pos = self.pos;
            match self.read_identifier() {
                Ok("local") => {
                    is_local = true;
                }
                Ok(_) => {
                    // Backtrack if not "local"
                    self.pos = saved_pos;
                }
                Err(_) => {
                    // No modifier, just continue
                    self.pos = saved_pos;
                }
            }

            // Parse function body
            self.expect('{')?;

            let block_begin_idx = self.ir.blocks.len() as u32;

            // Parse blocks
            while !self.try_read('}') {
                self.parse_block()?;
            }

            let block_end_idx = self.ir.blocks.len() as u32;

            // Resolve references for this function
            self.resolve_function_references()?;

            // Add function
            self.funcs.insert(func_name, func_idx);
            self.ir.functions.push(Function {
                name: func_name.to_string(),
                declaration: false,
                local_only: is_local,
                has_call: false,
                block_begin_idx,
                block_end_idx,
                arg_begin_idx,
                arg_end_idx,
            });
        } else {
            // Function declaration - no body
            let block_begin_idx = self.ir.blocks.len() as u32;
            let block_end_idx = block_begin_idx;

            self.funcs.insert(func_name, func_idx);
            self.ir.functions.push(Function {
                name: func_name.to_string(),
                declaration: true,
                local_only: false,
                has_call: false,
                block_begin_idx,
                block_end_idx,
                arg_begin_idx,
                arg_end_idx,
            });
        }

        Ok(())
    }

    fn parse_block(&mut self) -> Result<(), String> {
        self.skip_whitespace(true);
        let block_name = self.read_identifier()?;
        self.expect(':')?;

        let block_idx = self.ir.blocks.len() as u32;
        self.blocks.insert(block_name, block_idx);

        let inst_begin_idx = self.ir.values.len() as u32;
        let mut phi_end_idx = inst_begin_idx;

        // We'll collect successors after parsing all instructions
        let mut successor_refs = Vec::new();

        // Parse instructions
        while !self.is_at_block_end() {
            self.skip_whitespace(true);

            if self.is_eof() {
                break;
            }

            // Check if this is a PHI
            if self.peek_phi() {
                // Only allow PHIs if we haven't parsed any non-PHI instructions yet
                if self.ir.values.len() as u32 > phi_end_idx {
                    return Err("PHI nodes must be at the beginning of a block".to_string());
                }
                self.parse_phi()?;
                phi_end_idx = self.ir.values.len() as u32;
            } else {
                // Parse instruction and collect any successor references
                let succs = self.parse_instruction_with_successors()?;
                successor_refs.extend(succs);
            }
        }

        let inst_end_idx = self.ir.values.len() as u32;

        // Now add successor references to value_operands
        let succ_begin_idx = self.ir.value_operands.len() as u32;
        for succ_name in &successor_refs {
            self.block_resolves.push(Resolve {
                name: succ_name,
                index: self.ir.value_operands.len() as u32,
            });
            self.ir.value_operands.push(0); // Placeholder
        }
        let succ_end_idx = self.ir.value_operands.len() as u32;

        // Create block
        self.ir.blocks.push(Block {
            name: block_name.to_string(),
            succ_begin_idx,
            succ_end_idx,
            inst_begin_idx,
            phi_end_idx,
            inst_end_idx,
            block_info: 0,
            block_info2: 0,
        });

        Ok(())
    }

    fn is_at_block_end(&mut self) -> bool {
        self.skip_whitespace(true);

        // At end of function
        if self.current_char() == Some('}') {
            return true;
        }

        // Next block starts
        if let Ok(name) = self.read_identifier() {
            // Backtrack
            self.pos -= name.len();

            // Check if followed by ':'
            let saved_pos = self.pos;
            self.pos += name.len();
            self.skip_whitespace(true);
            let has_colon = self.current_char() == Some(':');
            self.pos = saved_pos;

            return has_colon;
        }

        false
    }

    fn is_at_line_end(&self) -> bool {
        let mut pos = self.pos;
        while pos < self.text.len() {
            match self.text.chars().nth(pos) {
                Some('\n') | None => return true,
                Some(' ') | Some('\t') | Some('\r') => pos += 1,
                Some(';') => return true, // Comment starts
                _ => return false,
            }
        }
        true
    }

    fn peek_phi(&mut self) -> bool {
        let saved_pos = self.pos;
        self.skip_whitespace(true);

        // Try to parse "%name = phi"
        if self.current_char() == Some('%') {
            self.advance();
            if self.read_identifier().is_ok() {
                self.skip_whitespace(true);
                if self.current_char() == Some('!') {
                    self.advance();
                }
                self.skip_whitespace(true);
                if self.current_char() == Some('=') {
                    self.advance();
                    self.skip_whitespace(true);
                    if let Ok(op) = self.read_identifier() {
                        self.pos = saved_pos;
                        return op == "phi";
                    }
                }
            }
        }

        self.pos = saved_pos;
        false
    }

    fn parse_phi(&mut self) -> Result<(), String> {
        let (name, force_fixed) = self.read_value_name()?;
        self.expect('=')?;

        let op_name = self.read_identifier()?;
        if op_name != "phi" {
            return Err(format!("Expected 'phi' but found '{}'", op_name));
        }

        let val_idx = self.ir.values.len() as u32;
        self.values.insert(name, val_idx);

        let op_begin_idx = self.ir.value_operands.len() as u32;
        let mut incoming_count = 0;

        // Temporarily store values and blocks
        let mut temp_values = Vec::new();
        let mut temp_blocks = Vec::new();

        // Parse incoming values: [^block, %value], ...
        self.expect('[')?;
        loop {
            let block_name = self.read_block_name()?;
            self.expect(',')?;
            let (val_name, _) = self.read_value_name()?;
            self.expect(']')?;

            temp_values.push((val_name, self.ir.value_operands.len() as u32));
            temp_blocks.push((
                block_name,
                self.ir.value_operands.len() as u32 + incoming_count,
            ));

            incoming_count += 1;

            if !self.try_read(',') {
                break;
            }
            self.skip_whitespace(true);
            self.expect('[')?;
        }

        // Now push values first, then blocks, in the correct order
        // Push value placeholders and create resolves
        for (val_name, _) in &temp_values {
            self.value_resolves.push(Resolve {
                name: val_name,
                index: self.ir.value_operands.len() as u32,
            });
            self.ir.value_operands.push(0); // Placeholder
        }

        // Push block placeholders and create resolves
        for (block_name, _) in &temp_blocks {
            self.block_resolves.push(Resolve {
                name: block_name,
                index: self.ir.value_operands.len() as u32,
            });
            self.ir.value_operands.push(0); // Placeholder
        }

        let op_end_idx = self.ir.value_operands.len() as u32;

        self.ir.values.push(Value {
            name: name.to_string(),
            value_type: ValueType::Phi,
            op: Operation::None,
            force_fixed_assignment: force_fixed,
            call_func_idx: 0,
            op_count: incoming_count,
            op_begin_idx,
            op_end_idx,
        });

        Ok(())
    }

    fn parse_instruction_with_successors(&mut self) -> Result<Vec<&'a str>, String> {
        let mut successors = Vec::new();
        self.parse_instruction(&mut successors)?;
        Ok(successors)
    }

    fn parse_instruction(&mut self, successors: &mut Vec<&'a str>) -> Result<(), String> {
        self.skip_whitespace(true);

        // Check for value definition
        let (has_result, name, force_fixed, op) = if self.current_char() == Some('%') {
            let (name, force_fixed) = self.read_value_name()?;
            self.expect('=')?;

            // Check if there's an operation or just empty value definition
            self.skip_whitespace(false);

            // Check if we're at the end of the line
            let at_eol = self.current_char() == Some('\n') || self.is_at_line_end();

            let op = if at_eol {
                // Empty value definition ("any" operation) with no operands
                Operation::Any
            } else if self.current_char() == Some('%') {
                // This is an "any" operation with operands (e.g., "%dep = %alloca, %val")
                Operation::Any
            } else {
                // Read operation
                let op_str = self.read_identifier()?;
                Operation::parse(op_str).ok_or_else(|| format!("Unknown operation: {}", op_str))?
            };

            (true, Some(name), force_fixed, op)
        } else {
            // No value definition, read operation directly
            let op_str = self.read_identifier()?;
            let op =
                Operation::parse(op_str).ok_or_else(|| format!("Unknown operation: {}", op_str))?;
            (false, None, false, op)
        };

        let info = op.info();

        // Check consistency
        if has_result && !info.is_def {
            return Err(format!(
                "Operation '{}' does not produce a value",
                info.name
            ));
        }
        if !has_result && info.is_def && op != Operation::Call {
            return Err(format!("Operation '{}' requires a result value", info.name));
        }

        let val_idx = self.ir.values.len() as u32;
        if let Some(name) = name {
            self.values.insert(name, val_idx);
        }

        let op_begin_idx = self.ir.value_operands.len() as u32;

        // Parse operands based on operation type
        match op {
            Operation::Alloca => {
                // alloca <size>, <align>
                let size = self.read_number()?;
                self.expect(',')?;
                let align = self.read_number()?;
                self.ir.value_operands.push(size);
                self.ir.value_operands.push(align);
            }
            Operation::Terminate => {
                // No operands
            }
            Operation::Br => {
                // br ^block
                let block_name = self.read_block_name()?;
                self.block_resolves.push(Resolve {
                    name: block_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                // Add to successors
                successors.push(block_name);
            }
            Operation::CondBr => {
                // condbr %cond, ^true_block, ^false_block
                let (cond_name, _) = self.read_value_name()?;
                self.value_resolves.push(Resolve {
                    name: cond_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.expect(',')?;
                let true_block = self.read_block_name()?;
                self.expect(',')?;
                let false_block = self.read_block_name()?;

                self.block_resolves.push(Resolve {
                    name: true_block,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.block_resolves.push(Resolve {
                    name: false_block,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                // Add to successors
                successors.push(true_block);
                successors.push(false_block);
            }
            Operation::Jump => {
                // jump ^block1, ^block2, ...
                loop {
                    let block_name = self.read_block_name()?;
                    self.block_resolves.push(Resolve {
                        name: block_name,
                        index: self.ir.value_operands.len() as u32,
                    });
                    self.ir.value_operands.push(0); // Placeholder

                    // Add to successors
                    successors.push(block_name);

                    if !self.try_read(',') {
                        break;
                    }
                }
            }
            Operation::Ret => {
                // ret %value
                let (val_name, _) = self.read_value_name()?;
                self.value_resolves.push(Resolve {
                    name: val_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder
            }
            Operation::Add | Operation::Sub => {
                // binary ops: add %a, %b
                let (a_name, _) = self.read_value_name()?;
                self.expect(',')?;
                let (b_name, _) = self.read_value_name()?;

                self.value_resolves.push(Resolve {
                    name: a_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.value_resolves.push(Resolve {
                    name: b_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder
            }
            Operation::Tbz => {
                // tbz %val, ^zero_block, ^nonzero_block, <bit>
                let (val_name, _) = self.read_value_name()?;
                self.value_resolves.push(Resolve {
                    name: val_name,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.expect(',')?;
                let zero_block = self.read_block_name()?;
                self.expect(',')?;
                let nonzero_block = self.read_block_name()?;
                self.expect(',')?;
                let bit = self.read_number()?;

                self.block_resolves.push(Resolve {
                    name: zero_block,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.block_resolves.push(Resolve {
                    name: nonzero_block,
                    index: self.ir.value_operands.len() as u32,
                });
                self.ir.value_operands.push(0); // Placeholder

                self.ir.value_operands.push(bit);

                // Add to successors
                successors.push(zero_block);
                successors.push(nonzero_block);
            }
            Operation::Call => {
                // call @func_name or call @func_name, %arg1, %arg2
                self.skip_whitespace(false);

                let _func_name = if self.current_char() == Some('@') {
                    self.advance(); // Skip '@'
                    self.read_identifier()?
                } else {
                    return Err("Expected '@' before function name in call".to_string());
                };

                // TODO: Resolve function reference
                let call_func_idx = 0;

                // Parse arguments if any
                let mut arg_count = 0;
                self.skip_whitespace(false);

                // Check if there are arguments
                if self.current_char() == Some(',') {
                    self.advance(); // Skip comma
                    self.skip_whitespace(false);

                    // Parse arguments
                    loop {
                        if self.current_char() != Some('%') {
                            break;
                        }

                        let (arg_name, _) = self.read_value_name()?;
                        self.value_resolves.push(Resolve {
                            name: arg_name,
                            index: self.ir.value_operands.len() as u32,
                        });
                        self.ir.value_operands.push(0); // Placeholder
                        arg_count += 1;

                        self.skip_whitespace(false);
                        if self.current_char() == Some(',') {
                            self.advance();
                            self.skip_whitespace(false);
                        } else {
                            break;
                        }
                    }
                }

                self.ir.values.push(Value {
                    name: name.map(|n| n.to_string()).unwrap_or_default(),
                    value_type: if info.is_terminator {
                        ValueType::Terminator
                    } else {
                        ValueType::Normal
                    },
                    op,
                    force_fixed_assignment: force_fixed,
                    call_func_idx,
                    op_count: arg_count,
                    op_begin_idx,
                    op_end_idx: self.ir.value_operands.len() as u32,
                });

                return Ok(());
            }
            Operation::ZeroFill => {
                // zerofill <size>
                let size = self.read_number()?;
                self.ir.value_operands.push(size);
            }
            Operation::Any | Operation::None => {
                // Variable number of operands
                // For "any" op, check if there are operands
                if op == Operation::Any && self.current_char() == Some('%') {
                    // Parse comma-separated value references
                    let mut first = true;
                    loop {
                        if !first {
                            self.skip_whitespace(false);
                            if !self.try_read(',') {
                                break;
                            }
                            self.skip_whitespace(false);
                        }

                        if self.current_char() != Some('%') {
                            break;
                        }

                        first = false;
                        let (val_name, _) = self.read_value_name()?;
                        self.value_resolves.push(Resolve {
                            name: val_name,
                            index: self.ir.value_operands.len() as u32,
                        });
                        self.ir.value_operands.push(0); // Placeholder
                    }
                }
            }
        }

        let op_end_idx = self.ir.value_operands.len() as u32;

        self.ir.values.push(Value {
            name: name.map(|n| n.to_string()).unwrap_or_default(),
            value_type: if info.is_terminator {
                ValueType::Terminator
            } else {
                ValueType::Normal
            },
            op,
            force_fixed_assignment: force_fixed,
            call_func_idx: 0,
            op_count: if op == Operation::Any {
                op_end_idx - op_begin_idx
            } else {
                info.op_count
            },
            op_begin_idx,
            op_end_idx,
        });

        Ok(())
    }

    fn resolve_function_references(&mut self) -> Result<(), String> {
        // Resolve value references
        for resolve in &self.value_resolves {
            if let Some(&idx) = self.values.get(resolve.name) {
                self.ir.value_operands[resolve.index as usize] = idx;
            } else {
                return Err(format!("Undefined value reference: {}", resolve.name));
            }
        }

        // Resolve block references
        for resolve in &self.block_resolves {
            if let Some(&idx) = self.blocks.get(resolve.name) {
                self.ir.value_operands[resolve.index as usize] = idx;
            } else {
                return Err(format!("Undefined block reference: {}", resolve.name));
            }
        }

        Ok(())
    }

    fn resolve_all_references(&mut self) -> Result<(), String> {
        // Resolve function references (cross-function)
        for resolve in &self.func_resolves {
            if let Some(&idx) = self.funcs.get(resolve.name) {
                self.ir.value_operands[resolve.index as usize] = idx;
            } else {
                return Err(format!("Undefined function reference: {}", resolve.name));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_add() {
        let tir = r#"
; Simple add function
func() {
entry:
  %a =
  %b =
  %c = add %a, %b
  terminate
}
"#;

        let ir = match TestIR::parse(tir) {
            Ok(ir) => ir,
            Err(e) => {
                eprintln!("Parse error: {}", e);
                panic!("Failed to parse");
            }
        };

        assert_eq!(ir.functions.len(), 1);
        assert_eq!(ir.functions[0].name, "func");

        assert_eq!(ir.blocks.len(), 1);
        assert_eq!(ir.blocks[0].name, "entry");

        assert_eq!(ir.values.len(), 4);
        assert_eq!(ir.values[0].name, "a");
        assert_eq!(ir.values[1].name, "b");
        assert_eq!(ir.values[2].name, "c");
        assert_eq!(ir.values[2].op, Operation::Add);
    }

    #[test]
    fn test_parse_with_branches() {
        let tir = r#"
br1() {
entry:
  br ^secBlock
secBlock:
  br ^retBlock
retBlock:
  terminate
}
"#;

        let ir = TestIR::parse(tir).unwrap();

        assert_eq!(ir.functions.len(), 1);
        assert_eq!(ir.blocks.len(), 3);
        assert_eq!(ir.blocks[0].name, "entry");
        assert_eq!(ir.blocks[1].name, "secBlock");
        assert_eq!(ir.blocks[2].name, "retBlock");
    }

    #[test]
    fn test_parse_conditional_branch() {
        let tir = r#"
condbr1() {
entry:
  %val =
  condbr %val, ^ret1, ^ret2
ret1:
  terminate
ret2:
  terminate
}
"#;

        let ir = TestIR::parse(tir).unwrap();

        assert_eq!(ir.functions.len(), 1);
        assert_eq!(ir.blocks.len(), 3);

        // Check conditional branch instruction
        let condbr = &ir.values[1];
        assert_eq!(condbr.op, Operation::CondBr);
        assert_eq!(condbr.op_count, 1);
    }

    #[test]
    fn test_parse_phi_node() {
        let tir = r#"
myfunc(%a) {
entry:
  %val =
  jump ^ret, ^other
other:
  %val2 =
  jump ^ret
ret:
  %phi = phi [^entry, %val], [^other, %val2]
  terminate
}
"#;

        let ir = TestIR::parse(tir).unwrap();

        // Find the phi node
        let phi = ir
            .values
            .iter()
            .find(|v| v.value_type == ValueType::Phi)
            .unwrap();
        assert_eq!(phi.name, "phi");
        assert_eq!(phi.op_count, 2); // Two incoming values
    }

    #[test]
    fn test_parse_alloca() {
        let tir = r#"
func() {
entry:
  %alloca = alloca 16, 8
  terminate
}
"#;

        let ir = TestIR::parse(tir).unwrap();

        let alloca = &ir.values[0];
        assert_eq!(alloca.op, Operation::Alloca);

        // Check immediate values
        let size = ir.value_operands[alloca.op_begin_idx as usize];
        let align = ir.value_operands[(alloca.op_begin_idx + 1) as usize];
        assert_eq!(size, 16);
        assert_eq!(align, 8);
    }

    #[test]
    fn test_print_ir() {
        let tir = r#"
func() {
entry:
  %a =
  %b =
  %c = add %a, %b
  terminate
}
"#;

        let ir = TestIR::parse(tir).unwrap();
        let output = ir.print();

        assert!(output.contains("Printing IR"));
        assert!(output.contains("Function func"));
        assert!(output.contains("Block entry"));
        assert!(output.contains("Value a (any)"));
        assert!(output.contains("Value b (any)"));
        assert!(output.contains("Value c (add)"));
        assert!(output.contains("Op a"));
        assert!(output.contains("Op b"));
        assert!(output.contains("Value (terminate)"));
    }
}
