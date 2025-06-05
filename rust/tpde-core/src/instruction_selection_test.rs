//! Test real instruction selection with enhanced LLVM adaptor.
//!
//! This module tests the integration of the enhanced LLVM adaptor with
//! improved instruction selection that uses opcode-based dispatch.

#[cfg(test)]
pub mod tests {
    use crate::{
        complete_compiler::CompleteCompiler,
        llvm_compiler::{LlvmAdaptorInterface, InstructionCategory},
        adaptor::IrAdaptor,
    };
    
    /// Test adaptor that simulates enhanced LLVM adaptor behavior.
    ///
    /// This provides a concrete implementation of the LlvmAdaptorInterface
    /// trait to test opcode-based instruction selection.
    #[derive(Debug)]
    pub struct TestLlvmAdaptor {
        current_func: Option<usize>,
        functions: Vec<TestFunction>,
    }
    
    #[derive(Debug)]
    struct TestFunction {
        name: String,
        blocks: Vec<TestBlock>,
        values: Vec<TestValue>,
        instructions: Vec<TestInstruction>,
    }
    
    #[derive(Debug)]
    struct TestBlock {
        instruction_indices: Vec<usize>,
    }
    
    #[derive(Debug)]
    struct TestValue {
        local_idx: usize,
    }
    
    #[derive(Debug)]
    struct TestInstruction {
        category: InstructionCategory,
        operands: Vec<usize>,
        results: Vec<usize>,
    }
    
    impl TestLlvmAdaptor {
        /// Create a test adaptor with arithmetic function: add(a, b) = a + b
        pub fn new_with_arithmetic_function() -> Self {
            let mut adaptor = Self {
                current_func: None,
                functions: Vec::new(),
            };
            
            // Create function: int add(int a, int b) { return a + b; }
            let mut add_func = TestFunction {
                name: "add".to_string(),
                blocks: Vec::new(),
                values: Vec::new(),
                instructions: Vec::new(),
            };
            
            // Create values: %a (arg0), %b (arg1), %result (local)
            add_func.values.push(TestValue { local_idx: 0 }); // %a
            add_func.values.push(TestValue { local_idx: 1 }); // %b  
            add_func.values.push(TestValue { local_idx: 2 }); // %result
            
            // Create instructions
            // %result = add %a, %b
            add_func.instructions.push(TestInstruction {
                category: InstructionCategory::Arithmetic,
                operands: vec![0, 1], // %a, %b
                results: vec![2],     // %result
            });
            
            // ret %result
            add_func.instructions.push(TestInstruction {
                category: InstructionCategory::ControlFlow,
                operands: vec![2], // %result
                results: vec![],   // no results
            });
            
            // Create single block with all instructions
            let mut block = TestBlock {
                instruction_indices: vec![0, 1], // both instructions
            };
            
            add_func.blocks.push(block);
            adaptor.functions.push(add_func);
            
            adaptor
        }
        
        /// Create a test adaptor with comparison function: cmp(a, b) = a < b
        pub fn new_with_comparison_function() -> Self {
            let mut adaptor = Self {
                current_func: None,
                functions: Vec::new(),
            };
            
            // Create function: bool cmp(int a, int b) { return a < b; }
            let mut cmp_func = TestFunction {
                name: "cmp".to_string(),
                blocks: Vec::new(),
                values: Vec::new(),
                instructions: Vec::new(),
            };
            
            // Create values: %a, %b, %cmp_result
            cmp_func.values.push(TestValue { local_idx: 0 }); // %a
            cmp_func.values.push(TestValue { local_idx: 1 }); // %b  
            cmp_func.values.push(TestValue { local_idx: 2 }); // %cmp_result
            
            // Create instructions
            // %cmp_result = icmp slt %a, %b
            cmp_func.instructions.push(TestInstruction {
                category: InstructionCategory::Comparison,
                operands: vec![0, 1], // %a, %b
                results: vec![2],     // %cmp_result
            });
            
            // ret %cmp_result
            cmp_func.instructions.push(TestInstruction {
                category: InstructionCategory::ControlFlow,
                operands: vec![2], // %cmp_result
                results: vec![],   // no results
            });
            
            // Create single block
            let block = TestBlock {
                instruction_indices: vec![0, 1],
            };
            
            cmp_func.blocks.push(block);
            adaptor.functions.push(cmp_func);
            
            adaptor
        }
    }
    
    impl crate::adaptor::IrAdaptor for TestLlvmAdaptor {
        type ValueRef = usize;
        type InstRef = usize;
        type BlockRef = usize;
        type FuncRef = usize;

        const INVALID_VALUE_REF: Self::ValueRef = usize::MAX;
        const INVALID_BLOCK_REF: Self::BlockRef = usize::MAX;
        const INVALID_FUNC_REF: Self::FuncRef = usize::MAX;

        fn func_count(&self) -> u32 {
            self.functions.len() as u32
        }

        fn funcs(&self) -> Box<dyn Iterator<Item = Self::FuncRef> + '_> {
            Box::new(0..self.functions.len())
        }

        fn func_link_name(&self, func: Self::FuncRef) -> &str {
            &self.functions[func].name
        }

        fn switch_func(&mut self, func: Self::FuncRef) -> bool {
            if func < self.functions.len() {
                self.current_func = Some(func);
                true
            } else {
                false
            }
        }

        fn reset(&mut self) {
            self.current_func = None;
        }

        fn entry_block(&self) -> Self::BlockRef {
            0
        }

        fn blocks(&self) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
            if let Some(func_idx) = self.current_func {
                Box::new(0..self.functions[func_idx].blocks.len())
            } else {
                Box::new(std::iter::empty())
            }
        }

        fn block_insts(&self, block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::InstRef> + '_> {
            if let Some(func_idx) = self.current_func {
                if block < self.functions[func_idx].blocks.len() {
                    let instruction_indices = &self.functions[func_idx].blocks[block].instruction_indices;
                    Box::new(instruction_indices.clone().into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                Box::new(std::iter::empty())
            }
        }

        fn block_succs(&self, _block: Self::BlockRef) -> Box<dyn Iterator<Item = Self::BlockRef> + '_> {
            Box::new(std::iter::empty())
        }

        fn inst_operands(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
            if let Some(func_idx) = self.current_func {
                if inst < self.functions[func_idx].instructions.len() {
                    let operands = self.functions[func_idx].instructions[inst].operands.clone();
                    return Box::new(operands.into_iter());
                }
            }
            Box::new(std::iter::empty())
        }

        fn inst_results(&self, inst: Self::InstRef) -> Box<dyn Iterator<Item = Self::ValueRef> + '_> {
            if let Some(func_idx) = self.current_func {
                if inst < self.functions[func_idx].instructions.len() {
                    let results = self.functions[func_idx].instructions[inst].results.clone();
                    return Box::new(results.into_iter());
                }
            }
            Box::new(std::iter::empty())
        }

        fn val_local_idx(&self, val: Self::ValueRef) -> usize {
            if let Some(func_idx) = self.current_func {
                if val < self.functions[func_idx].values.len() {
                    return self.functions[func_idx].values[val].local_idx;
                }
            }
            0
        }

        fn val_ignore_liveness(&self, _val: Self::ValueRef) -> bool {
            false
        }
    }
    
    impl LlvmAdaptorInterface for TestLlvmAdaptor {
        fn get_instruction_category(&self, inst: Self::InstRef) -> InstructionCategory {
            if let Some(func_idx) = self.current_func {
                if inst < self.functions[func_idx].instructions.len() {
                    return self.functions[func_idx].instructions[inst].category;
                }
            }
            InstructionCategory::Other
        }
        
        fn makes_calls(&self) -> bool {
            false
        }
        
        fn has_allocas(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_arithmetic_instruction_selection() {
        let adaptor = TestLlvmAdaptor::new_with_arithmetic_function();
        let mut compiler = CompleteCompiler::new(adaptor).unwrap();
        
        // This should compile using opcode-based instruction selection
        let result = compiler.compile_all();
        assert!(result.is_ok(), "Arithmetic instruction compilation failed: {:?}", result.err());
        
        // Check that we have compiled functions
        let compiled = compiler.get_compiled_functions();
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].name, "add");
        assert!(!compiled[0].code.is_empty());
        
        println!("Successfully compiled arithmetic function '{}' to {} bytes of machine code", 
                compiled[0].name, compiled[0].code.len());
    }

    #[test]
    fn test_comparison_instruction_selection() {
        let adaptor = TestLlvmAdaptor::new_with_comparison_function();
        let mut compiler = CompleteCompiler::new(adaptor).unwrap();
        
        // This should compile comparison instructions
        let result = compiler.compile_all();
        assert!(result.is_ok(), "Comparison instruction compilation failed: {:?}", result.err());
        
        // Check that we have compiled functions
        let compiled = compiler.get_compiled_functions();
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].name, "cmp");
        assert!(!compiled[0].code.is_empty());
        
        println!("Successfully compiled comparison function '{}' to {} bytes of machine code", 
                compiled[0].name, compiled[0].code.len());
    }
    
    #[test]
    fn test_opcode_based_dispatch() {
        let adaptor = TestLlvmAdaptor::new_with_arithmetic_function();
        
        // Test that the adaptor correctly provides instruction categories
        let mut test_adaptor = adaptor;
        assert!(test_adaptor.switch_func(0));
        
        // Check instruction categories
        let blocks: Vec<_> = test_adaptor.blocks().collect();
        assert_eq!(blocks.len(), 1);
        
        let instructions: Vec<_> = test_adaptor.block_insts(blocks[0]).collect();
        assert_eq!(instructions.len(), 2);
        
        // First instruction should be arithmetic
        let category0 = test_adaptor.get_instruction_category(instructions[0]);
        assert_eq!(category0, InstructionCategory::Arithmetic);
        
        // Second instruction should be control flow
        let category1 = test_adaptor.get_instruction_category(instructions[1]);
        assert_eq!(category1, InstructionCategory::ControlFlow);
        
        println!("Opcode-based dispatch test passed: arithmetic={:?}, control_flow={:?}", 
                category0, category1);
    }
}