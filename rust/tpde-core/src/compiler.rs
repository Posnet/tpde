use crate::{adaptor::IrAdaptor, analyzer::Analyzer, assembler::Assembler};

/// Architecture independent compiler driver.
#[allow(dead_code)]
pub struct CompilerBase<A: IrAdaptor, ASM: Assembler<A>> {
    adaptor: A,
    analyzer: Analyzer<A>,
    assembler: ASM,
}

impl<A: IrAdaptor, ASM: Assembler<A>> CompilerBase<A, ASM> {
    /// Create a new compiler base from an adaptor and assembler.
    pub fn new(adaptor: A, assembler: ASM) -> Self {
        Self {
            adaptor,
            analyzer: Analyzer::new(),
            assembler,
        }
    }

    /// Compile all functions provided by the adaptor.
    pub fn compile(&mut self) -> bool {
        let funcs: Vec<_> = self.adaptor.funcs().collect();
        for func in funcs {
            if !self.adaptor.switch_func(func) {
                continue;
            }
            self.analyzer.switch_func(&mut self.adaptor, func);
            // architecture specific code generation would go here
        }
        true
    }
}
