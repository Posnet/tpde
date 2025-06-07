use tpde::core::IrAdaptor;
use tpde::test_ir::{TestIR, TestIRAdaptor};

fn main() {
    let tir_content = r#"
simple(%a) {
entry:
  %const =
  jump ^loop
loop:
  %b = phi [^entry, %const], [^loop, %e]
  %c = phi [^entry, %a], [^loop, %f]
  %d = %c
  %e = %b, %d
  %f = %c
  jump ^loop, ^ret
ret:
  %ret = %b
  terminate
}
"#;

    let ir = TestIR::parse(tir_content).expect("Failed to parse TIR");

    println!("Debug: blocks:");
    for (i, block) in ir.blocks.iter().enumerate() {
        println!("  [{}] = {}", i, block.name);
    }

    println!("\nDebug: values:");
    for (i, val) in ir.values.iter().enumerate() {
        println!("  [{}] = {} ({:?})", i, val.name, val.value_type);
    }

    println!("\nDebug: value_operands array:");
    for (i, &val) in ir.value_operands.iter().enumerate() {
        println!("  [{i}] = {val}");
    }

    let mut adaptor = TestIRAdaptor::new(&ir);

    // Process the function
    let func = adaptor.funcs().next().unwrap();
    adaptor.switch_func(func);

    println!("Manually tracing liveness calculation:");
    println!("\n1. Function arguments (if TPDE_LIVENESS_VISIT_ARGS):");
    for arg in adaptor.cur_args() {
        let local_idx = adaptor.val_local_idx(arg);
        println!(
            "   - {} (local_idx={}) defined in block 0",
            adaptor.value_name(arg),
            local_idx
        );
    }

    println!("\n2. Processing blocks:");

    // Entry block
    println!("\n   Block 0 (entry):");
    let entry_block = adaptor.entry_block();

    println!("   PHIs in entry:");
    for phi in adaptor.block_phis(entry_block) {
        println!("     - {} (shouldn't have any)", adaptor.value_name(phi));
    }

    println!("   Instructions in entry:");
    for inst in adaptor.block_insts(entry_block) {
        let inst_idx = inst.0;
        let val = &ir.values[inst_idx as usize];
        println!("     - inst {} = {} ({:?})", inst_idx, val.name, val.op);

        // Check if it produces a result
        let results: Vec<_> = adaptor.inst_results(inst).collect();
        for res in &results {
            let local_idx = adaptor.val_local_idx(*res);
            println!(
                "       -> defines {} (local_idx={})",
                adaptor.value_name(*res),
                local_idx
            );
        }

        // Check operands
        let operands: Vec<_> = adaptor.inst_operands(inst).collect();
        for op in &operands {
            let local_idx = adaptor.val_local_idx(*op);
            println!(
                "       -> uses {} (local_idx={})",
                adaptor.value_name(*op),
                local_idx
            );
        }
    }

    // Get all blocks
    let blocks: Vec<_> = adaptor.blocks().collect();

    // Loop block
    if blocks.len() > 1 {
        println!("\n   Block 1 (loop):");
        let loop_block = blocks[1];

        println!("   PHIs in loop:");
        for phi in adaptor.block_phis(loop_block) {
            let phi_name = adaptor.value_name(phi);
            let local_idx = adaptor.val_local_idx(phi);
            let phi_info = &ir.values[phi.0 as usize];
            println!(
                "     - {} (local_idx={}, op_begin={}, op_count={}, op_end={})",
                phi_name, local_idx, phi_info.op_begin_idx, phi_info.op_count, phi_info.op_end_idx
            );

            // Debug: print what's in value_operands
            println!("       value_operands length: {}", ir.value_operands.len());
            println!(
                "       trying to access indices {} to {}",
                phi_info.op_begin_idx, phi_info.op_end_idx
            );

            // Check incoming values
            let incoming_count = adaptor.phi_incoming_count(phi);
            for slot in 0..incoming_count {
                let val_idx = phi_info.op_begin_idx + slot;
                let block_idx = phi_info.op_begin_idx + phi_info.op_count + slot;
                println!("       slot {slot}: val_idx={val_idx}, block_idx={block_idx}");

                if block_idx as usize >= ir.value_operands.len() {
                    println!("       ERROR: block_idx {block_idx} is out of bounds!");
                    continue;
                }

                let inc_block = adaptor.phi_incoming_block_for_slot(phi, slot);
                let inc_val = adaptor.phi_incoming_val_for_slot(phi, slot);
                let inc_block_name = adaptor.block_name(inc_block);
                let inc_val_name = adaptor.value_name(inc_val);
                let inc_val_idx = adaptor.val_local_idx(inc_val);

                println!("       -> from {inc_block_name}: uses {inc_val_name} (local_idx={inc_val_idx})");
                println!("       -> PHI {phi_name} (local_idx={local_idx}) is marked as used in block {inc_block_name}");
            }
        }

        println!("   Instructions in loop:");
        for inst in adaptor.block_insts(loop_block) {
            let inst_idx = inst.0;
            let val = &ir.values[inst_idx as usize];
            println!("     - inst {} = {} ({:?})", inst_idx, val.name, val.op);

            // Check if it produces a result
            let results: Vec<_> = adaptor.inst_results(inst).collect();
            for res in &results {
                let local_idx = adaptor.val_local_idx(*res);
                println!(
                    "       -> defines {} (local_idx={})",
                    adaptor.value_name(*res),
                    local_idx
                );
            }

            // Check operands
            let operands: Vec<_> = adaptor.inst_operands(inst).collect();
            for op in &operands {
                let local_idx = adaptor.val_local_idx(*op);
                println!(
                    "       -> uses {} (local_idx={})",
                    adaptor.value_name(*op),
                    local_idx
                );
            }
        }
    }

    // Ret block
    if blocks.len() > 2 {
        println!("\n   Block 2 (ret):");
        let ret_block = blocks[2];

        println!("   Instructions in ret:");
        for inst in adaptor.block_insts(ret_block) {
            let inst_idx = inst.0;
            let val = &ir.values[inst_idx as usize];
            println!("     - inst {} = {} ({:?})", inst_idx, val.name, val.op);

            // Check if it produces a result
            let results: Vec<_> = adaptor.inst_results(inst).collect();
            for res in &results {
                let local_idx = adaptor.val_local_idx(*res);
                println!(
                    "       -> defines {} (local_idx={})",
                    adaptor.value_name(*res),
                    local_idx
                );
            }

            // Check operands
            let operands: Vec<_> = adaptor.inst_operands(inst).collect();
            for op in &operands {
                let local_idx = adaptor.val_local_idx(*op);
                println!(
                    "       -> uses {} (local_idx={})",
                    adaptor.value_name(*op),
                    local_idx
                );
            }
        }
    }

    println!("\nExpected ref counts:");
    println!("  %a (idx 0): 1 def + 1 use in PHI = 2 refs");
    println!("  %const (idx 1): 1 def + 1 use in PHI = 2 refs");
    println!("  %b (idx 3): 2 PHI edges + 1 use in %e + 1 use in %ret = 4 refs");
    println!("  %c (idx 4): 2 PHI edges + 1 use in %d + 1 use in %f = 4 refs");
}
