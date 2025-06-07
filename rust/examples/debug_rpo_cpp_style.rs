use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Testing C++ Style Stack Sorting ===\n");

    // Simulate the loop1 structure
    let blocks = ["entry", "loop_head", "loop_body", "cont"];
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    successors.insert("entry", vec!["loop_head"]);
    successors.insert("loop_head", vec!["loop_body", "cont"]);
    successors.insert("loop_body", vec!["loop_head"]);
    successors.insert("cont", vec![]);

    // Create block order map
    let mut block_order_map = HashMap::new();
    for (idx, &block) in blocks.iter().enumerate() {
        block_order_map.insert(block, idx as u32);
    }

    println!("Block order map:");
    for (block, idx) in &block_order_map {
        println!("  {block} -> {idx}");
    }

    // Run the C++ style algorithm
    println!("\n=== C++ Style RPO Algorithm ===");
    let rpo = compute_rpo_cpp_style("entry", &successors, &block_order_map);
    println!("\nResult: {rpo:?}");
    println!("Expected: [\"entry\", \"loop_head\", \"loop_body\", \"cont\"]");

    if rpo == vec!["entry", "loop_head", "loop_body", "cont"] {
        println!("✓ SUCCESS!");
    } else {
        println!("✗ FAIL");
    }
}

fn compute_rpo_cpp_style<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    block_order_map: &HashMap<&'a str, u32>,
) -> Vec<&'a str> {
    let mut post = Vec::new();
    let mut stack = vec![(entry, false)];
    let mut visited = HashSet::new();

    println!("\nDFS Trace:");
    while let Some((block, processed)) = stack.pop() {
        println!("Stack pop: ({block}, {processed})");

        if processed {
            println!("  Adding {block} to post-order");
            post.push(block);
            continue;
        }

        if !visited.insert(block) {
            println!("  {block} already visited, skipping");
            continue;
        }

        println!("  Visiting {block}");
        stack.push((block, true));

        // Push successors onto stack first
        if let Some(succs) = successors.get(block) {
            let start_idx = stack.len();
            println!("    Successors: {succs:?}");

            for succ in succs {
                println!("    Pushing ({succ}, false) to stack");
                stack.push((succ, false));
            }

            // Sort the portion of the stack we just added
            let len = stack.len() - start_idx;
            if len > 1 {
                println!("    Sorting stack[{}..{}]", start_idx, stack.len());
                let slice = &mut stack[start_idx..];
                slice.sort_by_key(|(block, _)| {
                    block_order_map.get(block).copied().unwrap_or(u32::MAX)
                });

                println!("    After sorting:");
                for (i, (b, _)) in stack[start_idx..].iter().enumerate() {
                    println!("      stack[{}] = {}", start_idx + i, b);
                }
            }
        }

        println!("  Current stack: {stack:?}");
    }

    println!("\nPost-order: {post:?}");
    post.reverse();
    println!("RPO: {post:?}");
    post
}
