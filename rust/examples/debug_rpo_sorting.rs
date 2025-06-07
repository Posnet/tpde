use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Testing Sorting Approach ===\n");
    
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
    
    // Test the sorting on loop_head's successors
    let loop_head_succs = &successors["loop_head"];
    println!("\nloop_head successors before sorting: {loop_head_succs:?}");
    
    let mut sorted_succs = loop_head_succs.clone();
    sorted_succs.sort_by_key(|&succ| block_order_map.get(&succ).copied().unwrap_or(u32::MAX));
    println!("loop_head successors after sorting: {sorted_succs:?}");
    
    // Now simulate the full RPO algorithm with sorting
    println!("\n=== Full RPO Algorithm ===");
    let rpo = compute_rpo_with_sorting("entry", &successors, &block_order_map);
    println!("Result: {rpo:?}");
    println!("Expected: [\"entry\", \"loop_head\", \"loop_body\", \"cont\"]");
}

fn compute_rpo_with_sorting<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    block_order_map: &HashMap<&'a str, u32>
) -> Vec<&'a str> {
    let mut post = Vec::new();
    let mut stack = vec![(entry, false)];
    let mut visited = HashSet::new();
    
    println!("\nDFS Trace:");
    while let Some((block, processed)) = stack.pop() {
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
        
        // Get and sort successors
        if let Some(succs) = successors.get(block) {
            let mut sorted_succs: Vec<_> = succs.clone();
            sorted_succs.sort_by_key(|&succ| block_order_map.get(&succ).copied().unwrap_or(u32::MAX));
            
            println!("    Successors: {succs:?}");
            println!("    Sorted: {sorted_succs:?}");
            
            // Push in reverse order
            for succ in sorted_succs.into_iter().rev() {
                println!("    Pushing {succ} to stack");
                stack.push((succ, false));
            }
        }
    }
    
    println!("\nPost-order: {post:?}");
    post.reverse();
    println!("RPO: {post:?}");
    post
}