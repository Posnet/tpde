use std::collections::{HashMap, HashSet};

fn main() {
    // Define the loop1 function structure
    println!("=== Analyzing loop1 function ===");
    
    // Blocks in order they appear in IR
    let blocks = vec!["entry", "loop_head", "loop_body", "cont"];
    println!("\n1. Blocks in IR order:");
    for (i, block) in blocks.iter().enumerate() {
        println!("   {}: {}", i, block);
    }
    
    // Define successors for each block
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    successors.insert("entry", vec!["loop_head"]);
    successors.insert("loop_head", vec!["loop_body", "cont"]);
    successors.insert("loop_body", vec!["loop_head"]);
    successors.insert("cont", vec![]); // terminate
    
    println!("\n2. Successors of each block:");
    for block in &blocks {
        if let Some(succs) = successors.get(block) {
            println!("   {} -> {:?}", block, succs);
        } else {
            println!("   {} -> []", block);
        }
    }
    
    // Compute RPO
    let rpo = compute_rpo("entry", &successors);
    println!("\n3. RPO order generated:");
    for (i, block) in rpo.iter().enumerate() {
        println!("   {}: {}", i, block);
    }
    
    // Expected order
    println!("\n4. Expected RPO order:");
    println!("   0: entry");
    println!("   1: loop_head");
    println!("   2: loop_body");
    println!("   3: cont");
    
    // Analyze the difference
    println!("\n5. Analysis:");
    if rpo[2] == "cont" && rpo[3] == "loop_body" {
        println!("   Issue detected: cont comes before loop_body");
        println!("   This happens because when visiting loop_head's successors,");
        println!("   cont is visited before loop_body in the DFS traversal.");
    }
    
    // Try with different successor ordering
    println!("\n=== Testing with reversed successor order ===");
    let mut successors_rev: HashMap<&str, Vec<&str>> = HashMap::new();
    successors_rev.insert("entry", vec!["loop_head"]);
    successors_rev.insert("loop_head", vec!["cont", "loop_body"]); // Reversed!
    successors_rev.insert("loop_body", vec!["loop_head"]);
    successors_rev.insert("cont", vec![]);
    
    let rpo_rev = compute_rpo("entry", &successors_rev);
    println!("RPO with reversed successors:");
    for (i, block) in rpo_rev.iter().enumerate() {
        println!("   {}: {}", i, block);
    }
}

fn compute_rpo<'a>(entry: &'a str, successors: &HashMap<&'a str, Vec<&'a str>>) -> Vec<&'a str> {
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    
    // DFS to compute post-order
    dfs(entry, successors, &mut visited, &mut post_order);
    
    // Reverse to get RPO
    post_order.reverse();
    post_order
}

fn dfs<'a>(
    node: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    visited: &mut HashSet<&'a str>,
    post_order: &mut Vec<&'a str>,
) {
    if visited.contains(node) {
        return;
    }
    
    visited.insert(node);
    
    // Visit successors
    if let Some(succs) = successors.get(node) {
        for succ in succs {
            dfs(succ, successors, visited, post_order);
        }
    }
    
    // Add to post-order after visiting all successors
    post_order.push(node);
}