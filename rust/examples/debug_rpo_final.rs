use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Understanding RPO Algorithm ===\n");
    
    // The key insight from analyzing all test cases:
    // The RPO algorithm should visit blocks in a way that when there's a choice
    // between unvisited successors, it prefers the one that leads to fewer
    // unvisited blocks first (i.e., the "dead end" or terminating path).
    
    // For loop1:
    // From loop_head with successors [loop_body, cont]:
    // - loop_body leads back to loop_head (cycle)
    // - cont leads to termination
    // Standard DFS visits loop_body first, but the expected behavior
    // seems to visit the non-looping path first when possible.
    
    println!("Standard DFS approach (what we have):");
    test_standard_dfs();
    
    println!("\nModified approach (what tests expect):");
    test_modified_approach();
}

fn test_standard_dfs() {
    // loop1 structure
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    successors.insert("entry", vec!["loop_head"]);
    successors.insert("loop_head", vec!["loop_body", "cont"]);
    successors.insert("loop_body", vec!["loop_head"]);
    successors.insert("cont", vec![]);
    
    let rpo = compute_rpo_standard("entry", &successors);
    println!("loop1 RPO: {rpo:?}");
    println!("Expected:  [\"entry\", \"loop_head\", \"loop_body\", \"cont\"]");
}

fn test_modified_approach() {
    // The insight: The C++ implementation might be using an iterative
    // worklist algorithm instead of recursive DFS, which processes
    // blocks in a different order.
    
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    successors.insert("entry", vec!["loop_head"]);
    successors.insert("loop_head", vec!["loop_body", "cont"]);
    successors.insert("loop_body", vec!["loop_head"]);
    successors.insert("cont", vec![]);
    
    let blocks_in_source_order = vec!["entry", "loop_head", "loop_body", "cont"];
    
    let rpo = compute_rpo_worklist("entry", &successors, &blocks_in_source_order);
    println!("loop1 RPO: {rpo:?}");
}

fn compute_rpo_standard<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>
) -> Vec<&'a str> {
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    
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
        
        if let Some(succs) = successors.get(node) {
            for succ in succs {
                dfs(succ, successors, visited, post_order);
            }
        }
        
        post_order.push(node);
    }
    
    dfs(entry, successors, &mut visited, &mut post_order);
    post_order.reverse();
    post_order
}

fn compute_rpo_worklist<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    blocks_in_source_order: &[&'a str]
) -> Vec<&'a str> {
    // Create a mapping of block names to their source order
    let mut source_indices: HashMap<&str, usize> = HashMap::new();
    for (i, block) in blocks_in_source_order.iter().enumerate() {
        source_indices.insert(block, i);
    }
    
    // The key difference: when we have multiple unvisited successors,
    // we need to ensure that blocks appearing later in source order
    // are visited first in the DFS, so they appear later in the RPO.
    
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    let mut stack = vec![(entry, false)];
    
    while let Some((block, processed)) = stack.pop() {
        if processed {
            post_order.push(block);
            continue;
        }
        
        if !visited.insert(block) {
            continue;
        }
        
        // Push this block to be added to post_order after processing successors
        stack.push((block, true));
        
        // Get successors and push them in REVERSE order
        // This ensures that the first successor is processed last (DFS property)
        if let Some(succs) = successors.get(block) {
            for succ in succs.iter().rev() {
                if !visited.contains(succ) {
                    stack.push((*succ, false));
                }
            }
        }
    }
    
    post_order.reverse();
    post_order
}