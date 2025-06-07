use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Analyzing RPO behavior ===\n");
    
    // Test case 1: loop1 - blocks in order
    test_rpo("loop1", 
        vec!["entry", "loop_head", "loop_body", "cont"],
        vec![
            ("entry", vec!["loop_head"]),
            ("loop_head", vec!["loop_body", "cont"]),
            ("loop_body", vec!["loop_head"]),
            ("cont", vec![]),
        ],
        vec!["entry", "loop_head", "loop_body", "cont"]
    );
    
    // Test case 2: loop2 - cont defined before loop_body
    test_rpo("loop2",
        vec!["entry", "cont", "loop_head", "loop_body"],
        vec![
            ("entry", vec!["loop_head"]),
            ("loop_head", vec!["loop_body", "cont"]),
            ("loop_body", vec!["loop_head"]),
            ("cont", vec![]),
        ],
        vec!["entry", "loop_head", "cont", "loop_body"]
    );
    
    // Test case 3: loop3 - entry has two successors
    test_rpo("loop3",
        vec!["entry", "cont", "loop_head", "loop_body"],
        vec![
            ("entry", vec!["loop_head", "cont"]),
            ("loop_head", vec!["loop_body", "cont"]),
            ("loop_body", vec!["loop_head"]),
            ("cont", vec![]),
        ],
        vec!["entry", "loop_head", "cont", "loop_body"]
    );
    
    // Test case 4: loop4 - cont between loop_head and loop_body
    test_rpo("loop4",
        vec!["entry", "loop_head", "cont", "loop_body"],
        vec![
            ("entry", vec!["loop_head"]),
            ("loop_head", vec!["loop_body", "cont"]),
            ("loop_body", vec!["loop_head"]),
            ("cont", vec![]),
        ],
        vec!["entry", "loop_head", "cont", "loop_body"]
    );
}

fn test_rpo(
    name: &str,
    blocks: Vec<&str>,
    edges: Vec<(&str, Vec<&str>)>,
    expected: Vec<&str>
) {
    println!("Test case: {}", name);
    println!("Block order in IR: {:?}", blocks);
    
    // Create block index map
    let mut block_indices: HashMap<&str, usize> = HashMap::new();
    for (i, block) in blocks.iter().enumerate() {
        block_indices.insert(block, i);
    }
    
    // Build successors map
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    for (block, succs) in edges {
        successors.insert(block, succs);
    }
    
    // Compute RPO with source-order awareness
    let rpo = compute_rpo_with_source_order("entry", &successors, &block_indices);
    
    println!("Generated RPO: {:?}", rpo);
    println!("Expected RPO:  {:?}", expected);
    
    if rpo == expected {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }
}

fn compute_rpo_with_source_order<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    block_indices: &HashMap<&'a str, usize>
) -> Vec<&'a str> {
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    
    // Modified DFS that considers source order
    dfs_source_order(entry, successors, block_indices, &mut visited, &mut post_order);
    
    // Reverse to get RPO
    post_order.reverse();
    post_order
}

fn dfs_source_order<'a>(
    node: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>,
    block_indices: &HashMap<&'a str, usize>,
    visited: &mut HashSet<&'a str>,
    post_order: &mut Vec<&'a str>,
) {
    if visited.contains(node) {
        return;
    }
    
    visited.insert(node);
    
    // Visit successors
    if let Some(succs) = successors.get(node) {
        // Key insight: For RPO to match source order, we need to visit
        // successors in a specific way. The test seems to expect that
        // when there are multiple unvisited successors, we should prefer
        // the one that appears first in the source order.
        
        // Sort successors by their source order index
        let mut sorted_succs: Vec<_> = succs.iter().copied().collect();
        sorted_succs.sort_by_key(|s| block_indices.get(s).copied().unwrap_or(usize::MAX));
        
        for succ in sorted_succs {
            dfs_source_order(succ, successors, block_indices, visited, post_order);
        }
    }
    
    // Add to post-order after visiting all successors
    post_order.push(node);
}

// Also test standard DFS for comparison
#[allow(dead_code)]
fn compute_rpo_standard<'a>(
    entry: &'a str,
    successors: &HashMap<&'a str, Vec<&'a str>>
) -> Vec<&'a str> {
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    
    dfs_standard(entry, successors, &mut visited, &mut post_order);
    
    post_order.reverse();
    post_order
}

#[allow(dead_code)]
fn dfs_standard<'a>(
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
            dfs_standard(succ, successors, visited, post_order);
        }
    }
    
    post_order.push(node);
}