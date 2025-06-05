#!/usr/bin/env python3
"""
Find duplicate function definitions in Rust files to help fix compilation errors.
"""

import re
import os
import sys
from collections import defaultdict

def extract_functions(file_path):
    """Extract function signatures from a Rust file."""
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match function definitions
        # Matches: pub fn name(...) -> ReturnType or fn name(...) -> ReturnType
        pattern = r'^\s*(pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)(?:\s*->\s*[^{]+)?'
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                visibility = match.group(1) or ""
                func_name = match.group(2)
                functions.append({
                    'name': func_name,
                    'signature': line.strip(),
                    'line': i,
                    'file': file_path
                })
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return functions

def find_duplicate_functions(directory):
    """Find all duplicate function names in Rust files."""
    
    # Find all .rs files
    rust_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))
    
    print(f"Scanning {len(rust_files)} Rust files...\n")
    
    # Extract functions from all files
    all_functions = []
    file_function_counts = defaultdict(lambda: defaultdict(int))
    
    for file_path in rust_files:
        functions = extract_functions(file_path)
        all_functions.extend(functions)
        
        # Count functions per file
        for func in functions:
            file_function_counts[file_path][func['name']] += 1
    
    # Find duplicates within files
    print("=== DUPLICATE FUNCTIONS WITHIN SAME FILE ===")
    found_file_duplicates = False
    
    for file_path, func_counts in file_function_counts.items():
        duplicates = {name: count for name, count in func_counts.items() if count > 1}
        if duplicates:
            found_file_duplicates = True
            print(f"\n📁 {os.path.relpath(file_path, directory)}:")
            
            for func_name, count in duplicates.items():
                print(f"  ❌ {func_name} appears {count} times")
                
                # Show line numbers for each occurrence
                matching_functions = [f for f in all_functions 
                                    if f['file'] == file_path and f['name'] == func_name]
                for func in matching_functions:
                    print(f"     Line {func['line']}: {func['signature']}")
    
    if not found_file_duplicates:
        print("✅ No duplicate functions found within individual files")
    
    # Find functions with same name across different files
    print("\n\n=== FUNCTIONS WITH SAME NAME ACROSS FILES ===")
    func_locations = defaultdict(list)
    
    for func in all_functions:
        func_locations[func['name']].append(func)
    
    found_cross_file_duplicates = False
    cross_file_duplicates = {name: funcs for name, funcs in func_locations.items() 
                           if len(set(f['file'] for f in funcs)) > 1}
    
    for func_name, funcs in cross_file_duplicates.items():
        if len(funcs) > 1:
            found_cross_file_duplicates = True
            files = set(f['file'] for f in funcs)
            if len(files) > 1:  # Only show if across different files
                print(f"\n🔄 {func_name} appears in {len(files)} files:")
                for func in funcs:
                    rel_path = os.path.relpath(func['file'], directory)
                    print(f"     {rel_path}:{func['line']} - {func['signature']}")
    
    if not found_cross_file_duplicates:
        print("✅ No duplicate function names found across different files")
    
    # Summary
    print(f"\n\n=== SUMMARY ===")
    print(f"Total functions found: {len(all_functions)}")
    print(f"Files scanned: {len(rust_files)}")
    
    if found_file_duplicates or found_cross_file_duplicates:
        print("❌ Duplicates found - see details above")
        return 1
    else:
        print("✅ No problematic duplicates found")
        return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "/Users/entropy/Developer/tpde/rust/tpde-core/src"
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    print(f"🔍 Searching for duplicate functions in: {directory}")
    sys.exit(find_duplicate_functions(directory))