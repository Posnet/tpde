#!/usr/bin/env python3
"""Replace println! statements with appropriate logging calls."""

import re
import sys
from pathlib import Path

# Patterns to identify different types of println usage
PATTERNS = [
    # Debug/trace messages with emojis
    (r'println!\("(🔧|🔨|⚙️|🛠️).*?"', 'log::info!("{}', 'Compilation progress'),
    (r'println!\("(📊|📈|📉|📍).*?"', 'log::debug!("{}', 'Analysis/stats'),
    (r'println!\("(📦|📥|📤).*?"', 'log::trace!("{}', 'Block/instruction details'),
    (r'println!\("(➕|➖|✖️|➗|🔢).*?"', 'log::trace!("{}', 'Arithmetic operations'),
    (r'println!\("(🔍|🔎|🔬).*?"', 'log::trace!("{}', 'Comparison/search'),
    (r'println!\("(⚠️|⚡|❌|❗).*?"', 'log::warn!("{}', 'Warnings'),
    (r'println!\("(✅|✔️|☑️).*?"', 'log::debug!("{}', 'Success/completion'),
    (r'println!\("(🔄|↩️|↪️|🔃).*?"', 'log::trace!("{}', 'Control flow'),
    (r'println!\("(📞|🔔|📲).*?"', 'log::trace!("{}', 'Function calls'),
    (r'println!\("(🔙|⬅️|➡️|↔️).*?"', 'log::trace!("{}', 'Returns/jumps'),
    (r'println!\("(🪧|🏷️|📌).*?"', 'log::trace!("{}', 'Labels/markers'),
    
    # Messages with "Generated:" or "Emit:"
    (r'println!\("\s*Generated:', 'log::trace!("   Generated:', 'Code generation'),
    (r'println!\("\s*Emit:', 'log::trace!("   Emit:', 'Code emission'),
    
    # PHI node messages
    (r'println!\("\s*PHI', 'log::trace!("   PHI', 'PHI operations'),
    
    # Test success messages (keep as println for test output)
    (r'println!\(".*test.*passed.*?"', None, 'Test output - keep as println'),
    (r'println!\(".*Test.*complete.*?"', None, 'Test output - keep as println'),
]

def process_file(filepath):
    """Process a single file and replace println statements."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    replacements = []
    
    # First, handle multi-line println statements
    # Match println!("...", args) including multi-line
    println_pattern = re.compile(
        r'println!\s*\(\s*"([^"]*?)"\s*(,\s*[^)]+?)?\s*\)',
        re.MULTILINE | re.DOTALL
    )
    
    for match in println_pattern.finditer(content):
        full_match = match.group(0)
        message = match.group(1)
        args = match.group(2) if match.group(2) else ""
        
        # Check each pattern
        replacement = None
        category = None
        for pattern, repl, cat in PATTERNS:
            if re.search(pattern.replace('println!\\("', ''), '"' + message + '"'):
                replacement = repl
                category = cat
                break
        
        if replacement is None:
            # Check for specific keywords
            message_lower = message.lower()
            if any(word in message_lower for word in ['compiling', 'compile', 'processing']):
                replacement = 'log::info!("{}'
                category = 'Compilation progress'
            elif any(word in message_lower for word in ['generated', 'emit', 'instruction']):
                replacement = 'log::trace!("{}'
                category = 'Code generation'
            elif any(word in message_lower for word in ['found', 'detected', 'analyzing']):
                replacement = 'log::debug!("{}'
                category = 'Analysis'
            elif 'unsupported' in message_lower or 'warning' in message_lower:
                replacement = 'log::warn!("{}'
                category = 'Warning'
            elif re.search(r'(Block|block) \d+', message):
                replacement = 'log::trace!("{}'
                category = 'Block details'
            else:
                # Default to debug for other internal messages
                if not any(word in message_lower for word in ['error', 'result', 'output']):
                    replacement = 'log::debug!("{}'
                    category = 'General debug'
        
        if replacement:
            new_call = full_match.replace('println!("', replacement)
            replacements.append((full_match, new_call, category))
    
    # Apply replacements
    for old, new, category in replacements:
        content = content.replace(old, new)
        print(f"  Replaced: {category}")
    
    # Check if file needs log import
    if replacements and 'use log::' not in content:
        # Add log import after the first use statement block or at the top
        lines = content.split('\n')
        insert_idx = 0
        in_use_block = False
        
        for i, line in enumerate(lines):
            if line.startswith('use '):
                in_use_block = True
                insert_idx = i + 1
            elif in_use_block and not line.startswith('use ') and line.strip():
                insert_idx = i
                break
        
        if insert_idx == 0:
            # No use statements found, add after module docs
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('//'):
                    insert_idx = i
                    break
        
        lines.insert(insert_idx, '')
        content = '\n'.join(lines)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return len(replacements)
    return 0

def main():
    src_dir = Path('src')
    total_replacements = 0
    
    # Process all .rs files
    for rs_file in src_dir.rglob('*.rs'):
        # Skip test files and bin files
        if 'test' in str(rs_file) or rs_file.parent.name == 'bin':
            continue
            
        print(f"\nProcessing {rs_file}...")
        count = process_file(rs_file)
        total_replacements += count
        if count > 0:
            print(f"  Made {count} replacements")
    
    print(f"\nTotal replacements: {total_replacements}")

if __name__ == '__main__':
    main()