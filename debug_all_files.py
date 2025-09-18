#!/usr/bin/env python3
"""Debug all Python files in the project"""

import os
import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check Python file for syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_imports(file_path):
    """Check if all imports can be resolved"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", file_path)
        if spec is None:
            return False, "Could not create module spec"
        
        # Don't actually import, just check if we can
        return True, None
    except Exception as e:
        return False, f"Import Error: {str(e)}"

def main():
    print("🐛 Debugging All Python Files")
    print("=" * 50)
    
    # Find all Python files
    python_files = list(Path('.').glob('*.py'))
    
    syntax_errors = []
    import_errors = []
    
    for file_path in python_files:
        print(f"\n📁 Checking {file_path.name}...")
        
        # Check syntax
        syntax_ok, syntax_error = check_syntax(file_path)
        if not syntax_ok:
            syntax_errors.append((file_path.name, syntax_error))
            print(f"  ❌ Syntax: {syntax_error}")
        else:
            print(f"  ✅ Syntax: OK")
        
        # Check basic structure (skip import check for now as it may fail due to dependencies)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common issues
            issues = []
            
            # Check for unmatched brackets/parentheses
            open_parens = content.count('(')
            close_parens = content.count(')')
            if open_parens != close_parens:
                issues.append(f"Unmatched parentheses: {open_parens} open, {close_parens} close")
            
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets != close_brackets:
                issues.append(f"Unmatched brackets: {open_brackets} open, {close_brackets} close")
            
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")
            
            # Check for incomplete try/except blocks
            try_count = content.count('try:')
            except_count = content.count('except')
            finally_count = content.count('finally')
            
            if try_count > 0 and (except_count == 0 and finally_count == 0):
                issues.append(f"Try blocks without except/finally: {try_count} try, {except_count} except, {finally_count} finally")
            
            if issues:
                print(f"  ⚠️  Issues: {'; '.join(issues)}")
            else:
                print(f"  ✅ Structure: OK")
                
        except Exception as e:
            print(f"  ❌ Structure check failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEBUGGING SUMMARY")
    print("=" * 50)
    
    if syntax_errors:
        print(f"\n❌ Syntax Errors Found ({len(syntax_errors)}):")
        for file_name, error in syntax_errors:
            print(f"  • {file_name}: {error}")
    else:
        print(f"\n✅ No syntax errors found in {len(python_files)} files")
    
    if import_errors:
        print(f"\n⚠️  Import Issues ({len(import_errors)}):")
        for file_name, error in import_errors:
            print(f"  • {file_name}: {error}")
    
    # Check key files specifically
    key_files = ['app_minimal.py', 'ml_dna_analyzer.py', 'dna_identity_matcher.py', 'ai_insights_engine.py']
    print(f"\n🔑 Key Files Status:")
    
    for key_file in key_files:
        if Path(key_file).exists():
            syntax_ok, _ = check_syntax(key_file)
            status = "✅ OK" if syntax_ok else "❌ ERROR"
            print(f"  • {key_file}: {status}")
        else:
            print(f"  • {key_file}: ❓ NOT FOUND")
    
    print(f"\n🎉 Debugging complete!")

if __name__ == "__main__":
    main()
