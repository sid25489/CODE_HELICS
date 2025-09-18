#!/usr/bin/env python3
"""Test script to identify import and dependency issues"""

import sys
import traceback

def test_import(module_name, package_name=None):
    try:
        if package_name:
            exec(f"from {package_name} import {module_name}")
        else:
            exec(f"import {module_name}")
        print(f"✓ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} - ERROR: {e}")
        return False

def main():
    print("Testing Python imports for DNA Analyzer...")
    print("=" * 50)
    
    # Test basic imports
    modules = [
        ('os', None),
        ('re', None),
        ('json', None),
        ('collections', None),
        ('flask', None),
        ('pandas', None),
        ('numpy', None),
        ('sklearn', None),
        ('joblib', None),
        ('dotenv', 'python-dotenv')
    ]
    
    failed = []
    for module, package in modules:
        if not test_import(module):
            failed.append((module, package))
    
    print("\n" + "=" * 50)
    if failed:
        print("FAILED IMPORTS:")
        for module, package in failed:
            pkg_name = package if package else module
            print(f"  pip install {pkg_name}")
    else:
        print("All imports successful!")
    
    # Test Flask app syntax
    print("\nTesting Flask app syntax...")
    try:
        with open('app.py', 'r') as f:
            code = f.read()
        compile(code, 'app.py', 'exec')
        print("✓ app.py syntax - OK")
    except SyntaxError as e:
        print(f"✗ app.py syntax - ERROR: {e}")
    except Exception as e:
        print(f"✗ app.py - ERROR: {e}")

if __name__ == "__main__":
    main()
