#!/usr/bin/env python3
"""Final comprehensive debug report for DNAAadeshak"""

import os
import sys
from pathlib import Path
import traceback

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Check if all required directories and files exist"""
    print("📁 Checking Directory Structure...")
    
    files_to_check = [
        ('app_minimal.py', 'Main Flask Application'),
        ('ml_dna_analyzer.py', 'ML DNA Analyzer'),
        ('dna_identity_matcher.py', 'DNA Identity Matcher'),
        ('ai_insights_engine.py', 'AI Insights Engine'),
        ('synthetic_dna_dataset.csv', 'Enhanced DNA Dataset'),
        ('dna_model.pkl', 'Trained ML Model'),
        ('requirements.txt', 'Dependencies'),
        ('templates/base.html', 'Base HTML Template'),
        ('templates/analyze.html', 'Analysis Page Template'),
        ('templates/index.html', 'Home Page Template'),
        ('templates/ai_insights.html', 'AI Insights Template'),
        ('templates/identity_match.html', 'Identity Match Template'),
    ]
    
    missing_files = []
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            missing_files.append(filepath)
    
    return len(missing_files) == 0, missing_files

def check_python_syntax():
    """Check Python files for syntax errors"""
    print("\n🐍 Checking Python Syntax...")
    
    python_files = [
        'app_minimal.py',
        'ml_dna_analyzer.py', 
        'dna_identity_matcher.py',
        'ai_insights_engine.py'
    ]
    
    syntax_errors = []
    
    for py_file in python_files:
        if os.path.exists(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import ast
                ast.parse(content)
                print(f"  ✅ {py_file}: Syntax OK")
            except SyntaxError as e:
                error_msg = f"Line {e.lineno}: {e.msg}"
                print(f"  ❌ {py_file}: {error_msg}")
                syntax_errors.append((py_file, error_msg))
            except Exception as e:
                print(f"  ❌ {py_file}: {str(e)}")
                syntax_errors.append((py_file, str(e)))
        else:
            print(f"  ⚠️  {py_file}: File not found")
    
    return len(syntax_errors) == 0, syntax_errors

def check_imports():
    """Check if key imports work"""
    print("\n📦 Checking Key Imports...")
    
    import_tests = [
        ('Flask', 'from flask import Flask'),
        ('ML Analyzer', 'from ml_dna_analyzer import MLDNAAnalyzer'),
        ('Identity Matcher', 'from dna_identity_matcher import DNAIdentityMatcher'),
        ('AI Insights', 'from ai_insights_engine import AIInsightsEngine'),
        ('ReportLab', 'from reportlab.lib.pagesizes import letter'),
        ('Pandas', 'import pandas as pd'),
        ('NumPy', 'import numpy as np'),
        ('Scikit-learn', 'from sklearn.ensemble import RandomForestClassifier'),
    ]
    
    import_failures = []
    
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}: Import successful")
        except ImportError as e:
            print(f"  ❌ {name}: Import failed - {e}")
            import_failures.append((name, str(e)))
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
            import_failures.append((name, str(e)))
    
    return len(import_failures) == 0, import_failures

def check_functionality():
    """Test basic functionality"""
    print("\n⚙️  Testing Basic Functionality...")
    
    functionality_issues = []
    
    # Test ML Analyzer
    try:
        from ml_dna_analyzer import MLDNAAnalyzer
        analyzer = MLDNAAnalyzer()
        test_seq = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCG"
        result = analyzer.predict(test_seq)
        if result and 'class' in result:
            print(f"  ✅ ML Analyzer: Working (predicted: {result['class']})")
        else:
            print(f"  ❌ ML Analyzer: Invalid result format")
            functionality_issues.append("ML Analyzer returns invalid format")
    except Exception as e:
        print(f"  ❌ ML Analyzer: {e}")
        functionality_issues.append(f"ML Analyzer: {e}")
    
    # Test Identity Matcher
    try:
        from dna_identity_matcher import DNAIdentityMatcher
        matcher = DNAIdentityMatcher()
        if matcher.load_database():
            print(f"  ✅ Identity Matcher: Database loaded")
        else:
            print(f"  ❌ Identity Matcher: Database loading failed")
            functionality_issues.append("Identity Matcher database loading failed")
    except Exception as e:
        print(f"  ❌ Identity Matcher: {e}")
        functionality_issues.append(f"Identity Matcher: {e}")
    
    # Test AI Insights
    try:
        from ai_insights_engine import AIInsightsEngine
        ai_engine = AIInsightsEngine()
        test_seq = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCG"
        insights = ai_engine.generate_insights(test_seq)
        if insights:
            print(f"  ✅ AI Insights: Working (generated {len(insights)} insights)")
        else:
            print(f"  ❌ AI Insights: No insights generated")
            functionality_issues.append("AI Insights generates no results")
    except Exception as e:
        print(f"  ❌ AI Insights: {e}")
        functionality_issues.append(f"AI Insights: {e}")
    
    return len(functionality_issues) == 0, functionality_issues

def generate_final_report():
    """Generate comprehensive final debug report"""
    print("🔍 DNAAadeshak - Final Debug Report")
    print("=" * 60)
    
    all_issues = []
    
    # Check directory structure
    structure_ok, missing_files = check_directory_structure()
    if not structure_ok:
        all_issues.extend([f"Missing file: {f}" for f in missing_files])
    
    # Check Python syntax
    syntax_ok, syntax_errors = check_python_syntax()
    if not syntax_ok:
        all_issues.extend([f"Syntax error in {f}: {e}" for f, e in syntax_errors])
    
    # Check imports
    imports_ok, import_failures = check_imports()
    if not imports_ok:
        all_issues.extend([f"Import failure {n}: {e}" for n, e in import_failures])
    
    # Check functionality
    functionality_ok, functionality_issues = check_functionality()
    if not functionality_ok:
        all_issues.extend(functionality_issues)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL DEBUG SUMMARY")
    print("=" * 60)
    
    categories = [
        ("Directory Structure", structure_ok),
        ("Python Syntax", syntax_ok),
        ("Import Dependencies", imports_ok),
        ("Core Functionality", functionality_ok)
    ]
    
    for category, status in categories:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {category}: {'PASS' if status else 'FAIL'}")
    
    if all_issues:
        print(f"\n⚠️  Issues Found ({len(all_issues)}):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n🔧 Recommended Actions:")
        print(f"  • Fix syntax errors in Python files")
        print(f"  • Install missing dependencies: pip install -r requirements.txt")
        print(f"  • Ensure all required files are present")
        print(f"  • Check file permissions and paths")
    else:
        print(f"\n🎉 All checks passed! DNAAadeshak should be working correctly.")
        print(f"\n🚀 Ready to run:")
        print(f"  • Start the application: python app_minimal.py")
        print(f"  • Open browser: http://localhost:5000")
        print(f"  • Test all features: DNA Analysis, AI Insights, Identity Matching")
    
    return len(all_issues) == 0

if __name__ == "__main__":
    success = generate_final_report()
    sys.exit(0 if success else 1)
