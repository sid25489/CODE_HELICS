#!/usr/bin/env python3
"""Check and fix core functionality errors"""

import os
import sys
import traceback

def check_and_fix_ml_analyzer():
    """Check ML Analyzer functionality"""
    print("🧬 Checking ML DNA Analyzer...")
    
    try:
        from ml_dna_analyzer import MLDNAAnalyzer
        analyzer = MLDNAAnalyzer()
        
        # Test basic prediction
        test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
        
        result = analyzer.predict(test_sequence)
        
        if result and 'class' in result and 'probability' in result:
            print(f"  ✅ ML Analyzer working: {result['class']} ({result['probability']:.2%})")
            return True, None
        else:
            error = f"Invalid result format: {result}"
            print(f"  ❌ ML Analyzer error: {error}")
            return False, error
            
    except Exception as e:
        error = f"ML Analyzer failed: {str(e)}"
        print(f"  ❌ {error}")
        traceback.print_exc()
        return False, error

def check_and_fix_identity_matcher():
    """Check Identity Matcher functionality"""
    print("\n👤 Checking DNA Identity Matcher...")
    
    try:
        from dna_identity_matcher import DNAIdentityMatcher
        matcher = DNAIdentityMatcher()
        
        # Test database loading
        if not matcher.load_database():
            error = "Database loading failed"
            print(f"  ❌ {error}")
            return False, error
        
        print("  ✅ Database loaded successfully")
        
        # Test identity matching
        test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
        
        matches = matcher.identify_person(test_sequence, top_n=3)
        
        if matches and len(matches) > 0:
            print(f"  ✅ Identity matching working: Found {len(matches)} matches")
            
            # Test database stats
            stats = matcher.get_database_stats()
            if stats:
                print(f"  ✅ Database stats working: {stats['total_records']} records")
                return True, None
            else:
                error = "Database stats failed"
                print(f"  ❌ {error}")
                return False, error
        else:
            error = "No identity matches found"
            print(f"  ❌ {error}")
            return False, error
            
    except Exception as e:
        error = f"Identity Matcher failed: {str(e)}"
        print(f"  ❌ {error}")
        traceback.print_exc()
        return False, error

def check_and_fix_ai_insights():
    """Check AI Insights functionality"""
    print("\n🤖 Checking AI Insights Engine...")
    
    try:
        from ai_insights_engine import AIInsightsEngine
        ai_engine = AIInsightsEngine()
        
        # Test insights generation
        test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
        
        insights = ai_engine.generate_insights(test_sequence)
        
        if insights and len(insights) > 0:
            print(f"  ✅ AI Insights working: Generated {len(insights)} insights")
            return True, None
        else:
            error = "No AI insights generated"
            print(f"  ❌ {error}")
            return False, error
            
    except Exception as e:
        error = f"AI Insights failed: {str(e)}"
        print(f"  ❌ {error}")
        traceback.print_exc()
        return False, error

def check_and_fix_flask_app():
    """Check Flask app configuration"""
    print("\n🌐 Checking Flask Application...")
    
    try:
        # Import the app
        from app_minimal import app, analyzer, ai_engine, identity_matcher
        
        print("  ✅ Flask app imported successfully")
        
        # Check if components are initialized
        if analyzer:
            print("  ✅ ML Analyzer initialized")
        else:
            print("  ❌ ML Analyzer not initialized")
            return False, "ML Analyzer not initialized"
        
        if ai_engine:
            print("  ✅ AI Engine initialized")
        else:
            print("  ❌ AI Engine not initialized")
            return False, "AI Engine not initialized"
        
        if identity_matcher:
            print("  ✅ Identity Matcher initialized")
        else:
            print("  ❌ Identity Matcher not initialized")
            return False, "Identity Matcher not initialized"
        
        # Test PDF generation
        from app_minimal import generate_pdf_report
        test_results = [
            {'identity': 'Gene', 'probability': 0.85, 'confidence': 'High'},
            {'identity': 'Promoter', 'probability': 0.12, 'confidence': 'Low'},
            {'identity': 'Junk', 'probability': 0.03, 'confidence': 'Very Low'}
        ]
        test_sequence_info = {'length': 100, 'source': 'Test'}
        
        pdf_buffer = generate_pdf_report(test_results, test_sequence_info)
        if pdf_buffer and pdf_buffer.getvalue():
            print(f"  ✅ PDF generation working: {len(pdf_buffer.getvalue())} bytes")
            return True, None
        else:
            error = "PDF generation failed"
            print(f"  ❌ {error}")
            return False, error
            
    except Exception as e:
        error = f"Flask app check failed: {str(e)}"
        print(f"  ❌ {error}")
        traceback.print_exc()
        return False, error

def check_required_files():
    """Check if all required files exist"""
    print("\n📁 Checking Required Files...")
    
    required_files = [
        'synthetic_dna_dataset.csv',
        'dna_model.pkl',
        'templates/base.html',
        'templates/analyze.html',
        'templates/index.html',
        'templates/ai_insights.html',
        'templates/identity_match.html'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    if missing_files:
        return False, f"Missing files: {', '.join(missing_files)}"
    else:
        return True, None

def fix_enhanced_dna_analyzer():
    """Fix the EnhancedDNAAnalyzer class reference"""
    print("\n🔧 Fixing EnhancedDNAAnalyzer reference...")
    
    try:
        # Check if EnhancedDNAAnalyzer is defined in app_minimal.py
        with open('app_minimal.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'class EnhancedDNAAnalyzer' not in content and 'EnhancedDNAAnalyzer()' in content:
            print("  🔧 Fixing EnhancedDNAAnalyzer reference...")
            
            # Replace EnhancedDNAAnalyzer with MLDNAAnalyzer
            content = content.replace('EnhancedDNAAnalyzer()', 'MLDNAAnalyzer()')
            content = content.replace('EnhancedDNAAnalyzer', 'MLDNAAnalyzer')
            
            with open('app_minimal.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("  ✅ Fixed EnhancedDNAAnalyzer reference")
            return True, None
        else:
            print("  ✅ EnhancedDNAAnalyzer reference is correct")
            return True, None
            
    except Exception as e:
        error = f"Failed to fix EnhancedDNAAnalyzer: {str(e)}"
        print(f"  ❌ {error}")
        return False, error

def main():
    """Main function to check and fix core functionality"""
    print("🔍 DNAAadeshak - Core Functionality Check & Fix")
    print("=" * 60)
    
    checks = [
        ("Required Files", check_required_files),
        ("EnhancedDNAAnalyzer Fix", fix_enhanced_dna_analyzer),
        ("ML DNA Analyzer", check_and_fix_ml_analyzer),
        ("Identity Matcher", check_and_fix_identity_matcher),
        ("AI Insights Engine", check_and_fix_ai_insights),
        ("Flask Application", check_and_fix_flask_app)
    ]
    
    results = []
    errors = []
    
    for check_name, check_func in checks:
        try:
            success, error = check_func()
            results.append((check_name, success))
            if not success and error:
                errors.append(f"{check_name}: {error}")
        except Exception as e:
            print(f"  ❌ {check_name} check crashed: {e}")
            results.append((check_name, False))
            errors.append(f"{check_name}: Check crashed - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CORE FUNCTIONALITY SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if errors:
        print(f"\n⚠️  Errors Found ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        
        print(f"\n🔧 Recommended Fixes:")
        print(f"  • Install missing dependencies: pip install -r requirements.txt")
        print(f"  • Retrain ML model if missing: python retrain_model.py")
        print(f"  • Check file paths and permissions")
        print(f"  • Verify dataset integrity")
    else:
        print(f"\n🎉 All core functionality checks passed!")
        print(f"\n🚀 System is ready to run:")
        print(f"  • Start: python app_minimal.py")
        print(f"  • URL: http://localhost:5000")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
