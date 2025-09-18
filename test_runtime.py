#!/usr/bin/env python3
"""Test runtime functionality of key components"""

import sys
import traceback

def test_imports():
    """Test if all key imports work"""
    print("🔍 Testing Imports...")
    
    try:
        from ml_dna_analyzer import MLDNAAnalyzer
        print("  ✅ MLDNAAnalyzer imported successfully")
    except Exception as e:
        print(f"  ❌ MLDNAAnalyzer import failed: {e}")
        return False
    
    try:
        from ai_insights_engine import AIInsightsEngine
        print("  ✅ AIInsightsEngine imported successfully")
    except Exception as e:
        print(f"  ❌ AIInsightsEngine import failed: {e}")
        return False
    
    try:
        from dna_identity_matcher import DNAIdentityMatcher
        print("  ✅ DNAIdentityMatcher imported successfully")
    except Exception as e:
        print(f"  ❌ DNAIdentityMatcher import failed: {e}")
        return False
    
    return True

def test_ml_analyzer():
    """Test ML DNA Analyzer functionality"""
    print("\n🧬 Testing ML DNA Analyzer...")
    
    try:
        from ml_dna_analyzer import MLDNAAnalyzer
        analyzer = MLDNAAnalyzer()
        
        # Test basic functionality
        test_sequence = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCG"
        result = analyzer.predict(test_sequence)
        
        if result and 'class' in result:
            print(f"  ✅ ML prediction works: {result['class']} ({result['probability']:.2%})")
            return True
        else:
            print(f"  ❌ ML prediction returned invalid result: {result}")
            return False
            
    except Exception as e:
        print(f"  ❌ ML Analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_identity_matcher():
    """Test DNA Identity Matcher functionality"""
    print("\n👤 Testing DNA Identity Matcher...")
    
    try:
        from dna_identity_matcher import DNAIdentityMatcher
        matcher = DNAIdentityMatcher()
        
        # Test database loading
        if matcher.load_database():
            print("  ✅ Database loaded successfully")
            
            # Test basic matching
            test_sequence = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCGATCGATCGATCGTAGCTAGCTAGCTAATCGATCG"
            matches = matcher.identify_person(test_sequence, top_n=3)
            
            if matches and len(matches) > 0:
                print(f"  ✅ Identity matching works: Found {len(matches)} matches")
                return True
            else:
                print(f"  ❌ Identity matching returned no results")
                return False
        else:
            print("  ❌ Database loading failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Identity Matcher test failed: {e}")
        traceback.print_exc()
        return False

def test_ai_insights():
    """Test AI Insights Engine functionality"""
    print("\n🤖 Testing AI Insights Engine...")
    
    try:
        from ai_insights_engine import AIInsightsEngine
        ai_engine = AIInsightsEngine()
        
        # Test basic functionality
        test_sequence = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCG"
        insights = ai_engine.generate_insights(test_sequence)
        
        if insights and len(insights) > 0:
            print(f"  ✅ AI insights generation works: Generated {len(insights)} insights")
            return True
        else:
            print(f"  ❌ AI insights returned no results")
            return False
            
    except Exception as e:
        print(f"  ❌ AI Insights test failed: {e}")
        traceback.print_exc()
        return False

def test_pdf_generation():
    """Test PDF generation functionality"""
    print("\n📄 Testing PDF Generation...")
    
    try:
        # Import the function from app_minimal
        sys.path.append('.')
        from app_minimal import generate_pdf_report
        
        # Test data
        test_results = [
            {'identity': 'Gene', 'probability': 0.85, 'confidence': 'High'},
            {'identity': 'Promoter', 'probability': 0.12, 'confidence': 'Low'},
            {'identity': 'Junk', 'probability': 0.03, 'confidence': 'Very Low'}
        ]
        
        test_sequence_info = {
            'length': 100,
            'source': 'Test Input'
        }
        
        pdf_buffer = generate_pdf_report(test_results, test_sequence_info)
        
        if pdf_buffer and pdf_buffer.getvalue():
            print(f"  ✅ PDF generation works: Generated {len(pdf_buffer.getvalue())} bytes")
            return True
        else:
            print(f"  ❌ PDF generation returned empty buffer")
            return False
            
    except Exception as e:
        print(f"  ❌ PDF generation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🧪 Runtime Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("ML Analyzer", test_ml_analyzer),
        ("Identity Matcher", test_identity_matcher),
        ("AI Insights", test_ai_insights),
        ("PDF Generation", test_pdf_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 RUNTIME TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All runtime tests passed! Application should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
