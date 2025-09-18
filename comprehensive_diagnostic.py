#!/usr/bin/env python3
"""Comprehensive diagnostic tool for DNAAadeshak analysis issues"""

import requests
import json
import time
import os
import sys

def check_file_integrity():
    """Check if all required files exist and are valid"""
    print("🔍 Checking File Integrity")
    print("=" * 50)
    
    required_files = {
        'app_minimal.py': 'Main Flask application',
        'ml_dna_analyzer.py': 'ML DNA Analyzer',
        'ai_insights_engine.py': 'AI Insights Engine',
        'dna_identity_matcher.py': 'Identity Matcher',
        'dna_model.pkl': 'ML Model',
        'synthetic_dna_dataset.csv': 'Dataset',
        'templates/live_dashboard.html': 'Live Dashboard Template',
        'templates/base.html': 'Base Template'
    }
    
    all_files_ok = True
    for file_path, description in required_files.items():
        full_path = os.path.join('c:\\HACKATHON', file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✅ {description}: {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {description}: {file_path} - MISSING")
            all_files_ok = False
    
    return all_files_ok

def test_imports():
    """Test if all Python modules can be imported"""
    print("\n🐍 Testing Python Imports")
    print("=" * 50)
    
    modules_to_test = [
        'flask',
        'flask_cors',
        'ml_dna_analyzer',
        'ai_insights_engine', 
        'dna_identity_matcher',
        'pandas',
        'numpy',
        'sklearn',
        'joblib'
    ]
    
    import_issues = []
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            import_issues.append(module)
    
    return len(import_issues) == 0

def test_server_endpoints():
    """Test all critical server endpoints"""
    print("\n🌐 Testing Server Endpoints")
    print("=" * 50)
    
    base_url = 'http://localhost:5000'
    endpoints = [
        ('/', 'Home Page (Live Dashboard)'),
        ('/live-dashboard', 'Live Dashboard (legacy path)'),
        ('/analyze', 'Analyze Page'),
        ('/ai-insights', 'AI Insights'),
        ('/identity-match', 'Identity Match')
    ]
    
    endpoint_issues = []
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: {endpoint}")
            else:
                print(f"❌ {name}: {endpoint} - Status {response.status_code}")
                endpoint_issues.append(endpoint)
        except Exception as e:
            print(f"❌ {name}: {endpoint} - {str(e)}")
            endpoint_issues.append(endpoint)
    
    return len(endpoint_issues) == 0

def test_api_endpoints():
    """Test API endpoints with actual data"""
    print("\n🔧 Testing API Endpoints")
    print("=" * 50)
    
    base_url = 'http://localhost:5000'
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    api_tests = []
    
    # Test 1: Regular DNA Analysis
    try:
        response = requests.post(f"{base_url}/api/analyze-dna", 
                               json={'sequence': test_sequence},
                               timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                print(f"✅ DNA Analysis API: Got {len(data['results'])} results")
                api_tests.append(True)
            else:
                print("❌ DNA Analysis API: No results returned")
                print(f"Response: {data}")
                api_tests.append(False)
        else:
            print(f"❌ DNA Analysis API: Status {response.status_code}")
            print(f"Response: {response.text}")
            api_tests.append(False)
    except Exception as e:
        print(f"❌ DNA Analysis API: {e}")
        api_tests.append(False)
    
    # Test 2: Streaming Analysis
    try:
        response = requests.get(f"{base_url}/api/analyze-dna-stream?sequence={test_sequence}",
                              stream=True, timeout=30)
        if response.status_code == 200:
            events_count = 0
            result_found = False
            
            for line in response.iter_lines():
                if line and events_count < 10:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        event = data.get('event')
                        if event == 'result':
                            result_found = True
                            break
                        events_count += 1
                    except:
                        events_count += 1
                        continue
                elif events_count >= 10:
                    break
            
            if result_found or events_count > 0:
                print(f"✅ Streaming Analysis API: {events_count} events, result: {result_found}")
                api_tests.append(True)
            else:
                print("❌ Streaming Analysis API: No events received")
                api_tests.append(False)
        else:
            print(f"❌ Streaming Analysis API: Status {response.status_code}")
            api_tests.append(False)
    except Exception as e:
        print(f"❌ Streaming Analysis API: {e}")
        api_tests.append(False)
    
    # Test 3: Identity Matching
    try:
        response = requests.post(f"{base_url}/api/identify-person", 
                               json={'sequence': test_sequence, 'top_matches': 3},
                               timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('matches'):
                print(f"✅ Identity Matching API: Found {len(data['matches'])} matches")
                api_tests.append(True)
            else:
                print(f"❌ Identity Matching API: {data.get('error', 'No matches')}")
                api_tests.append(False)
        else:
            print(f"❌ Identity Matching API: Status {response.status_code}")
            api_tests.append(False)
    except Exception as e:
        print(f"❌ Identity Matching API: {e}")
        api_tests.append(False)
    
    return all(api_tests)

def test_javascript_functionality():
    """Test if JavaScript on live dashboard is working"""
    print("\n📜 Testing JavaScript Functionality")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:5000/live-dashboard', timeout=10)
        if response.status_code == 200:
            content = response.text
            
            js_checks = [
                ("LiveDashboard class", "class LiveDashboard" in content),
                ("Event listeners", "addEventListener" in content),
                ("Start analysis function", "startAnalysis" in content),
                ("File upload handler", "handleFileUpload" in content),
                ("Streaming fetch", "fetch(" in content and "stream" in content),
                ("Progress update", "updateProgress" in content),
                ("Results display", "displayResults" in content)
            ]
            
            all_js_ok = True
            for check_name, result in js_checks:
                status = "✅" if result else "❌"
                print(f"{status} {check_name}")
                if not result:
                    all_js_ok = False
            
            return all_js_ok
        else:
            print("❌ Could not load live dashboard page")
            return False
    except Exception as e:
        print(f"❌ JavaScript test failed: {e}")
        return False

def test_ml_model_functionality():
    """Test ML model directly"""
    print("\n🤖 Testing ML Model Functionality")
    print("=" * 50)
    
    try:
        sys.path.append('c:\\HACKATHON')
        from ml_dna_analyzer import MLDNAAnalyzer
        
        analyzer = MLDNAAnalyzer()
        model_loaded = analyzer.load_model()
        
        if model_loaded:
            print("✅ ML Model loaded successfully")
            
            # Test prediction
            test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGT"
            result = analyzer.predict(test_sequence)
            
            if result and 'class' in result:
                print(f"✅ ML Prediction working: {result['class']} ({result.get('probability', 0):.2%})")
                return True
            else:
                print("❌ ML Prediction failed")
                return False
        else:
            print("❌ ML Model failed to load")
            return False
    except Exception as e:
        print(f"❌ ML Model test failed: {e}")
        return False

def diagnose_specific_issue():
    """Try to identify the specific analysis issue"""
    print("\n🔍 Diagnosing Specific Analysis Issues")
    print("=" * 50)
    
    # Test with minimal sequence
    test_cases = [
        ("Short sequence", "ATGAAACGCAT"),
        ("Medium sequence", "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGT"),
        ("Long sequence", "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA")
    ]
    
    issues_found = []
    
    for test_name, sequence in test_cases:
        try:
            response = requests.post('http://localhost:5000/api/analyze-dna', 
                                   json={'sequence': sequence},
                                   timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    print(f"✅ {test_name}: Working ({len(data['results'])} results)")
                else:
                    print(f"❌ {test_name}: No results - {data}")
                    issues_found.append(f"{test_name}: No results returned")
            else:
                print(f"❌ {test_name}: HTTP {response.status_code} - {response.text}")
                issues_found.append(f"{test_name}: HTTP error {response.status_code}")
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")
            issues_found.append(f"{test_name}: {str(e)}")
    
    return len(issues_found) == 0, issues_found

def main():
    """Run comprehensive diagnostics"""
    print("🔧 DNAAadeshak Comprehensive Diagnostic Tool")
    print("=" * 60)
    print("This tool will identify and help fix analysis issues")
    print("=" * 60)
    
    # Wait for server
    print("Waiting for server to be ready...")
    time.sleep(3)
    
    # Run all diagnostic tests
    tests = [
        ("File Integrity", check_file_integrity),
        ("Python Imports", test_imports),
        ("Server Endpoints", test_server_endpoints),
        ("API Endpoints", test_api_endpoints),
        ("JavaScript Functionality", test_javascript_functionality),
        ("ML Model Functionality", test_ml_model_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)
    
    # Specific issue diagnosis
    analysis_working, specific_issues = diagnose_specific_issue()
    results.append(("Analysis Functionality", analysis_working))
    
    # Final Summary
    print(f"\n{'='*60}")
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed_tests = []
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed_tests.append(test_name)
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if failed_tests:
        print(f"\n⚠️  ISSUES IDENTIFIED:")
        for test in failed_tests:
            print(f"  - {test}")
        
        if specific_issues:
            print(f"\n🔍 SPECIFIC ANALYSIS ISSUES:")
            for issue in specific_issues:
                print(f"  - {issue}")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        if "File Integrity" in failed_tests:
            print("  1. Check if all required files are present")
        if "Python Imports" in failed_tests:
            print("  2. Install missing Python packages: pip install -r requirements.txt")
        if "Server Endpoints" in failed_tests:
            print("  3. Check Flask server configuration and routes")
        if "API Endpoints" in failed_tests:
            print("  4. Debug API endpoint logic and error handling")
        if "JavaScript Functionality" in failed_tests:
            print("  5. Check JavaScript console for errors in browser")
        if "ML Model Functionality" in failed_tests:
            print("  6. Retrain or reload the ML model")
        if "Analysis Functionality" in failed_tests:
            print("  7. Check sequence validation and processing logic")
    else:
        print("\n🎉 ALL DIAGNOSTICS PASSED!")
        print("The analysis functionality should be working correctly.")
        print("If you're still experiencing issues, try:")
        print("  - Clear browser cache and reload")
        print("  - Check browser console for JavaScript errors")
        print("  - Try different DNA sequences")

if __name__ == "__main__":
    main()
