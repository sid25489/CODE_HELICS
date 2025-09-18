#!/usr/bin/env python3
"""Test analysis functionality to identify and fix issues"""

import requests
import json
import time

def test_server_status():
    """Test if server is running"""
    print("🔍 Testing Server Status")
    print("=" * 40)
    
    try:
        response = requests.get('http://localhost:5000/', timeout=10)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

def test_live_dashboard_page():
    """Test if live dashboard loads"""
    print("\n🎯 Testing Live Dashboard Page")
    print("=" * 40)
    
    try:
        response = requests.get('http://localhost:5000/live-dashboard', timeout=10)
        if response.status_code == 200:
            content = response.text
            print("✅ Live dashboard page loads")
            
            # Check for critical elements
            checks = [
                ("JavaScript class", "LiveDashboard" in content),
                ("Start button", "startAnalysis" in content),
                ("Progress ring", "progress-ring" in content),
                ("File input", "fileInput" in content),
                ("Sequence input", "sequenceInput" in content)
            ]
            
            all_good = True
            for check_name, result in checks:
                status = "✅" if result else "❌"
                print(f"{status} {check_name}")
                if not result:
                    all_good = False
            
            return all_good
        else:
            print(f"❌ Dashboard failed to load: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_analyze_dna_endpoint():
    """Test the analyze DNA endpoint"""
    print("\n🧬 Testing Analyze DNA Endpoint")
    print("=" * 40)
    
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    try:
        response = requests.post('http://localhost:5000/api/analyze-dna', 
                               json={'sequence': test_sequence},
                               timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Analysis endpoint working")
            
            if 'results' in data and data['results']:
                print(f"✅ Got {len(data['results'])} results")
                for i, result in enumerate(data['results'], 1):
                    identity = result.get('identity', 'Unknown')
                    probability = result.get('probability', 0) * 100
                    confidence = result.get('confidence', 'Unknown')
                    print(f"  {i}. {identity}: {probability:.1f}% ({confidence})")
                return True
            else:
                print("❌ No results returned")
                return False
        else:
            print(f"❌ Analysis failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def test_streaming_endpoint():
    """Test the streaming analysis endpoint"""
    print("\n🔄 Testing Streaming Analysis")
    print("=" * 40)
    
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    try:
        response = requests.get(f'http://localhost:5000/api/analyze-dna-stream?sequence={test_sequence}',
                              stream=True, timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Streaming endpoint accessible")
            
            events_received = 0
            result_received = False
            
            for line in response.iter_lines():
                if line and events_received < 10:  # Limit to prevent hanging
                    try:
                        data = json.loads(line.decode('utf-8'))
                        event = data.get('event', 'unknown')
                        
                        if event == 'progress':
                            progress = data.get('progress', 0)
                            message = data.get('message', '')
                            print(f"📊 Progress: {progress}% - {message}")
                            events_received += 1
                            
                        elif event == 'result':
                            print("✅ Final results received")
                            results = data.get('data', {}).get('results', [])
                            print(f"Got {len(results)} analysis results")
                            result_received = True
                            break
                            
                        elif event == 'error':
                            print(f"❌ Streaming error: {data.get('error')}")
                            return False
                            
                    except json.JSONDecodeError:
                        events_received += 1
                        continue
                elif events_received >= 10:
                    print("⚠️  Stopped after 10 events to prevent hanging")
                    break
            
            return result_received or events_received > 0
        else:
            print(f"❌ Streaming failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

def test_identity_matching():
    """Test identity matching functionality"""
    print("\n👤 Testing Identity Matching")
    print("=" * 40)
    
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    try:
        response = requests.post('http://localhost:5000/api/identify-person', 
                               json={'sequence': test_sequence, 'top_matches': 3},
                               timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('matches'):
                matches = data['matches']
                print(f"✅ Identity matching working - found {len(matches)} matches")
                
                high_confidence_count = 0
                for i, match in enumerate(matches, 1):
                    person_name = match.get('person_name', 'Unknown')
                    similarity = match.get('similarity_score', 0) * 100
                    confidence = match.get('confidence', 'Unknown')
                    
                    print(f"  {i}. {person_name}: {similarity:.1f}% ({confidence})")
                    
                    if confidence in ['High', 'Very High']:
                        high_confidence_count += 1
                
                print(f"High confidence matches: {high_confidence_count}/{len(matches)}")
                return True
            else:
                print(f"❌ Identity matching failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Identity matching failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Identity matching test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 DNAAadeshak Analysis Functionality Test")
    print("=" * 60)
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(5)
    
    tests = [
        ("Server Status", test_server_status),
        ("Live Dashboard Page", test_live_dashboard_page),
        ("Analyze DNA Endpoint", test_analyze_dna_endpoint),
        ("Streaming Analysis", test_streaming_endpoint),
        ("Identity Matching", test_identity_matching)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
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
        print(f"\n⚠️  Failed tests that need attention:")
        for test in failed_tests:
            print(f"  - {test}")
        print(f"\n🔧 Recommendations:")
        if "Server Status" in failed_tests:
            print("  - Check if Flask server is running")
        if "Live Dashboard Page" in failed_tests:
            print("  - Check template rendering and JavaScript")
        if "Analyze DNA Endpoint" in failed_tests:
            print("  - Check ML model and analyzer functionality")
        if "Streaming Analysis" in failed_tests:
            print("  - Check streaming endpoint and NDJSON format")
        if "Identity Matching" in failed_tests:
            print("  - Check identity matcher database and scoring")
    else:
        print("\n🎉 All tests passed! Analysis functionality is working correctly.")

if __name__ == "__main__":
    main()
