#!/usr/bin/env python3
"""Test the live dashboard page"""

import requests
import time

def test_live_dashboard_page():
    """Test if the live dashboard page loads correctly"""
    print("🔍 Testing Live Dashboard Page")
    print("=" * 40)
    
    try:
        # Test the live dashboard route
        response = requests.get('http://localhost:5000/live-dashboard', timeout=10)
        print(f"Live Dashboard Status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            print(f"✅ Page loaded successfully ({len(content)} characters)")
            
            # Check for key elements
            checks = [
                ("Base template", "{% extends \"base.html\" %}" in content),
                ("Title block", "Live Analysis Dashboard" in content),
                ("Progress ring", "progress-ring" in content),
                ("JavaScript class", "LiveDashboard" in content),
                ("Start button", "startAnalysis" in content),
                ("File input", "fileInput" in content)
            ]
            
            for check_name, result in checks:
                status = "✅" if result else "❌"
                print(f"{status} {check_name}: {'Found' if result else 'Missing'}")
            
            # Test results route (should now point to live dashboard)
            response2 = requests.get('http://localhost:5000/results', timeout=10)
            print(f"\nResults Route Status: {response2.status_code}")
            
            if response2.status_code == 200:
                print("✅ Results route successfully redirects to live dashboard")
                return True
            else:
                print("❌ Results route failed")
                return False
                
        else:
            print(f"❌ Page failed to load: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_streaming_endpoint():
    """Test if the streaming endpoint is working"""
    print("\n🔄 Testing Streaming Endpoint")
    print("=" * 40)
    
    try:
        test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGT"
        response = requests.get(
            f'http://localhost:5000/api/analyze-dna-stream?sequence={test_sequence}',
            stream=True, 
            timeout=15
        )
        
        print(f"Streaming Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Streaming endpoint accessible")
            
            # Read first few lines
            lines_read = 0
            for line in response.iter_lines():
                if line and lines_read < 3:
                    try:
                        import json
                        data = json.loads(line.decode('utf-8'))
                        event = data.get('event', 'unknown')
                        print(f"✅ Received event: {event}")
                        lines_read += 1
                    except:
                        print(f"✅ Received data: {line.decode('utf-8')[:50]}...")
                        lines_read += 1
                elif lines_read >= 3:
                    break
            
            return True
        else:
            print(f"❌ Streaming failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Live Dashboard Functionality")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server...")
    time.sleep(3)
    
    tests = [
        ("Live Dashboard Page", test_live_dashboard_page),
        ("Streaming Endpoint", test_streaming_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
        
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")

if __name__ == "__main__":
    main()
