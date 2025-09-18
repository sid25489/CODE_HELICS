#!/usr/bin/env python3
"""Quick API test to identify issues"""

import requests
import json

def test_analyze_api():
    """Test the analyze API"""
    print("Testing DNA Analysis API...")
    
    try:
        response = requests.post('http://localhost:5000/api/analyze-dna', 
                               json={'sequence': 'ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGT'},
                               timeout=10)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_streaming_api():
    """Test the streaming API"""
    print("\nTesting Streaming API...")
    
    try:
        response = requests.get('http://localhost:5000/api/analyze-dna-stream?sequence=ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGT',
                              stream=True, timeout=10)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            count = 0
            for line in response.iter_lines():
                if line and count < 5:
                    print(f"Stream data: {line.decode('utf-8')}")
                    count += 1
                elif count >= 5:
                    break
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_dashboard_page():
    """Test dashboard page"""
    print("\nTesting Dashboard Page...")
    
    try:
        response = requests.get('http://localhost:5000/live-dashboard', timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            if "LiveDashboard" in content and "startAnalysis" in content:
                print("✅ Dashboard page loads with JavaScript")
                return True
            else:
                print("❌ Dashboard missing JavaScript components")
                return False
        else:
            print(f"Error loading dashboard: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Quick API Test")
    print("=" * 30)
    
    api_ok = test_analyze_api()
    stream_ok = test_streaming_api()
    dashboard_ok = test_dashboard_page()
    
    print(f"\nResults:")
    print(f"API: {'✅' if api_ok else '❌'}")
    print(f"Streaming: {'✅' if stream_ok else '❌'}")
    print(f"Dashboard: {'✅' if dashboard_ok else '❌'}")
