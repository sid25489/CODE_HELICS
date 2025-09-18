#!/usr/bin/env python3
"""Test DNA analyze feature"""

import requests
import json

def test_dna_analyze():
    """Test the DNA analyze endpoint"""
    print("🧬 Testing DNA Analyze Feature")
    print("=" * 40)
    
    # Test data
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    # Test JSON endpoint
    print("Testing JSON endpoint...")
    try:
        response = requests.post('http://localhost:5000/api/analyze-dna', 
                               json={'sequence': test_sequence},
                               timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ JSON Analysis successful")
            print(f"Results: {len(data.get('results', []))} items")
            if data.get('results'):
                for result in data['results']:
                    print(f"  - {result.get('identity', 'Unknown')}: {result.get('probability', 0):.2%}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ JSON test failed: {e}")
    
    # Test streaming endpoint
    print("\nTesting streaming endpoint...")
    try:
        response = requests.get(f'http://localhost:5000/api/analyze-dna-stream?sequence={test_sequence}',
                              stream=True, timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Streaming started")
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if data.get('event') == 'result':
                            print(f"✅ Streaming result received")
                            results = data.get('data', {}).get('results', [])
                            print(f"Results: {len(results)} items")
                            break
                        elif data.get('event') == 'progress':
                            print(f"Progress: {data.get('progress', 0)}%")
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"❌ Streaming error: {response.text}")
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")

if __name__ == "__main__":
    test_dna_analyze()
