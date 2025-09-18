#!/usr/bin/env python3
"""Test the live dashboard features"""

import requests
import json
import time

def test_identity_matching():
    """Test enhanced identity matching"""
    print("🧬 Testing Enhanced Identity Matching")
    print("=" * 50)
    
    # Test sequence (longer for better matching)
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    try:
        # Test identity matching with 3 top matches
        response = requests.post('http://localhost:5000/api/identify-person', 
                               json={
                                   'sequence': test_sequence,
                                   'top_matches': 3
                               },
                               timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Identity matching successful")
            
            matches = data.get('matches', [])
            print(f"Found {len(matches)} matches:")
            
            high_confidence_count = 0
            for i, match in enumerate(matches, 1):
                person_name = match.get('person_name', 'Unknown')
                similarity = match.get('similarity_score', 0) * 100
                confidence = match.get('confidence', 'Unknown')
                
                print(f"  {i}. {person_name}")
                print(f"     Similarity: {similarity:.1f}%")
                print(f"     Confidence: {confidence}")
                print(f"     DOB: {match.get('metadata', {}).get('dob', 'N/A')}")
                print(f"     Mobile: {match.get('metadata', {}).get('mobile', 'N/A')}")
                print()
                
                if confidence in ['High', 'Very High']:
                    high_confidence_count += 1
            
            print(f"High confidence matches: {high_confidence_count}/{len(matches)}")
            
            if high_confidence_count >= 2:
                print("✅ Successfully returning 2+ high-confidence matches")
                return True
            else:
                print("⚠️  Less than 2 high-confidence matches found")
                return False
                
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_streaming_analysis():
    """Test streaming analysis endpoint"""
    print("🔄 Testing Streaming Analysis")
    print("=" * 50)
    
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    try:
        response = requests.get(f'http://localhost:5000/api/analyze-dna-stream?sequence={test_sequence}',
                              stream=True, timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Streaming started")
            
            progress_updates = 0
            results_received = False
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        event = data.get('event')
                        
                        if event == 'progress':
                            progress = data.get('progress', 0)
                            message = data.get('message', '')
                            print(f"Progress: {progress}% - {message}")
                            progress_updates += 1
                            
                        elif event == 'result':
                            print("✅ Final results received")
                            results = data.get('data', {}).get('results', [])
                            print(f"Analysis results: {len(results)} items")
                            
                            for result in results:
                                identity = result.get('identity', 'Unknown')
                                probability = result.get('probability', 0) * 100
                                confidence = result.get('confidence', 'Unknown')
                                print(f"  - {identity}: {probability:.1f}% ({confidence})")
                            
                            results_received = True
                            break
                            
                        elif event == 'error':
                            print(f"❌ Streaming error: {data.get('error')}")
                            return False
                            
                    except json.JSONDecodeError:
                        continue
            
            if progress_updates > 0 and results_received:
                print(f"✅ Streaming test successful ({progress_updates} progress updates)")
                return True
            else:
                print("❌ Streaming test incomplete")
                return False
                
        else:
            print(f"❌ Streaming error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Live Dashboard Features")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Identity Matching", test_identity_matching),
        ("Streaming Analysis", test_streaming_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
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
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All live dashboard features are working!")
    else:
        print("⚠️  Some features need attention")

if __name__ == "__main__":
    main()
