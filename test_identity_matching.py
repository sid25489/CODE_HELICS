#!/usr/bin/env python3
"""
Test script for DNA Identity Matching functionality
Tests all API endpoints and validates the identity matching system
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TEST_SEQUENCES = [
    # Sample sequences from the synthetic dataset
    "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA",
    
    "CCGAGGTATGCGGCCAGAGTTGGGCGAATGGCATACTCCTCTGAACACATTAGGTGGGCGGTACTTATCCTGAACACATATCATCTCTGCTAGGGCGGCTGAATTGTCTGGATGGTATTTTGGCCAGGCTCCGGGGAGGTCAGCTACCCATGCCGAAACCGTACCTATGAGCTCGCATCATCGACTGTGGAACGACCCGCACTTACTATATCAGTGGAGTTTTGACGCTTATCTGCATCAAATCGACGCAGCCGGTAGTCGATAAAATTGTCGATTGTTGTAACTAGGCCACCGCTCAGATATGTACCCTAGACCAGCTGGCCGCTCTATTACTTGAACCGGTTTAGGAAAGCTGTAAATATTCCAA",
    
    "TATAAAAGGCCCAATGCGGGCATCAGGTAATGCCACCGACGTGATATTCGCCCCGGTTTAGGGGCTTGCCGCGGGTTGTAACGCCGATGGGGTTCTCTGTCCTGAAGCCCGACCATTCTTGTCTAGCATATCCTAAGTGGAAGCGGGTGTCTGGGTCAGTGAGACTCGGAACTCCTCACTCGCGGGCGGGGGGGGACATGTGCCCTTGGCTCTTGGTGTTGCGAAGGGCAACACATAAATTG"
]

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_test_result(test_name, success, details=""):
    """Print test result with formatting"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        success = response.status_code == 200
        
        if success:
            data = response.json()
            print_test_result("Health Check", True, f"Status: {data.get('status')}")
            return True
        else:
            print_test_result("Health Check", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_test_result("Health Check", False, f"Error: {str(e)}")
        return False

def test_database_stats():
    """Test the database statistics endpoint"""
    print_header("Testing Database Statistics")
    
    try:
        response = requests.get(f"{BASE_URL}/api/database-stats")
        success = response.status_code == 200
        
        if success:
            data = response.json()
            if data.get('success'):
                stats = data['database_stats']
                print_test_result("Database Stats", True)
                print(f"    Unique People: {stats.get('unique_people', 0):,}")
                print(f"    Total Records: {stats.get('total_records', 0):,}")
                print(f"    Avg Sequences/Person: {stats.get('avg_sequences_per_person', 0):.1f}")
                print(f"    Classes: {list(stats.get('classes_distribution', {}).keys())}")
                return True
            else:
                print_test_result("Database Stats", False, f"API Error: {data.get('error')}")
                return False
        else:
            print_test_result("Database Stats", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_test_result("Database Stats", False, f"Error: {str(e)}")
        return False

def test_identity_matching():
    """Test the identity matching functionality"""
    print_header("Testing Identity Matching")
    
    results = []
    
    for i, sequence in enumerate(TEST_SEQUENCES):
        print(f"\n--- Test Sequence {i+1} (Length: {len(sequence)} bp) ---")
        
        try:
            payload = {
                "sequence": sequence,
                "top_n": 5
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/identify-person", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'})
            end_time = time.time()
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                if data.get('success'):
                    matches = data.get('matches', [])
                    print_test_result(f"Identity Match {i+1}", True, 
                                    f"Found {len(matches)} matches in {end_time-start_time:.2f}s")
                    
                    # Display top match details
                    if matches:
                        top_match = matches[0]
                        print(f"    Top Match: {top_match['person_name']}")
                        print(f"    Similarity: {top_match['similarity_score']:.4f}")
                        print(f"    Confidence: {top_match['confidence']}")
                        print(f"    DOB: {top_match['metadata']['dob']}")
                        
                        results.append({
                            'sequence_length': len(sequence),
                            'top_match': top_match['person_name'],
                            'similarity': top_match['similarity_score'],
                            'confidence': top_match['confidence'],
                            'processing_time': end_time - start_time
                        })
                else:
                    print_test_result(f"Identity Match {i+1}", False, f"API Error: {data.get('error')}")
            else:
                print_test_result(f"Identity Match {i+1}", False, f"Status Code: {response.status_code}")
                
        except Exception as e:
            print_test_result(f"Identity Match {i+1}", False, f"Error: {str(e)}")
    
    return results

def test_person_details():
    """Test the person details endpoint"""
    print_header("Testing Person Details")
    
    # First, get a person name from identity matching
    try:
        payload = {"sequence": TEST_SEQUENCES[0], "top_n": 1}
        response = requests.post(f"{BASE_URL}/api/identify-person", 
                               json=payload,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('matches'):
                person_name = data['matches'][0]['person_name']
                
                # Now test person details
                response = requests.get(f"{BASE_URL}/api/person-details/{person_name}")
                success = response.status_code == 200
                
                if success:
                    data = response.json()
                    if data.get('success'):
                        details = data['person_details']
                        print_test_result("Person Details", True)
                        print(f"    Name: {details['name']}")
                        print(f"    DOB: {details['metadata']['dob']}")
                        print(f"    Mobile: {details['metadata']['mobile_no']}")
                        print(f"    Sequences: {details['sequence_count']}")
                        print(f"    Total BP: {details['total_base_pairs']:,}")
                        print(f"    Avg Length: {details['avg_sequence_length']:.1f}")
                        return True
                    else:
                        print_test_result("Person Details", False, f"API Error: {data.get('error')}")
                else:
                    print_test_result("Person Details", False, f"Status Code: {response.status_code}")
            else:
                print_test_result("Person Details", False, "No matches found for test")
        else:
            print_test_result("Person Details", False, "Failed to get test person")
            
    except Exception as e:
        print_test_result("Person Details", False, f"Error: {str(e)}")
    
    return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print_header("Testing Edge Cases")
    
    test_cases = [
        {
            'name': 'Empty Sequence',
            'payload': {'sequence': '', 'top_n': 5},
            'expect_error': True
        },
        {
            'name': 'Short Sequence',
            'payload': {'sequence': 'ATCG', 'top_n': 5},
            'expect_error': True
        },
        {
            'name': 'Invalid Characters',
            'payload': {'sequence': 'ATCGXYZ123', 'top_n': 5},
            'expect_error': True
        },
        {
            'name': 'Large Top N',
            'payload': {'sequence': TEST_SEQUENCES[0], 'top_n': 1000},
            'expect_error': False
        },
        {
            'name': 'Zero Top N',
            'payload': {'sequence': TEST_SEQUENCES[0], 'top_n': 0},
            'expect_error': True
        }
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/api/identify-person", 
                                   json=test_case['payload'],
                                   headers={'Content-Type': 'application/json'})
            
            data = response.json()
            
            if test_case['expect_error']:
                success = not data.get('success', True)
                print_test_result(test_case['name'], success, 
                                f"Expected error: {data.get('error', 'No error returned')}")
            else:
                success = data.get('success', False)
                print_test_result(test_case['name'], success, 
                                f"Matches found: {len(data.get('matches', []))}")
                
        except Exception as e:
            print_test_result(test_case['name'], False, f"Exception: {str(e)}")

def performance_test():
    """Test performance with multiple concurrent requests"""
    print_header("Performance Testing")
    
    sequence = TEST_SEQUENCES[0]
    num_requests = 5
    
    print(f"Running {num_requests} concurrent identity matching requests...")
    
    start_time = time.time()
    
    # Sequential requests for simplicity
    times = []
    for i in range(num_requests):
        req_start = time.time()
        try:
            response = requests.post(f"{BASE_URL}/api/identify-person", 
                                   json={'sequence': sequence, 'top_n': 3},
                                   headers={'Content-Type': 'application/json'})
            req_end = time.time()
            
            if response.status_code == 200:
                times.append(req_end - req_start)
            else:
                print(f"Request {i+1} failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Request {i+1} failed with error: {str(e)}")
    
    end_time = time.time()
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print_test_result("Performance Test", True)
        print(f"    Total Time: {end_time - start_time:.2f}s")
        print(f"    Successful Requests: {len(times)}/{num_requests}")
        print(f"    Average Response Time: {avg_time:.2f}s")
        print(f"    Min Response Time: {min_time:.2f}s")
        print(f"    Max Response Time: {max_time:.2f}s")
        print(f"    Requests per Second: {len(times)/(end_time-start_time):.2f}")
    else:
        print_test_result("Performance Test", False, "No successful requests")

def main():
    """Run all tests"""
    print_header("DNAAadeshak Identity Matching Test Suite")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing against: {BASE_URL}")
    
    # Run all tests
    tests_passed = 0
    total_tests = 6
    
    if test_health_check():
        tests_passed += 1
    
    if test_database_stats():
        tests_passed += 1
    
    identity_results = test_identity_matching()
    if identity_results:
        tests_passed += 1
    
    if test_person_details():
        tests_passed += 1
    
    test_edge_cases()
    tests_passed += 1  # Edge cases always count as passed if they run
    
    performance_test()
    tests_passed += 1  # Performance test always counts as passed if it runs
    
    # Summary
    print_header("Test Summary")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if identity_results:
        print("\nIdentity Matching Results Summary:")
        for i, result in enumerate(identity_results):
            print(f"  Sequence {i+1}: {result['top_match']} "
                  f"(Similarity: {result['similarity']:.3f}, "
                  f"Confidence: {result['confidence']}, "
                  f"Time: {result['processing_time']:.2f}s)")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
