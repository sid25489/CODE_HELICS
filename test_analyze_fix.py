#!/usr/bin/env python3
"""Test the DNA analyze fix"""

import sys
import os
sys.path.append('.')

def test_enhanced_analyzer():
    """Test the EnhancedDNAAnalyzer"""
    print("🧬 Testing Enhanced DNA Analyzer Fix")
    print("=" * 40)
    
    try:
        from app_minimal import EnhancedDNAAnalyzer
        
        # Create analyzer
        analyzer = EnhancedDNAAnalyzer()
        
        # Test sequence
        test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
        
        # Test analysis
        results = analyzer.analyze_sequence(test_sequence)
        
        print(f"✅ Analysis successful!")
        print(f"Results type: {type(results)}")
        print(f"Number of results: {len(results)}")
        
        if isinstance(results, list) and len(results) > 0:
            print("\nResults:")
            for i, result in enumerate(results, 1):
                identity = result.get('identity', 'Unknown')
                probability = result.get('probability', 0)
                confidence = result.get('confidence', 'Unknown')
                print(f"  {i}. {identity}: {probability:.2%} ({confidence})")
            
            # Check if format is correct
            required_fields = ['identity', 'probability', 'confidence']
            first_result = results[0]
            
            missing_fields = [field for field in required_fields if field not in first_result]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                return False
            else:
                print(f"✅ All required fields present")
                return True
        else:
            print(f"❌ Results format incorrect: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_analyzer()
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
    sys.exit(0 if success else 1)
