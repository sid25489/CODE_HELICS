#!/usr/bin/env python3
"""Quick test for the DNA model"""

from ml_dna_analyzer import MLDNAAnalyzer

def test_model():
    analyzer = MLDNAAnalyzer()
    analyzer.load_model()
    
    # Test sequences
    sequences = [
        ("ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG", "gene"),
        ("TATAWACTTGGCGCCCGGGACTTGGTCCCGGGTCCCCGGGGAGTGC", "promoter"),
        ("ATATCTTTATGTCTTTCCCCCGCCATATATTTATGTATAAATCTCT", "junk")
    ]
    
    print("🧬 DNA Model Quick Test")
    print("=" * 40)
    
    for seq, expected in sequences:
        result = analyzer.predict(seq)
        print(f"\nSequence: {seq[:30]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {result['class']}")
        print(f"Confidence: {result['probability']*100:.1f}%")
        
        correct = "✅" if result['class'] == expected else "❌"
        print(f"Result: {correct}")

if __name__ == "__main__":
    test_model()
