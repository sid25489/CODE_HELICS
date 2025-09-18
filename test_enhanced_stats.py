#!/usr/bin/env python3
"""Test the enhanced database stats"""

from dna_identity_matcher import DNAIdentityMatcher
import json

def test_enhanced_stats():
    print("🧬 Testing Enhanced Database Stats")
    print("=" * 50)
    
    # Initialize matcher
    matcher = DNAIdentityMatcher()
    
    # Load database
    if matcher.load_database():
        print("✅ Database loaded successfully")
        
        # Get enhanced stats
        stats = matcher.get_database_stats()
        
        if stats:
            print("\n📊 Enhanced Database Statistics:")
            print("=" * 50)
            
            # Database info
            db_info = stats['database_info']
            print(f"📁 Database: {db_info['name']}")
            print(f"🔢 Version: {db_info['version']}")
            print(f"📅 Enhanced: {db_info['enhancement_date']}")
            print(f"📝 Description: {db_info['description']}")
            
            # Basic stats
            print(f"\n📈 Basic Statistics:")
            print(f"  Total Records: {stats['total_records']:,}")
            print(f"  Unique People: {stats['unique_people']:,}")
            print(f"  Total Sequences: {stats['total_sequences']:,}")
            print(f"  Avg Sequences/Person: {stats['avg_sequences_per_person']}")
            
            # Class distribution
            print(f"\n🧬 Class Distribution:")
            for cls, count in stats['classes_distribution'].items():
                print(f"  {cls}: {count:,}")
            
            # Sequence length stats
            length_stats = stats['sequence_length_stats']
            print(f"\n📏 Sequence Length Statistics:")
            print(f"  Min: {length_stats['min']} bp")
            print(f"  Max: {length_stats['max']} bp")
            print(f"  Mean: {length_stats['mean']} bp")
            print(f"  Median: {length_stats['median']} bp")
            
            # Enhancement features
            enhancement = stats['enhancement_features']
            
            print(f"\n🧬 Gene Sequence Enhancements:")
            gene_stats = enhancement['gene_sequences']
            print(f"  Total Gene Sequences: {gene_stats['total']:,}")
            print(f"  With Start Codon (ATG): {gene_stats['with_start_codon']:,} ({gene_stats['start_codon_percentage']}%)")
            print(f"  With Stop Codon: {gene_stats['with_stop_codon']:,} ({gene_stats['stop_codon_percentage']}%)")
            
            print(f"\n🧬 Promoter Sequence Enhancements:")
            promoter_stats = enhancement['promoter_sequences']
            print(f"  Total Promoter Sequences: {promoter_stats['total']:,}")
            print(f"  With TATA Box: {promoter_stats['with_tata_box']:,} ({promoter_stats['tata_box_percentage']}%)")
            print(f"  Average GC Content: {promoter_stats['avg_gc_content']}%")
            print(f"  High GC Sequences (>60%): {promoter_stats['high_gc_sequences']:,}")
            
            print(f"\n🧬 Junk Sequence Enhancements:")
            junk_stats = enhancement['junk_sequences']
            print(f"  Total Junk Sequences: {junk_stats['total']:,}")
            print(f"  With Repetitive Elements: {junk_stats['with_repetitive_elements']:,} ({junk_stats['repetitive_percentage']}%)")
            
            # ML compatibility
            ml_stats = stats['ml_model_compatibility']
            print(f"\n🤖 ML Model Compatibility:")
            print(f"  Accuracy Achieved: {ml_stats['accuracy_achieved']}")
            print(f"  Feature Extraction: {ml_stats['feature_extraction']}")
            print(f"  Training Ready: {ml_stats['training_ready']}")
            
            print(f"\n🎉 Enhanced database stats successfully generated!")
            
        else:
            print("❌ Failed to get database stats")
    else:
        print("❌ Failed to load database")

if __name__ == "__main__":
    test_enhanced_stats()
