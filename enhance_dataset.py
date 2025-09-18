#!/usr/bin/env python3
"""Enhance synthetic DNA dataset with biologically realistic patterns"""

import pandas as pd
import numpy as np
import random
from collections import Counter
import shutil
import os

def create_gene_pattern(original_seq, enhancement_ratio=0.7):
    """Create gene-like patterns with start/stop codons and realistic features"""
    seq = original_seq.upper()
    length = len(seq)
    
    if random.random() < enhancement_ratio:
        # Create enhanced gene sequence
        enhanced = ""
        
        # Add start codon ATG at beginning (80% chance)
        if random.random() < 0.8:
            enhanced += "ATG"
        
        # Add coding sequence with realistic codon usage
        codons = ['GCT', 'GCC', 'GCA', 'GCG',  # Alanine
                 'TGT', 'TGC',                   # Cysteine
                 'GAT', 'GAC',                   # Aspartic acid
                 'GAA', 'GAG',                   # Glutamic acid
                 'TTT', 'TTC',                   # Phenylalanine
                 'GGT', 'GGC', 'GGA', 'GGG',    # Glycine
                 'CAT', 'CAC',                   # Histidine
                 'ATT', 'ATC', 'ATA',           # Isoleucine
                 'AAA', 'AAG',                   # Lysine
                 'CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG',  # Leucine
                 'ATG',                          # Methionine
                 'AAT', 'AAC',                   # Asparagine
                 'CCT', 'CCC', 'CCA', 'CCG',    # Proline
                 'CAA', 'CAG',                   # Glutamine
                 'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG',  # Arginine
                 'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC',  # Serine
                 'ACT', 'ACC', 'ACA', 'ACG',    # Threonine
                 'GTT', 'GTC', 'GTA', 'GTG',    # Valine
                 'TGG',                          # Tryptophan
                 'TAT', 'TAC']                   # Tyrosine
        
        # Fill middle with realistic codons
        remaining_length = length - len(enhanced) - 3  # Reserve space for stop codon
        while len(enhanced) < length - 3:
            if remaining_length >= 3:
                enhanced += random.choice(codons)
                remaining_length -= 3
            else:
                break
        
        # Add stop codon at end (90% chance)
        if random.random() < 0.9:
            stop_codons = ['TAA', 'TAG', 'TGA']
            enhanced += random.choice(stop_codons)
        
        # Pad or trim to original length
        if len(enhanced) < length:
            enhanced += seq[len(enhanced):]
        elif len(enhanced) > length:
            enhanced = enhanced[:length]
            
        # Mix with original sequence (30% original, 70% enhanced)
        final_seq = ""
        for i in range(length):
            if random.random() < 0.3:
                final_seq += seq[i] if i < len(seq) else enhanced[i]
            else:
                final_seq += enhanced[i] if i < len(enhanced) else seq[i]
                
        return final_seq
    else:
        return seq

def create_promoter_pattern(original_seq, enhancement_ratio=0.7):
    """Create promoter-like patterns with TATA boxes and high GC content"""
    seq = original_seq.upper()
    length = len(seq)
    
    if random.random() < enhancement_ratio:
        enhanced = ""
        
        # Add TATA box (70% chance)
        if random.random() < 0.7:
            tata_variants = ['TATAAA', 'TATAWAW', 'TATATAT', 'TATATA']
            enhanced += random.choice(tata_variants)
        
        # Add CpG islands and transcription factor binding sites
        tf_sites = ['CAAT', 'CCAAT', 'GGGCGG', 'CCGCCC', 'GCGCGC', 
                   'CCCGCC', 'GGGGCG', 'CGCGCG', 'CCCCGC', 'GCCCCC']
        
        # Fill with high GC content sequences
        gc_rich_patterns = ['GC', 'CG', 'GGG', 'CCC', 'GCGC', 'CGCG']
        
        while len(enhanced) < length:
            if random.random() < 0.3:  # 30% chance for TF binding site
                site = random.choice(tf_sites)
                if len(enhanced) + len(site) <= length:
                    enhanced += site
                else:
                    break
            else:  # Fill with GC-rich content
                pattern = random.choice(gc_rich_patterns)
                if len(enhanced) + len(pattern) <= length:
                    enhanced += pattern
                else:
                    break
        
        # Pad to original length
        while len(enhanced) < length:
            enhanced += random.choice(['G', 'C'])
            
        # Trim if too long
        if len(enhanced) > length:
            enhanced = enhanced[:length]
            
        # Mix with original (30% original, 70% enhanced)
        final_seq = ""
        for i in range(length):
            if random.random() < 0.3:
                final_seq += seq[i] if i < len(seq) else enhanced[i]
            else:
                final_seq += enhanced[i] if i < len(enhanced) else seq[i]
                
        return final_seq
    else:
        return seq

def create_junk_pattern(original_seq, enhancement_ratio=0.7):
    """Create junk DNA patterns with repetitive elements and low complexity"""
    seq = original_seq.upper()
    length = len(seq)
    
    if random.random() < enhancement_ratio:
        enhanced = ""
        
        # Repetitive elements
        repeats = ['AT', 'TA', 'ATAT', 'TATA', 'ATATAT', 'TATATA',
                  'AAAA', 'TTTT', 'AAAAT', 'TTTTA', 'AAAAAA', 'TTTTTT',
                  'GCGC', 'CGCG', 'GCGCGC', 'CGCGCG']
        
        # Low complexity patterns
        low_complexity = ['A'*6, 'T'*6, 'G'*4, 'C'*4, 'AT'*3, 'TA'*3]
        
        while len(enhanced) < length:
            if random.random() < 0.6:  # 60% repetitive elements
                repeat = random.choice(repeats)
                if len(enhanced) + len(repeat) <= length:
                    enhanced += repeat
                else:
                    break
            else:  # 40% low complexity
                pattern = random.choice(low_complexity)
                if len(enhanced) + len(pattern) <= length:
                    enhanced += pattern
                else:
                    break
        
        # Pad with AT-rich content (junk DNA is typically AT-rich)
        at_bases = ['A', 'T']
        while len(enhanced) < length:
            enhanced += random.choice(at_bases)
            
        # Trim if too long
        if len(enhanced) > length:
            enhanced = enhanced[:length]
            
        # Mix with original (30% original, 70% enhanced)
        final_seq = ""
        for i in range(length):
            if random.random() < 0.3:
                final_seq += seq[i] if i < len(seq) else enhanced[i]
            else:
                final_seq += enhanced[i] if i < len(enhanced) else seq[i]
                
        return final_seq
    else:
        return seq

def enhance_dataset():
    """Enhance the synthetic DNA dataset with biologically realistic patterns"""
    print("🧬 Enhancing Synthetic DNA Dataset")
    print("=" * 50)
    
    # Backup original dataset
    if os.path.exists('synthetic_dna_dataset.csv'):
        if not os.path.exists('synthetic_dna_dataset_original_backup.csv'):
            shutil.copy2('synthetic_dna_dataset.csv', 'synthetic_dna_dataset_original_backup.csv')
            print("✅ Original dataset backed up")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('synthetic_dna_dataset.csv')
    print(f"Loaded {len(df)} sequences")
    
    # Show class distribution
    class_counts = df['class'].value_counts()
    print(f"\nOriginal class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Enhance sequences based on class
    print("\nEnhancing sequences...")
    enhanced_sequences = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(df)} sequences...")
            
        original_seq = row['sequence']
        class_label = row['class']
        
        if class_label == 'gene':
            enhanced_seq = create_gene_pattern(original_seq)
        elif class_label == 'promoter':
            enhanced_seq = create_promoter_pattern(original_seq)
        elif class_label == 'junk':
            enhanced_seq = create_junk_pattern(original_seq)
        else:
            enhanced_seq = original_seq  # Keep unknown classes unchanged
            
        enhanced_sequences.append(enhanced_seq)
    
    # Update dataset
    df['sequence'] = enhanced_sequences
    
    # Save enhanced dataset
    df.to_csv('synthetic_dna_dataset.csv', index=False)
    print(f"\n✅ Enhanced dataset saved with {len(df)} sequences")
    
    # Analyze enhanced sequences
    print("\nAnalyzing enhanced sequences...")
    
    # Sample analysis for each class
    for class_label in ['gene', 'promoter', 'junk']:
        class_seqs = df[df['class'] == class_label]['sequence'].head(3)
        print(f"\n{class_label.upper()} samples:")
        for i, seq in enumerate(class_seqs):
            print(f"  {i+1}. {seq[:60]}...")
            
            # Calculate GC content
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
            print(f"     GC content: {gc_content:.1f}%")
            
            # Check for class-specific patterns
            if class_label == 'gene':
                has_start = 'ATG' in seq
                has_stop = any(stop in seq for stop in ['TAA', 'TAG', 'TGA'])
                print(f"     Start codon: {has_start}, Stop codon: {has_stop}")
            elif class_label == 'promoter':
                has_tata = 'TATA' in seq
                has_cg = seq.count('CG') > 5
                print(f"     TATA box: {has_tata}, CG-rich: {has_cg}")
            elif class_label == 'junk':
                repetitive = max(seq.count('ATAT'), seq.count('TATA'), seq.count('AAAA'))
                print(f"     Max repetitive count: {repetitive}")
    
    print(f"\n🎉 Dataset enhancement complete!")
    print(f"📁 Original dataset backed up as 'synthetic_dna_dataset_original_backup.csv'")
    print(f"📁 Enhanced dataset saved as 'synthetic_dna_dataset.csv'")

if __name__ == "__main__":
    enhance_dataset()
