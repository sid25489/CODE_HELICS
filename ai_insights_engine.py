import numpy as np
import re
from collections import Counter
from datetime import datetime
import json
import random

class AIInsightsEngine:
    """
    Advanced AI-powered DNA sequence analysis engine that provides
    intelligent insights, quality assessment, and biological predictions.
    """
    
    def __init__(self):
        self.codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        # Known regulatory sequences
        self.regulatory_patterns = {
            'TATA_box': r'TATAAA|TATAWAW',
            'CAAT_box': r'CCAAT',
            'GC_box': r'GGGCGG|CCGCCC',
            'initiator': r'YYANWYY',
            'kozak': r'GCCRCCATGG',
            'shine_dalgarno': r'AGGAGG'
        }
        
        # Repetitive elements
        self.repeat_patterns = {
            'simple_repeat': r'([ATCG])\1{4,}',
            'dinucleotide': r'([ATCG]{2})\1{3,}',
            'trinucleotide': r'([ATCG]{3})\1{2,}',
            'inverted_repeat': r'([ATCG]+)N{1,20}([ATCG]+)'
        }
    
    def analyze_sequence_quality(self, sequence):
        """Assess the quality and characteristics of a DNA sequence"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        length = len(sequence)
        
        if length == 0:
            return {'quality_score': 0, 'issues': ['Empty sequence']}
        
        # Calculate composition metrics
        composition = {base: sequence.count(base) for base in 'ATCG'}
        gc_content = (composition['G'] + composition['C']) / length
        
        # Quality assessment
        quality_score = 100
        issues = []
        recommendations = []
        
        # Check GC content (optimal range: 40-60%)
        if gc_content < 0.3:
            quality_score -= 15
            issues.append(f'Low GC content ({gc_content:.1%})')
            recommendations.append('Consider sequence context - may indicate AT-rich regions')
        elif gc_content > 0.7:
            quality_score -= 10
            issues.append(f'High GC content ({gc_content:.1%})')
            recommendations.append('High GC regions may affect PCR amplification')
        
        # Check for homopolymer runs
        max_homopolymer = self._find_max_homopolymer(sequence)
        if max_homopolymer > 6:
            quality_score -= 20
            issues.append(f'Long homopolymer run detected ({max_homopolymer} bases)')
            recommendations.append('Long homopolymers may cause sequencing errors')
        
        # Check sequence complexity
        complexity = self._calculate_complexity(sequence)
        if complexity < 0.5:
            quality_score -= 15
            issues.append('Low sequence complexity detected')
            recommendations.append('Low complexity regions may indicate repetitive elements')
        
        # Check for ambiguous bases
        ambiguous_count = len([b for b in sequence if b not in 'ATCG'])
        if ambiguous_count > 0:
            quality_score -= ambiguous_count * 2
            issues.append(f'{ambiguous_count} ambiguous bases found')
        
        return {
            'quality_score': max(0, quality_score),
            'gc_content': gc_content,
            'length': length,
            'composition': composition,
            'complexity': complexity,
            'max_homopolymer': max_homopolymer,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def detect_mutations(self, sequence, reference_patterns=None):
        """Detect potential mutations and variants in the sequence"""
        mutations = []
        
        # Detect transition/transversion patterns
        transitions = ['AG', 'GA', 'CT', 'TC']  # Purines <-> Purines, Pyrimidines <-> Pyrimidines
        transversions = ['AC', 'CA', 'AT', 'TA', 'GT', 'TG', 'GC', 'CG']
        
        # Analyze dinucleotide patterns for mutation signatures
        dinucleotides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dinuc_counts = Counter(dinucleotides)
        
        # CpG dinucleotides (methylation sites)
        cpg_count = dinuc_counts.get('CG', 0)
        cpg_frequency = cpg_count / max(1, len(dinucleotides))
        
        mutation_analysis = {
            'cpg_sites': cpg_count,
            'cpg_frequency': cpg_frequency,
            'potential_hotspots': []
        }
        
        # Detect CpG islands (regions with high CpG content)
        if cpg_frequency > 0.02:  # Typical threshold for CpG islands
            mutation_analysis['potential_hotspots'].append({
                'type': 'CpG_island',
                'description': 'High CpG content region - potential methylation site',
                'frequency': cpg_frequency
            })
        
        # Detect repetitive regions that are mutation-prone
        for pattern_name, pattern in self.repeat_patterns.items():
            matches = list(re.finditer(pattern, sequence))
            if matches:
                mutation_analysis['potential_hotspots'].append({
                    'type': pattern_name,
                    'description': f'Repetitive region prone to slippage mutations',
                    'count': len(matches),
                    'positions': [(m.start(), m.end()) for m in matches[:5]]  # Limit to first 5
                })
        
        return mutation_analysis
    
    def predict_biological_function(self, sequence):
        """Predict potential biological functions based on sequence features"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        predictions = []
        confidence_scores = {}
        
        # Check for Open Reading Frames (ORFs)
        orfs = self._find_orfs(sequence)
        if orfs:
            predictions.append('Protein-coding potential detected')
            confidence_scores['protein_coding'] = min(0.9, len(orfs) * 0.3)
        
        # Check for regulatory elements
        regulatory_found = []
        for element, pattern in self.regulatory_patterns.items():
            if re.search(pattern, sequence, re.IGNORECASE):
                regulatory_found.append(element)
        
        if regulatory_found:
            predictions.append(f'Regulatory elements detected: {", ".join(regulatory_found)}')
            confidence_scores['regulatory'] = min(0.8, len(regulatory_found) * 0.2)
        
        # Analyze codon usage if ORFs found
        codon_bias = None
        if orfs:
            codon_bias = self._analyze_codon_usage(sequence)
            if codon_bias['bias_score'] > 0.6:
                predictions.append('Strong codon usage bias - likely highly expressed gene')
                confidence_scores['high_expression'] = codon_bias['bias_score']
        
        # Check for structural features
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if gc_content > 0.6:
            predictions.append('High GC content - potential structural stability')
            confidence_scores['structural'] = gc_content
        
        # Predict based on length and composition
        length = len(sequence)
        if length > 1000:
            if orfs:
                predictions.append('Large sequence with ORFs - likely gene or gene cluster')
            else:
                predictions.append('Large non-coding sequence - potential regulatory region')
        elif length < 100:
            predictions.append('Short sequence - potential regulatory motif or primer')
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'orfs_found': len(orfs) if orfs else 0,
            'regulatory_elements': regulatory_found,
            'codon_bias': codon_bias
        }
    
    def calculate_similarity_score(self, sequence1, sequence2):
        """Calculate similarity between two DNA sequences"""
        if len(sequence1) != len(sequence2):
            # Align sequences using simple sliding window
            shorter, longer = (sequence1, sequence2) if len(sequence1) < len(sequence2) else (sequence2, sequence1)
            best_score = 0
            
            for i in range(len(longer) - len(shorter) + 1):
                window = longer[i:i+len(shorter)]
                matches = sum(1 for a, b in zip(shorter, window) if a == b)
                score = matches / len(shorter)
                best_score = max(best_score, score)
            
            return best_score
        else:
            matches = sum(1 for a, b in zip(sequence1, sequence2) if a == b)
            return matches / len(sequence1)
    
    def generate_insights(self, sequence):
        """Generate insights for a DNA sequence (wrapper method)"""
        # Create mock analysis results for compatibility
        analysis_results = {
            'class': 'gene',
            'probability': 0.8,
            'confidence': 'High'
        }
        return self.generate_ai_insights(sequence, analysis_results)
    
    def generate_ai_insights(self, sequence, analysis_results):
        """Generate intelligent insights and recommendations"""
        insights = []
        
        # Quality-based insights
        quality = analysis_results.get('quality', {})
        quality_score = quality.get('quality_score', 0)
        
        if quality_score >= 90:
            insights.append({
                'type': 'quality',
                'level': 'success',
                'message': 'Excellent sequence quality detected',
                'details': 'This sequence shows optimal characteristics for most applications'
            })
        elif quality_score >= 70:
            insights.append({
                'type': 'quality',
                'level': 'warning',
                'message': 'Good sequence quality with minor concerns',
                'details': f"Quality score: {quality_score}/100. {', '.join(quality.get('recommendations', []))}"
            })
        else:
            insights.append({
                'type': 'quality',
                'level': 'danger',
                'message': 'Sequence quality issues detected',
                'details': f"Multiple quality concerns found. {', '.join(quality.get('issues', []))}"
            })
        
        # Function-based insights
        function = analysis_results.get('function', {})
        predictions = function.get('predictions', [])
        
        if predictions:
            insights.append({
                'type': 'function',
                'level': 'info',
                'message': f'Predicted functions: {len(predictions)} identified',
                'details': '; '.join(predictions[:3])  # Show top 3 predictions
            })
        
        # Mutation analysis insights
        mutations = analysis_results.get('mutations', {})
        hotspots = mutations.get('potential_hotspots', [])
        
        if hotspots:
            insights.append({
                'type': 'mutation',
                'level': 'warning',
                'message': f'{len(hotspots)} mutation-prone regions detected',
                'details': 'These regions may be subject to higher mutation rates'
            })
        
        # Length-based insights
        length = len(sequence.replace(' ', '').replace('\n', ''))
        if length > 10000:
            insights.append({
                'type': 'structure',
                'level': 'info',
                'message': 'Large genomic sequence detected',
                'details': 'Consider analyzing in smaller segments for detailed analysis'
            })
        elif length < 50:
            insights.append({
                'type': 'structure',
                'level': 'warning',
                'message': 'Very short sequence',
                'details': 'Limited analysis possible due to sequence length'
            })
        
        return insights
    
    def comprehensive_analysis(self, sequence):
        """Perform comprehensive AI-powered analysis"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Run all analysis modules
        quality_analysis = self.analyze_sequence_quality(sequence)
        mutation_analysis = self.detect_mutations(sequence)
        function_analysis = self.predict_biological_function(sequence)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'sequence_length': len(sequence),
            'quality': quality_analysis,
            'mutations': mutation_analysis,
            'function': function_analysis
        }
        
        # Generate AI insights
        ai_insights = self.generate_ai_insights(sequence, results)
        results['ai_insights'] = ai_insights
        
        # Calculate overall confidence score
        confidence_scores = function_analysis.get('confidence_scores', {})
        overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
        results['overall_confidence'] = overall_confidence
        
        return results
    
    # Helper methods
    def _find_max_homopolymer(self, sequence):
        """Find the maximum length of homopolymer runs"""
        if not sequence:
            return 0
        
        max_length = 1
        current_length = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        
        return max_length
    
    def _calculate_complexity(self, sequence):
        """Calculate sequence complexity using Shannon entropy"""
        if not sequence:
            return 0
        
        # Count nucleotide frequencies
        counts = Counter(sequence)
        length = len(sequence)
        
        # Calculate Shannon entropy
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)
        
        # Normalize to 0-1 scale (max entropy for DNA is log2(4) = 2)
        return entropy / 2.0
    
    def _find_orfs(self, sequence):
        """Find Open Reading Frames in the sequence"""
        orfs = []
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        for frame in range(3):
            i = frame
            while i < len(sequence) - 2:
                codon = sequence[i:i+3]
                if codon in start_codons:
                    # Look for stop codon
                    j = i + 3
                    while j < len(sequence) - 2:
                        stop_codon = sequence[j:j+3]
                        if stop_codon in stop_codons:
                            orf_length = j - i + 3
                            if orf_length >= 150:  # Minimum ORF length (50 amino acids)
                                orfs.append({
                                    'start': i,
                                    'end': j + 3,
                                    'length': orf_length,
                                    'frame': frame
                                })
                            break
                        j += 3
                i += 3
        
        return orfs
    
    def _analyze_codon_usage(self, sequence):
        """Analyze codon usage bias"""
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        codon_counts = Counter(codons)
        
        # Calculate codon usage bias (simplified)
        total_codons = len(codons)
        if total_codons == 0:
            return {'bias_score': 0, 'most_frequent': None}
        
        # Find most frequent codon
        most_frequent = codon_counts.most_common(1)[0] if codon_counts else ('', 0)
        bias_score = most_frequent[1] / total_codons if total_codons > 0 else 0
        
        return {
            'bias_score': bias_score,
            'most_frequent': most_frequent[0],
            'total_codons': total_codons,
            'unique_codons': len(codon_counts)
        }

# Example usage and testing
if __name__ == "__main__":
    engine = AIInsightsEngine()
    
    # Test with sample sequence
    test_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA"
    
    print("Running comprehensive AI analysis...")
    results = engine.comprehensive_analysis(test_sequence)
    
    print(f"\nSequence Length: {results['sequence_length']}")
    print(f"Quality Score: {results['quality']['quality_score']}/100")
    print(f"Overall Confidence: {results['overall_confidence']:.2f}")
    
    print("\nAI Insights:")
    for insight in results['ai_insights']:
        print(f"- [{insight['level'].upper()}] {insight['message']}")
        print(f"  {insight['details']}")
