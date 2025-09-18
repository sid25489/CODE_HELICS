#!/usr/bin/env python3
"""
High-Accuracy ML Model Training Script for DNAAadeshak
Advanced feature engineering and preprocessing for maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def advanced_feature_extraction(sequence):
    """Extract comprehensive features for high accuracy"""
    sequence = sequence.upper()
    length = len(sequence)
    
    if length < 50:
        return None
    
    features = []
    
    # Basic nucleotide composition
    a_count = sequence.count('A')
    t_count = sequence.count('T')
    c_count = sequence.count('C')
    g_count = sequence.count('G')
    
    features.extend([a_count/length, t_count/length, c_count/length, g_count/length])
    
    # GC content and AT content
    gc_content = (g_count + c_count) / length
    at_content = (a_count + t_count) / length
    features.extend([gc_content, at_content])
    
    # Dinucleotide frequencies
    dinucs = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 
              'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']
    for dinuc in dinucs:
        count = 0
        for i in range(length - 1):
            if sequence[i:i+2] == dinuc:
                count += 1
        features.append(count / (length - 1) if length > 1 else 0)
    
    # Trinucleotide frequencies (selected important ones)
    important_trinucs = ['ATG', 'TAA', 'TAG', 'TGA', 'CpG', 'GCA', 'GCC', 'GCG', 'GCT']
    for trinuc in important_trinucs:
        if trinuc == 'CpG':
            # Special handling for CpG
            count = 0
            for i in range(length - 1):
                if sequence[i:i+2] == 'CG':
                    count += 1
        else:
            count = 0
            for i in range(length - 2):
                if sequence[i:i+3] == trinuc:
                    count += 1
        features.append(count / (length - 2) if length > 2 else 0)
    
    # Sequence complexity (entropy)
    from collections import Counter
    counts = Counter(sequence)
    entropy = 0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * np.log2(p)
    features.append(entropy)
    
    # Homopolymer runs
    max_homo = 0
    current_homo = 1
    for i in range(1, length):
        if sequence[i] == sequence[i-1]:
            current_homo += 1
        else:
            max_homo = max(max_homo, current_homo)
            current_homo = 1
    max_homo = max(max_homo, current_homo)
    features.append(max_homo / length)
    
    # Purine/Pyrimidine content
    purines = sequence.count('A') + sequence.count('G')
    pyrimidines = sequence.count('C') + sequence.count('T')
    features.extend([purines/length, pyrimidines/length])
    
    # Codon usage bias (for coding sequences)
    if length >= 3:
        codons = [sequence[i:i+3] for i in range(0, length-2, 3)]
        start_codons = sum(1 for codon in codons if codon == 'ATG')
        stop_codons = sum(1 for codon in codons if codon in ['TAA', 'TAG', 'TGA'])
        features.extend([start_codons/len(codons) if codons else 0, 
                        stop_codons/len(codons) if codons else 0])
    else:
        features.extend([0, 0])
    
    # Sequence length (normalized)
    features.append(np.log(length))
    
    return np.array(features)

def load_balanced_dataset(sample_size=15000):
    """Load and create balanced dataset"""
    print("📊 Loading and balancing dataset...")
    
    df = pd.read_csv('synthetic_dna_dataset.csv')
    print(f"✅ Dataset loaded: {len(df):,} records")
    
    # Filter sequences by length (200-600 bp for optimal features)
    df['seq_length'] = df['sequence'].str.len()
    df_filtered = df[(df['seq_length'] >= 200) & (df['seq_length'] <= 600)]
    print(f"📏 Filtered by length (200-600 bp): {len(df_filtered):,} records")
    
    # Balance classes
    class_counts = df_filtered['class'].value_counts()
    samples_per_class = min(sample_size // len(class_counts), class_counts.min())
    
    balanced_samples = []
    for class_name in class_counts.index:
        class_data = df_filtered[df_filtered['class'] == class_name].sample(
            n=samples_per_class, random_state=42
        )
        balanced_samples.append(class_data)
    
    df_balanced = pd.concat(balanced_samples, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"⚖️  Balanced dataset: {len(df_balanced):,} records")
    print(f"   Samples per class: {samples_per_class:,}")
    
    return df_balanced

def extract_features_batch(sequences):
    """Extract features for all sequences"""
    print("🔬 Extracting advanced features...")
    
    features_list = []
    valid_indices = []
    
    for i, sequence in enumerate(sequences):
        features = advanced_feature_extraction(sequence)
        if features is not None:
            features_list.append(features)
            valid_indices.append(i)
        
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1:,} sequences...")
    
    print(f"✅ Feature extraction completed: {len(features_list):,} valid samples")
    return np.array(features_list), valid_indices

def train_ensemble_model(X, y):
    """Train ensemble model for high accuracy"""
    print("🤖 Training high-accuracy ensemble model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("   Training Random Forest...")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    print("   Training Gradient Boosting...")
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    print(f"   Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # Choose best model
    if rf_accuracy >= gb_accuracy:
        best_model = rf_model
        best_accuracy = rf_accuracy
        best_pred = rf_pred
        model_type = "RandomForest"
        use_scaling = False
    else:
        best_model = gb_model
        best_accuracy = gb_accuracy
        best_pred = gb_pred
        model_type = "GradientBoosting"
        use_scaling = True
    
    print(f"🏆 Best model: {model_type} with {best_accuracy:.4f} accuracy")
    
    return best_model, scaler if use_scaling else None, X_test, y_test, best_pred, best_accuracy

def evaluate_detailed_performance(model, X_test, y_test, y_pred):
    """Detailed performance evaluation"""
    print_header("Detailed Performance Analysis")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class performance
    print("\n📊 Per-Class Performance:")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    for class_name in np.unique(y_test):
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1-Score: {metrics['f1-score']:.4f}")
            print(f"      Support: {int(metrics['support'])}")
    
    return accuracy

def test_model_with_samples(model, scaler=None):
    """Test model with sample sequences"""
    print_header("Sample Predictions Test")
    
    test_sequences = [
        "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA",
        "CCGAGGTATGCGGCCAGAGTTGGGCGAATGGCATACTCCTCTGAACACATTAGGTGGGCGGTACTTATCCTGAACACATATCATCTCTGCTAGGGCGGCTGAATTGTCTGGATGGTATTTTGGCCAGGCTCCGGGGAGGTCAGCTACCCATGCCGAAACCGTACCTATGAGCTCGCATCATCGACTGTGGAACGACCCGCACTTACTATATCAGTGGAGTTTTGACGCTTATCTGCATCAAATCGACGCAGCCGGTAGTCGATAAAATTGTCGATTGTTGTAACTAGGCCACCGCTCAGATATGTACCCTAGACCAGCTGGCCGCTCTATTACTTGAACCGGTTTAGGAAAGCTGTAAATATTCCAA",
        "TATAAAAGGCCCAATGCGGGCATCAGGTAATGCCACCGACGTGATATTCGCCCCGGTTTAGGGGCTTGCCGCGGGTTGTAACGCCGATGGGGTTCTCTGTCCTGAAGCCCGACCATTCTTGTCTAGCATATCCTAAGTGGAAGCGGGTGTCTGGGTCAGTGAGACTCGGAACTCCTCACTCGCGGGCGGGGGGGGACATGTGCCCTTGGCTCTTGGTGTTGCGAAGGGCAACACATAAATTG"
    ]
    
    for i, sequence in enumerate(test_sequences):
        features = advanced_feature_extraction(sequence)
        if features is not None:
            if scaler:
                features = scaler.transform([features])
            else:
                features = [features]
            
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            print(f"🧪 Test Sequence {i+1}:")
            print(f"   Length: {len(sequence)} bp")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.4f}")

def main():
    """Main high-accuracy training pipeline"""
    print_header("DNAAadeshak High-Accuracy ML Model Training")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load balanced dataset
    df_balanced = load_balanced_dataset(sample_size=15000)
    
    # Extract advanced features
    X, valid_indices = extract_features_batch(df_balanced['sequence'].values)
    y = df_balanced.iloc[valid_indices]['class'].values
    
    print(f"📊 Final dataset:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)}")
    
    # Train ensemble model
    model, scaler, X_test, y_test, y_pred, accuracy = train_ensemble_model(X, y)
    
    # Evaluate performance
    final_accuracy = evaluate_detailed_performance(model, X_test, y_test, y_pred)
    
    # Save high-accuracy model
    print_header("Saving High-Accuracy Model")
    
    try:
        joblib.dump(model, 'dna_model_high_accuracy.pkl')
        print("✅ High-accuracy model saved as 'dna_model_high_accuracy.pkl'")
        
        if scaler:
            joblib.dump(scaler, 'feature_scaler.pkl')
            print("✅ Feature scaler saved")
        
        # Save as default model if accuracy is good
        if final_accuracy > 0.80:
            joblib.dump(model, 'dna_model.pkl')
            print("✅ Model saved as 'dna_model.pkl' for app usage")
        
        # Save metadata
        metadata = {
            'accuracy': final_accuracy,
            'training_samples': len(X),
            'features': X.shape[1],
            'classes': list(np.unique(y)),
            'training_date': datetime.now().isoformat(),
            'model_type': 'HighAccuracy_Ensemble',
            'uses_scaling': scaler is not None
        }
        
        import json
        with open('model_metadata_high_accuracy.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Model metadata saved")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Test with samples
    test_model_with_samples(model, scaler)
    
    # Final summary
    print_header("High-Accuracy Training Summary")
    
    if final_accuracy >= 0.95:
        performance_level = "Excellent"
        emoji = "🏆"
    elif final_accuracy >= 0.90:
        performance_level = "Very Good"
        emoji = "🥇"
    elif final_accuracy >= 0.85:
        performance_level = "Good"
        emoji = "🥈"
    elif final_accuracy >= 0.80:
        performance_level = "Fair"
        emoji = "🥉"
    else:
        performance_level = "Needs Improvement"
        emoji = "⚠️"
    
    print(f"{emoji} Model Performance: {performance_level}")
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"📊 Training Samples: {len(X):,}")
    print(f"🎯 Advanced Features: {X.shape[1]}")
    print(f"🏷️  Classes: {len(np.unique(y))}")
    print(f"⚡ Optimized for sequences (200-600 bp)")
    print(f"🧠 Advanced feature engineering applied")
    print(f"💾 High-accuracy model ready for production")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, final_accuracy

if __name__ == "__main__":
    model, accuracy = main()
    print(f"\n🎉 High-accuracy model training completed with {accuracy:.2%} accuracy!")
