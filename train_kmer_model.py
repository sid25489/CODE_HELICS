#!/usr/bin/env python3
"""
K-mer Based ML Model Training for DNAAadeshak
Uses k-mer frequency vectors for better classification accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from itertools import product
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def generate_kmers(k):
    """Generate all possible k-mers"""
    return [''.join(p) for p in product('ATCG', repeat=k)]

def extract_kmer_features(sequence, k=4):
    """Extract k-mer frequency features"""
    sequence = sequence.upper()
    if len(sequence) < k:
        return None
    
    # Generate all possible k-mers
    all_kmers = generate_kmers(k)
    kmer_counts = Counter()
    
    # Count k-mers in sequence
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(base in 'ATCG' for base in kmer):
            kmer_counts[kmer] += 1
    
    # Create feature vector
    total_kmers = len(sequence) - k + 1
    features = []
    for kmer in all_kmers:
        frequency = kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
        features.append(frequency)
    
    return np.array(features)

def load_sample_data(sample_size=20000):
    """Load and sample data efficiently"""
    print("📊 Loading synthetic dataset...")
    
    df = pd.read_csv('synthetic_dna_dataset.csv')
    print(f"✅ Dataset loaded: {len(df):,} records")
    
    # Filter by sequence length for consistency
    df['seq_length'] = df['sequence'].str.len()
    df_filtered = df[(df['seq_length'] >= 150) & (df['seq_length'] <= 500)]
    print(f"📏 Filtered by length (150-500 bp): {len(df_filtered):,} records")
    
    # Sample data for faster training
    if len(df_filtered) > sample_size:
        df_sample = df_filtered.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_filtered
    
    print(f"📦 Sample size: {len(df_sample):,} records")
    
    # Check class distribution
    class_counts = df_sample['class'].value_counts()
    print(f"📈 Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df_sample)) * 100
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    return df_sample

def extract_features_parallel(sequences, k=4):
    """Extract k-mer features for all sequences"""
    print(f"🔬 Extracting {k}-mer features...")
    
    features_list = []
    valid_indices = []
    
    for i, sequence in enumerate(sequences):
        features = extract_kmer_features(sequence, k)
        if features is not None:
            features_list.append(features)
            valid_indices.append(i)
        
        if (i + 1) % 2000 == 0:
            print(f"   Processed {i + 1:,} sequences...")
    
    print(f"✅ Feature extraction completed: {len(features_list):,} valid samples")
    print(f"   Feature dimensions: {len(features_list[0]) if features_list else 0}")
    
    return np.array(features_list), valid_indices

def train_kmer_model(X, y):
    """Train Random Forest model with k-mer features"""
    print("🤖 Training k-mer based Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Train Random Forest with optimized parameters for k-mer data
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Model training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, X_test, y_test, y_pred, accuracy

def evaluate_model_performance(model, X_test, y_test, y_pred):
    """Detailed model evaluation"""
    print_header("K-mer Model Performance Evaluation")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📊 Detailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    for class_name in np.unique(y_test):
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1-Score: {metrics['f1-score']:.4f}")
            print(f"      Support: {int(metrics['support'])}")
    
    # Feature importance (top k-mers)
    print("\n🔍 Top 15 Most Important K-mers:")
    all_kmers = generate_kmers(4)  # Assuming k=4
    importance_df = pd.DataFrame({
        'kmer': all_kmers[:len(model.feature_importances_)],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        print(f"   {i+1:2d}. {row['kmer']:4} : {row['importance']:.6f}")
    
    return accuracy

def test_kmer_model(model, k=4):
    """Test model with sample sequences"""
    print_header("K-mer Model Testing")
    
    test_sequences = [
        "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA",
        "CCGAGGTATGCGGCCAGAGTTGGGCGAATGGCATACTCCTCTGAACACATTAGGTGGGCGGTACTTATCCTGAACACATATCATCTCTGCTAGGGCGGCTGAATTGTCTGGATGGTATTTTGGCCAGGCTCCGGGGAGGTCAGCTACCCATGCCGAAACCGTACCTATGAGCTCGCATCATCGACTGTGGAACGACCCGCACTTACTATATCAGTGGAGTTTTGACGCTTATCTGCATCAAATCGACGCAGCCGGTAGTCGATAAAATTGTCGATTGTTGTAACTAGGCCACCGCTCAGATATGTACCCTAGACCAGCTGGCCGCTCTATTACTTGAACCGGTTTAGGAAAGCTGTAAATATTCCAA",
        "CCGGAGCCGAATACGCCTGCGGGCTGTCGTTTCAAGACCGAGATGAATTGAATGGGGGCCCTCTACTGTACGGGCAGTCTCTTGCCAACCATGATCTCAGACGCATGAAAGACTCGTA"
    ]
    
    for i, sequence in enumerate(test_sequences):
        features = extract_kmer_features(sequence, k)
        if features is not None:
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            confidence = np.max(probabilities)
            
            print(f"🧪 Test Sequence {i+1}:")
            print(f"   Length: {len(sequence)} bp")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Class probabilities:")
            for class_name, prob in zip(model.classes_, probabilities):
                print(f"      {class_name}: {prob:.4f}")
        else:
            print(f"❌ Test Sequence {i+1}: Too short for k-mer analysis")

def main():
    """Main k-mer training pipeline"""
    print_header("DNAAadeshak K-mer Based ML Model Training")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load sample data
    df_sample = load_sample_data(sample_size=20000)
    
    # Extract k-mer features
    k = 4  # 4-mer features (256 features)
    X, valid_indices = extract_features_parallel(df_sample['sequence'].values, k=k)
    
    if len(X) == 0:
        print("❌ No valid features extracted")
        return
    
    # Get corresponding labels
    y = df_sample.iloc[valid_indices]['class'].values
    
    print(f"📊 Final dataset:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]} ({k}-mers)")
    print(f"   Classes: {np.unique(y)}")
    
    # Train k-mer model
    model, X_test, y_test, y_pred, accuracy = train_kmer_model(X, y)
    
    # Evaluate performance
    final_accuracy = evaluate_model_performance(model, X_test, y_test, y_pred)
    
    # Save k-mer model
    print_header("Saving K-mer Model")
    
    try:
        joblib.dump(model, 'dna_model_kmer.pkl')
        print("✅ K-mer model saved as 'dna_model_kmer.pkl'")
        
        # Save as default model if accuracy is good
        if final_accuracy > 0.70:
            joblib.dump(model, 'dna_model.pkl')
            print("✅ Model saved as 'dna_model.pkl' for app usage")
        
        # Save metadata
        metadata = {
            'accuracy': final_accuracy,
            'training_samples': len(X),
            'features': X.shape[1],
            'k_value': k,
            'classes': list(np.unique(y)),
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForest_Kmer'
        }
        
        import json
        with open('model_metadata_kmer.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ K-mer model metadata saved")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Test with samples
    test_kmer_model(model, k=k)
    
    # Final summary
    print_header("K-mer Training Summary")
    
    if final_accuracy >= 0.90:
        performance_level = "Excellent"
        emoji = "🏆"
    elif final_accuracy >= 0.80:
        performance_level = "Very Good"
        emoji = "🥇"
    elif final_accuracy >= 0.70:
        performance_level = "Good"
        emoji = "🥈"
    elif final_accuracy >= 0.60:
        performance_level = "Fair"
        emoji = "🥉"
    else:
        performance_level = "Needs Improvement"
        emoji = "⚠️"
    
    print(f"{emoji} Model Performance: {performance_level}")
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"📊 Training Samples: {len(X):,}")
    print(f"🧬 K-mer Features: {X.shape[1]} ({k}-mers)")
    print(f"🏷️  Classes: {len(np.unique(y))}")
    print(f"⚡ Optimized for sequences (150-500 bp)")
    print(f"🔬 K-mer based feature extraction")
    print(f"💾 K-mer model ready for production")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, final_accuracy

if __name__ == "__main__":
    model, accuracy = main()
    print(f"\n🎉 K-mer model training completed with {accuracy:.2%} accuracy!")
