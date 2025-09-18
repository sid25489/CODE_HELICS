#!/usr/bin/env python3
"""
Optimized ML Model Training Script for DNAAadeshak
Focuses on shorter data samples and high accuracy training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our ML DNA Analyzer
from ml_dna_analyzer import MLDNAAnalyzer

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def load_and_sample_data(sample_size=10000, min_length=100, max_length=1000):
    """Load and sample data for optimal training"""
    print(f"📊 Loading synthetic dataset...")
    
    try:
        df = pd.read_csv('synthetic_dna_dataset.csv')
        print(f"✅ Dataset loaded: {len(df):,} total records")
        
        # Filter by sequence length for shorter, more manageable sequences
        df['seq_length'] = df['sequence'].str.len()
        df_filtered = df[(df['seq_length'] >= min_length) & (df['seq_length'] <= max_length)]
        
        print(f"📏 Filtered by length ({min_length}-{max_length} bp): {len(df_filtered):,} records")
        
        # Balance classes and sample
        class_counts = df_filtered['class'].value_counts()
        print(f"📈 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count:,}")
        
        # Sample equally from each class for balanced training
        samples_per_class = min(sample_size // len(class_counts), class_counts.min())
        
        balanced_samples = []
        for class_name in class_counts.index:
            class_data = df_filtered[df_filtered['class'] == class_name].sample(
                n=samples_per_class, random_state=42
            )
            balanced_samples.append(class_data)
        
        df_sample = pd.concat(balanced_samples, ignore_index=True)
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"⚖️  Balanced sample created: {len(df_sample):,} records")
        print(f"   Samples per class: {samples_per_class:,}")
        
        return df_sample
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def extract_optimized_features(sequences):
    """Extract optimized features for high accuracy"""
    print("🔬 Extracting optimized features...")
    
    analyzer = MLDNAAnalyzer()
    features_list = []
    valid_indices = []
    
    for i, sequence in enumerate(sequences):
        try:
            features = analyzer.extract_features(sequence)
            if features is not None and len(features) > 0:
                features_list.append(features)
                valid_indices.append(i)
        except Exception:
            continue
        
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1:,} sequences...")
    
    print(f"✅ Feature extraction completed: {len(features_list):,} valid samples")
    return np.array(features_list), valid_indices

def train_optimized_model(X, y):
    """Train model with optimized hyperparameters"""
    print("🤖 Training optimized Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Optimized hyperparameters for high accuracy
    model = RandomForestClassifier(
        n_estimators=200,           # More trees for better accuracy
        max_depth=25,               # Deeper trees
        min_samples_split=2,        # Allow more splits
        min_samples_leaf=1,         # Allow smaller leaves
        max_features='sqrt',        # Optimal feature selection
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'     # Handle class imbalance
    )
    
    # Train the model
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
    print_header("Model Performance Evaluation")
    
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
    
    # Confusion Matrix
    print("\n📈 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)
    
    print("Predicted ->", end="")
    for class_name in classes:
        print(f"{class_name:>10}", end="")
    print()
    
    for i, actual_class in enumerate(classes):
        print(f"{actual_class:>10}   ", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_names = [
        'A_count', 'T_count', 'C_count', 'G_count',
        'AT_content', 'GC_content', 'sequence_length', 'entropy'
    ]
    
    # Add k-mer features
    kmers = [''.join(p) for p in __import__('itertools').product('ATCG', repeat=3)]
    feature_names.extend([f'kmer_{kmer}' for kmer in kmers])
    
    # Add remaining features
    while len(feature_names) < len(model.feature_importances_):
        feature_names.append(f'feature_{len(feature_names)}')
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(model.feature_importances_)],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:20} : {row['importance']:.6f}")
    
    return accuracy

def test_model_predictions(model, analyzer):
    """Test model with sample predictions"""
    print_header("Sample Predictions Test")
    
    # Test sequences
    test_sequences = [
        "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAA",
        "CCGAGGTATGCGGCCAGAGTTGGGCGAATGGCATACTCCTCTGAACACATTAGGTGGGCGGTACTTATCCTGAACACATATCATCTCTGCTAGGGCGGCTGAATTGTCTGGATGGTATTTTGGCCAGGCTCCGGGGAGGTCAGCTACCCATGCCGAAACCGTACCTATGAGCTCGCATCATCGACTGTGGAACGACCCGCACTTACTATATCAGTGGAGTTTTGACGCTTATCTGCATCAAATCGACGCAGCCGGTAGTCGATAAAATTGTCGATTGTTGTAACTAGGCCACCGCTCAGATATGTACCCTAGACCAGCTGGCCGCTCTATTACTTGAACCGGTTTAGGAAAGCTGTAAATATTCCAA",
        "TATAAAAGGCCCAATGCGGGCATCAGGTAATGCCACCGACGTGATATTCGCCCCGGTTTAGGGGCTTGCCGCGGGTTGTAACGCCGATGGGGTTCTCTGTCCTGAAGCCCGACCATTCTTGTCTAGCATATCCTAAGTGGAAGCGGGTGTCTGGGTCAGTGAGACTCGGAACTCCTCACTCGCGGGCGGGGGGGGACATGTGCCCTTGGCTCTTGGTGTTGCGAAGGGCAACACATAAATTG"
    ]
    
    for i, sequence in enumerate(test_sequences):
        try:
            features = analyzer.extract_features(sequence)
            if features is not None:
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = np.max(probabilities)
                
                print(f"🧪 Test Sequence {i+1}:")
                print(f"   Length: {len(sequence)} bp")
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {confidence:.4f}")
                print(f"   Probabilities: {dict(zip(model.classes_, probabilities))}")
            else:
                print(f"❌ Test Sequence {i+1}: Feature extraction failed")
        except Exception as e:
            print(f"❌ Test Sequence {i+1}: Error - {e}")

def main():
    """Main training pipeline"""
    print_header("DNAAadeshak Optimized ML Model Training")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and sample data
    df_sample = load_and_sample_data(sample_size=10000, min_length=100, max_length=800)
    if df_sample is None:
        return
    
    # Extract features
    X, valid_indices = extract_optimized_features(df_sample['sequence'].values)
    if len(X) == 0:
        print("❌ No valid features extracted")
        return
    
    # Get corresponding labels
    y = df_sample.iloc[valid_indices]['class'].values
    
    print(f"📊 Final dataset:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)}")
    
    # Train model
    model, X_test, y_test, y_pred, accuracy = train_optimized_model(X, y)
    
    # Evaluate performance
    final_accuracy = evaluate_model_performance(model, X_test, y_test, y_pred)
    
    # Save the optimized model
    print_header("Saving Optimized Model")
    
    try:
        # Save the trained model
        joblib.dump(model, 'dna_model_optimized.pkl')
        print("✅ Optimized model saved as 'dna_model_optimized.pkl'")
        
        # Also save as the default model for the app
        joblib.dump(model, 'dna_model.pkl')
        print("✅ Model saved as 'dna_model.pkl' for app usage")
        
        # Save model metadata
        metadata = {
            'accuracy': final_accuracy,
            'training_samples': len(X),
            'features': X.shape[1],
            'classes': list(np.unique(y)),
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier_Optimized'
        }
        
        import json
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Model metadata saved")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Test predictions
    analyzer = MLDNAAnalyzer()
    test_model_predictions(model, analyzer)
    
    # Final summary
    print_header("Training Summary")
    
    if final_accuracy >= 0.95:
        performance_level = "Excellent"
        emoji = "🏆"
    elif final_accuracy >= 0.90:
        performance_level = "Very Good"
        emoji = "🥇"
    elif final_accuracy >= 0.85:
        performance_level = "Good"
        emoji = "🥈"
    else:
        performance_level = "Fair"
        emoji = "🥉"
    
    print(f"{emoji} Model Performance: {performance_level}")
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"📊 Training Samples: {len(X):,}")
    print(f"🎯 Features: {X.shape[1]}")
    print(f"🏷️  Classes: {len(np.unique(y))}")
    print(f"⚡ Optimized for shorter sequences (100-800 bp)")
    print(f"💾 Model saved and ready for production use")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, final_accuracy

if __name__ == "__main__":
    model, accuracy = main()
    print(f"\n🎉 Optimized model training completed with {accuracy:.2%} accuracy!")
