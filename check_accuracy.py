#!/usr/bin/env python3
"""Check model accuracy on the enhanced dataset"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ml_dna_analyzer import MLDNAAnalyzer
import time

def evaluate_current_model():
    """Evaluate the current model on enhanced dataset"""
    print("🔍 Evaluating Current Model on Enhanced Dataset")
    print("=" * 60)
    
    # Load enhanced dataset
    print("Loading enhanced dataset...")
    df = pd.read_csv('synthetic_dna_dataset.csv')
    print(f"Dataset size: {len(df)} sequences")
    
    # Show class distribution
    class_counts = df['class'].value_counts()
    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Sample balanced test set
    print("\nPreparing test set...")
    test_size = 1500  # 500 per class
    samples_per_class = test_size // 3
    
    test_samples = []
    for class_name in ['gene', 'promoter', 'junk']:
        class_data = df[df['class'] == class_name].sample(
            samples_per_class, random_state=42
        )
        test_samples.append(class_data)
    
    test_df = pd.concat(test_samples, axis=0).sample(frac=1, random_state=42)
    print(f"Test set size: {len(test_df)} sequences")
    
    # Load current model
    print("\nLoading current model...")
    analyzer = MLDNAAnalyzer()
    if not analyzer.load_model():
        print("❌ Failed to load model. Please train a model first.")
        return
    
    # Make predictions
    print("Making predictions...")
    y_true = []
    y_pred = []
    prediction_times = []
    
    start_time = time.time()
    
    for idx, row in test_df.iterrows():
        sequence = row['sequence']
        true_class = row['class']
        
        pred_start = time.time()
        try:
            result = analyzer.predict(sequence)
            pred_class = result['class']
            confidence = result['probability']
            
            y_true.append(true_class)
            y_pred.append(pred_class)
            prediction_times.append(time.time() - pred_start)
            
        except Exception as e:
            print(f"Error predicting sequence {idx}: {e}")
            y_true.append(true_class)
            y_pred.append('unknown')
            prediction_times.append(0)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    avg_prediction_time = np.mean(prediction_times) * 1000  # ms
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
    print(f"⏱️  Average Prediction Time: {avg_prediction_time:.2f} ms")
    print(f"📈 Total Evaluation Time: {total_time:.2f} seconds")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=['gene', 'promoter', 'junk'])
    classes = ['gene', 'promoter', 'junk']
    
    print(f"{'':>12}", end="")
    for cls in classes:
        print(f"{cls:>10}", end="")
    print()
    
    for i, true_cls in enumerate(classes):
        print(f"{true_cls:>12}", end="")
        for j, pred_cls in enumerate(classes):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for i, cls in enumerate(classes):
        class_accuracy = cm[i][i] / cm[i].sum()
        print(f"  {cls}: {class_accuracy:.2%}")
    
    # Analyze misclassifications
    print(f"\n🔍 Misclassification Analysis:")
    misclassified = [(true, pred) for true, pred in zip(y_true, y_pred) if true != pred]
    
    if misclassified:
        misclass_counts = {}
        for true_cls, pred_cls in misclassified:
            key = f"{true_cls} → {pred_cls}"
            misclass_counts[key] = misclass_counts.get(key, 0) + 1
        
        print(f"Total misclassifications: {len(misclassified)}")
        for error_type, count in sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    else:
        print("🎉 No misclassifications found!")
    
    # Performance assessment
    print(f"\n🏆 Performance Assessment:")
    if accuracy >= 0.95:
        print("✅ EXCELLENT: Model accuracy is 95% or higher")
    elif accuracy >= 0.90:
        print("✅ GOOD: Model accuracy is 90-95%")
    elif accuracy >= 0.80:
        print("⚠️  FAIR: Model accuracy is 80-90%")
    else:
        print("❌ POOR: Model accuracy is below 80%")
    
    return accuracy

def compare_with_pattern_matching():
    """Compare ML model with pattern-based fallback"""
    print("\n" + "=" * 60)
    print("🔄 COMPARING WITH PATTERN MATCHING")
    print("=" * 60)
    
    # Load test data
    df = pd.read_csv('synthetic_dna_dataset.csv')
    test_samples = df.sample(300, random_state=42)  # Smaller sample for comparison
    
    analyzer = MLDNAAnalyzer()
    
    ml_correct = 0
    pattern_correct = 0
    
    print("Comparing predictions...")
    for idx, row in test_samples.iterrows():
        sequence = row['sequence']
        true_class = row['class']
        
        # ML prediction
        try:
            analyzer.load_model()
            ml_result = analyzer.predict(sequence)
            ml_pred = ml_result['class']
        except:
            ml_pred = 'unknown'
        
        # Pattern-based prediction
        pattern_result = analyzer._pattern_based_prediction(sequence)
        pattern_pred = pattern_result['class']
        
        if ml_pred == true_class:
            ml_correct += 1
        if pattern_pred == true_class:
            pattern_correct += 1
    
    ml_accuracy = ml_correct / len(test_samples)
    pattern_accuracy = pattern_correct / len(test_samples)
    
    print(f"\n📊 Comparison Results:")
    print(f"  ML Model Accuracy: {ml_accuracy:.2%}")
    print(f"  Pattern Matching Accuracy: {pattern_accuracy:.2%}")
    print(f"  Improvement: {(ml_accuracy - pattern_accuracy)*100:.1f} percentage points")

if __name__ == "__main__":
    accuracy = evaluate_current_model()
    compare_with_pattern_matching()
