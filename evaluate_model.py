#!/usr/bin/env python3
"""Evaluate DNA model accuracy on test data"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from ml_dna_analyzer import MLDNAAnalyzer
from dna_identity_matcher import DNAIdentityMatcher
import time

def load_test_data(sample_size=1000):
    """Load a sample of test data from the synthetic dataset"""
    print("Loading test data...")
    df = pd.read_csv('synthetic_dna_dataset.csv')
    
    # Ensure we have a balanced sample
    sample_per_class = sample_size // 3
    samples = []
    
    for class_name in ['gene', 'promoter', 'junk']:
        class_samples = df[df['class'] == class_name].sample(
            sample_per_class, 
            random_state=42
        )
        samples.append(class_samples)
    
    test_df = pd.concat(samples, axis=0).sample(frac=1, random_state=42)
    return test_df

def evaluate_model():
    """Evaluate the DNA model on test data"""
    print("🔍 Evaluating DNA Model Accuracy")
    print("=" * 50)
    
    # Load the model
    print("Loading DNA analyzer...")
    analyzer = MLDNAAnalyzer()
    if not analyzer.load_model():
        print("❌ Failed to load model")
        return
    
    # Load test data
    test_df = load_test_data(1000)  # Use 1000 samples for evaluation
    print(f"\nEvaluating on {len(test_df)} test sequences...")
    
    # Make predictions
    y_true = []
    y_pred = []
    probabilities = []
    
    start_time = time.time()
    
    for _, row in test_df.iterrows():
        sequence = row['sequence']
        true_class = row['class']
        
        try:
            # Get model prediction
            result = analyzer.predict(sequence)
            pred_class = result['class']
            prob = result['probability']
            
            y_true.append(true_class)
            y_pred.append(pred_class)
            probabilities.append(prob)
            
        except Exception as e:
            print(f"Error processing sequence: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    
    # Calculate average confidence
    avg_confidence = np.mean(probabilities) * 100
    
    # Calculate inference time
    total_time = time.time() - start_time
    avg_inference_time = (total_time / len(test_df)) * 1000  # ms per sequence
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\n🔹 Accuracy: {accuracy:.2%}")
    print(f"🔹 Average Confidence: {avg_confidence:.1f}%")
    print(f"🔹 Total Test Sequences: {len(test_df)}")
    print(f"🔹 Average Inference Time: {avg_inference_time:.2f} ms/sequence")
    
    print("\n📈 Classification Report:")
    print(class_report)
    
    # Plot confusion matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=analyzer.label_encoder.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=analyzer.label_encoder.classes_)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('DNA Sequence Classification Confusion Matrix')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_confusion_matrix.png')
    print("\n✅ Confusion matrix saved as 'model_confusion_matrix.png'")
    
    # Show top misclassifications
    print("\n🔍 Top Misclassifications:")
    misclassified = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'true': true,
                'predicted': pred,
                'sequence': test_df.iloc[i]['sequence'][:50] + '...',
                'length': len(test_df.iloc[i]['sequence'])
            })
    
    if misclassified:
        misclassified_df = pd.DataFrame(misclassified)
        print(f"\nFound {len(misclassified)} misclassified sequences (showing first 5):")
        print(misclassified_df.head())
    else:
        print("No misclassifications found!")

if __name__ == "__main__":
    evaluate_model()
