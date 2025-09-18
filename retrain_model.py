#!/usr/bin/env python3
"""Retrain the DNA classification model with improved features"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from ml_dna_analyzer import MLDNAAnalyzer

def load_and_prepare_data():
    """Load and prepare the synthetic DNA dataset"""
    print("Loading dataset...")
    df = pd.read_csv('synthetic_dna_dataset.csv')
    
    # Clean the data
    df = df.dropna()
    df = df[df['sequence'].str.match(r'^[ATCGatcg]+$')]
    
    # Ensure we have a balanced dataset
    min_samples = df['class'].value_counts().min()
    balanced_df = df.groupby('class').sample(n=min_samples, random_state=42)
    
    return balanced_df

def train_model():
    """Train and evaluate the DNA classification model"""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Initialize analyzer
    analyzer = MLDNAAnalyzer()
    
    # Extract features
    print("Extracting features...")
    X = np.array([analyzer.extract_features(seq) for seq in df['sequence']])
    y = df['class'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("=" * 50)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, 
                              target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = model.feature_importances_
    important_features = sorted(zip(analyzer.feature_names, feature_importance), 
                              key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 Important Features:")
    for feature, importance in important_features:
        print(f"{feature}: {importance:.4f}")
    
    # Save the model
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': analyzer.feature_names
    }
    
    joblib.dump(model_data, 'dna_model_improved.pkl')
    print("\n✅ Model saved as 'dna_model_improved.pkl'")
    
    # Also save as the default model
    joblib.dump(model_data, 'dna_model.pkl')
    print("✅ Model also saved as 'dna_model.pkl'")

if __name__ == "__main__":
    train_model()
