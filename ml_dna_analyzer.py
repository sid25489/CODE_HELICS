import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from collections import Counter
import re

class MLDNAAnalyzer:
    def __init__(self, dataset_path='synthetic_dna_dataset.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = None
        self.feature_names = []
        self.is_trained = False
        
    def extract_features(self, sequence):
        """Extract comprehensive features from DNA sequence"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Basic composition features
        features = {}
        
        # Nucleotide composition
        total_length = len(sequence)
        if total_length == 0:
            return np.zeros(100)  # Return zero vector for empty sequences
            
        features['length'] = total_length
        features['gc_content'] = (sequence.count('G') + sequence.count('C')) / total_length
        features['at_content'] = (sequence.count('A') + sequence.count('T')) / total_length
        
        # Individual nucleotide frequencies
        for nucleotide in ['A', 'T', 'G', 'C']:
            features[f'{nucleotide}_freq'] = sequence.count(nucleotide) / total_length
        
        # Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                        'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        for dinuc in dinucleotides:
            count = 0
            for i in range(len(sequence) - 1):
                if sequence[i:i+2] == dinuc:
                    count += 1
            features[f'{dinuc}_freq'] = count / max(1, total_length - 1)
        
        # Trinucleotide (codon) frequencies - top 20 most common
        trinucleotides = ['ATG', 'TAA', 'TAG', 'TGA', 'AAA', 'TTT', 'GGG', 'CCC',
                         'ATC', 'TCG', 'CGA', 'GAT', 'GCT', 'CTA', 'TTA', 'AAC',
                         'CCG', 'CGG', 'GGA', 'AAT']
        for trinuc in trinucleotides:
            count = 0
            for i in range(len(sequence) - 2):
                if sequence[i:i+3] == trinuc:
                    count += 1
            features[f'{trinuc}_freq'] = count / max(1, total_length - 2)
        
        # Sequence complexity measures
        # Entropy calculation
        nucleotide_counts = [sequence.count(n) for n in ['A', 'T', 'G', 'C']]
        entropy = 0
        for count in nucleotide_counts:
            if count > 0:
                p = count / total_length
                entropy -= p * np.log2(p)
        features['entropy'] = entropy
        
        # Repetitive elements
        features['max_homopolymer'] = self._max_homopolymer_length(sequence)
        
        # ORF (Open Reading Frame) features
        features['num_start_codons'] = sequence.count('ATG')
        features['num_stop_codons'] = sequence.count('TAA') + sequence.count('TAG') + sequence.count('TGA')
        
        # Ensure consistent feature order
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        # Create feature vector in consistent order
        feature_vector = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name in features:
                feature_vector[i] = features[name]
            
        return feature_vector
    
    def _max_homopolymer_length(self, sequence):
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
    
    def load_and_prepare_data(self):
        """Load dataset and prepare features"""
        print("Loading dataset...")
        
        # Read dataset in chunks to handle large file
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(self.dataset_path, chunksize=chunk_size):
            # Filter out rows with invalid sequences
            chunk = chunk[chunk['sequence'].str.match(r'^[ATCG]+$', na=False)]
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df)} valid sequences")
        
        # Extract features for all sequences
        print("Extracting features...")
        X = []
        y = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing sequence {idx}/{len(df)}")
                
            try:
                features = self.extract_features(row['sequence'])
                X.append(features)
                y.append(row['class'])
            except Exception as e:
                print(f"Error processing sequence {idx}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        
        return X, y
    
    def train_model(self):
        """Train the machine learning model"""
        print("Training ML model...")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        important_features = sorted(zip(self.feature_names, feature_importance), 
                                  key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 Important Features:")
        for feature, importance in important_features:
            print(f"{feature}: {importance:.3f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def predict(self, sequence):
        """Predict class for a DNA sequence with robust error handling"""
        try:
            if not self.is_trained:
                if not self.load_model():
                    # Fallback to pattern-based prediction
                    return self._pattern_based_prediction(sequence)
            
            # Extract features
            features = self.extract_features(sequence)
            
            # Handle both dictionary and numpy array returns
            if isinstance(features, dict):
                feature_vector = np.array(list(features.values())).reshape(1, -1)
            else:
                # Already a numpy array
                feature_vector = features.reshape(1, -1)
            
            # Check if feature count matches model expectations
            expected_features = len(self.feature_names) if self.feature_names else feature_vector.shape[1]
            if feature_vector.shape[1] != expected_features:
                print(f"Feature mismatch: got {feature_vector.shape[1]}, expected {expected_features}")
                # Pad or truncate features to match
                if feature_vector.shape[1] < expected_features:
                    # Pad with zeros
                    padding = np.zeros((1, expected_features - feature_vector.shape[1]))
                    feature_vector = np.hstack([feature_vector, padding])
                else:
                    # Truncate
                    feature_vector = feature_vector[:, :expected_features]
            
            # Make prediction
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            # Get class name
            class_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get probability for predicted class
            max_prob = max(probabilities)
            
            return {
                'class': class_name,
                'probability': float(max_prob),
                'all_probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            # Fallback to pattern-based prediction
            return self._pattern_based_prediction(sequence)
    
    def _pattern_based_prediction(self, sequence):
        """Fallback pattern-based prediction"""
        sequence = sequence.upper()
        
        # Simple pattern-based classification
        if 'ATG' in sequence and any(stop in sequence for stop in ['TAA', 'TAG', 'TGA']):
            return {'class': 'gene', 'probability': 0.7, 'all_probabilities': {'gene': 0.7, 'promoter': 0.2, 'junk': 0.1}}
        elif sequence.count('G') + sequence.count('C') > len(sequence) * 0.6:
            return {'class': 'promoter', 'probability': 0.6, 'all_probabilities': {'promoter': 0.6, 'gene': 0.3, 'junk': 0.1}}
        else:
            return {'class': 'junk', 'probability': 0.5, 'all_probabilities': {'junk': 0.5, 'gene': 0.3, 'promoter': 0.2}}
    
    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, 'dna_ml_model.pkl')
        print("Model saved to dna_ml_model.pkl")
    
    def load_model(self):
        """Load trained model from disk"""
        # Try multiple model files in order of preference
        model_files = ['dna_model.pkl', 'dna_model_kmer.pkl', 'dna_model_high_accuracy.pkl', 'dna_ml_model.pkl']
        
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                print(f"Model loaded successfully from {model_file}")
                return True
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading model from {model_file}: {e}")
                continue
        
        print("No saved model found")
        return False
    
    def validate_sequence(self, sequence):
        """Validate DNA sequence"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        if not re.match('^[ATCG]+$', sequence):
            raise ValueError("Invalid DNA sequence. Only A, T, C, G characters allowed.")
        return sequence

# Training script
if __name__ == "__main__":
    analyzer = MLDNAAnalyzer()
    
    # Check if model exists, if not train it
    if not analyzer.load_model():
        print("Training new model...")
        accuracy = analyzer.train_model()
        print(f"Model trained with accuracy: {accuracy:.3f}")
    else:
        print("Using existing trained model")
    
    # Test with a sample sequence
    test_sequence = "ATCGATCGATCGATCG"
    try:
        results = analyzer.predict(test_sequence)
        print(f"\nTest prediction for sequence: {test_sequence}")
        for result in results:
            print(f"{result['identity']}: {result['probability']:.3f} ({result['confidence']})")
    except Exception as e:
        print(f"Error in prediction: {e}")
