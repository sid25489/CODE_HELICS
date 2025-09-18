#!/usr/bin/env python3
"""Check model files structure and compatibility"""

import joblib
import os
import traceback

def check_model_file(filename):
    """Check individual model file"""
    print(f"\n=== Checking {filename} ===")
    
    if not os.path.exists(filename):
        print(f"❌ File does not exist: {filename}")
        return False
    
    try:
        data = joblib.load(filename)
        print(f"✅ File loaded successfully")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔑 Keys: {list(data.keys())}")
            
            # Check for required keys
            required_keys = ['model', 'label_encoder', 'feature_names']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys}")
                return False
            else:
                print(f"✅ All required keys present")
                print(f"🏷️  Model type: {type(data['model'])}")
                print(f"🏷️  Label encoder type: {type(data['label_encoder'])}")
                print(f"🏷️  Feature names count: {len(data['feature_names'])}")
                return True
        else:
            print(f"⚠️  Expected dict, got {type(data)}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 DNAAadeshak Model Files Debug Check")
    print("=" * 50)
    
    model_files = [
        'dna_model.pkl',
        'dna_model_kmer.pkl', 
        'dna_model_high_accuracy.pkl',
        'dna_model_optimized.pkl',
        'dna_ml_model.pkl'
    ]
    
    working_models = []
    
    for model_file in model_files:
        if check_model_file(model_file):
            working_models.append(model_file)
    
    print(f"\n{'='*50}")
    print(f"📈 SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Working models: {len(working_models)}")
    print(f"❌ Broken models: {len(model_files) - len(working_models)}")
    
    if working_models:
        print(f"🎯 Recommended model: {working_models[0]}")
    else:
        print(f"⚠️  No working models found - need to retrain!")

if __name__ == "__main__":
    main()
