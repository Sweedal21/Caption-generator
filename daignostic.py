import os
import numpy as np
from pickle import load

print("=== MODEL DIAGNOSTIC ===")

# Check if files exist
print("\n1. Checking required files:")
files_to_check = [
    "tokenizer.p",
    "features.p", 
    "descriptions.txt",
    "models2/model_9.h5"
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✓ {file} exists ({size} bytes)")
    else:
        print(f"✗ {file} MISSING")

# Check tokenizer
print("\n2. Checking tokenizer:")
try:
    tokenizer = load(open("tokenizer.p", "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    print(f"✓ Tokenizer loaded successfully")
    print(f"✓ Vocabulary size: {vocab_size}")
    
    # Check for start/end tokens
    if 'start' in tokenizer.word_index:
        print(f"✓ 'start' token found at index {tokenizer.word_index['start']}")
    else:
        print("✗ 'start' token NOT found")
        
    if 'end' in tokenizer.word_index:
        print(f"✓ 'end' token found at index {tokenizer.word_index['end']}")
    else:
        print("✗ 'end' token NOT found")
        
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")

# Check features
print("\n3. Checking features:")
try:
    features = load(open("features.p", "rb"))
    print(f"✓ Features loaded successfully")
    print(f"✓ Number of images: {len(features)}")
    
    # Check feature shape
    first_key = list(features.keys())[0]
    feature_shape = features[first_key].shape
    print(f"✓ Feature shape: {feature_shape}")
    
except Exception as e:
    print(f"✗ Error loading features: {e}")

# Check descriptions
print("\n4. Checking descriptions:")
try:
    with open("descriptions.txt", "r") as f:
        lines = f.readlines()
    print(f"✓ Descriptions file loaded")
    print(f"✓ Number of lines: {len(lines)}")
    
    # Show sample
    if lines:
        print(f"✓ Sample line: {lines[0].strip()}")
    
except Exception as e:
    print(f"✗ Error loading descriptions: {e}")

print("\n=== DIAGNOSTIC COMPLETE ===")
print("\nIf you see any ✗ errors above, those need to be fixed first!")