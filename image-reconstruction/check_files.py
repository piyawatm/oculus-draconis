"""
BDH-VQVAE File Structure Verification
Checks that all necessary files are present.
"""

import os
import sys

print("="*60)
print("BDH-VQVAE File Structure Check")
print("="*60)

required_files = {
    'Models': [
        'models/__init__.py',
        'models/quantizer.py',
        'models/vqvae.py',
        'models/vqvae2.py',
        'models/bdh_vqvae.py'
    ],
    'Scripts': [
        'train.py',
        'evaluate.py',
        'compare.py',
        'example.py',
        'utils.py',
        'config.py'
    ],
    'Documentation': [
        'README.md',
        'BDH_UPDATE.md',
        'QUICKSTART.md',
        'ARCHITECTURE.md'
    ],
    'Config': [
        'requirements.txt',
        'run_experiments.sh'
    ]
}

all_present = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for filepath in files:
        exists = os.path.exists(filepath)
        status = '✓' if exists else '✗'
        print(f"  {status} {filepath}")
        if not exists:
            all_present = False

print("\n" + "="*60)

if all_present:
    print("✓✓✓ All files present!")
    print("="*60)
    print("\nBDH-VQVAE Features:")
    print("  ✓ Baby Dragon Hatchling attention layers")
    print("  ✓ Sparse ReLU attention (no softmax)")
    print("  ✓ Q=K constraint (self-similarity)")
    print("  ✓ RoPE position embeddings")
    print("  ✓ Multiplicative gating")
    print("  ✓ Bio-inspired sparse activations")
    
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Train: python train.py --model bdh_vqvae --dataset cifar10")
    print("  3. Evaluate: python evaluate.py --model bdh_vqvae")
    print("  4. Read BDH_UPDATE.md for details on what makes BDH special")
    print("\n" + "="*60)
else:
    print("✗ Some files are missing!")
    print("="*60)
    print("\nPlease ensure you have the complete codebase.")
    sys.exit(1)

# Check if we can import the models (if torch is installed)
print("\nTrying to import models...")
try:
    from models import VQVAE, VQVAE2, BDHVQVAE
    print("✓ Successfully imported all models!")
    print("  - VQVAE (vanilla baseline)")
    print("  - VQVAE2 (hierarchical)")
    print("  - BDHVQVAE (with Baby Dragon Hatchling attention)")
except ImportError as e:
    print(f"⚠ Cannot import models yet: {e}")
    print("  → This is expected if PyTorch is not installed")
    print("  → Run: pip install -r requirements.txt")

print("\n✓ File structure verification complete!")