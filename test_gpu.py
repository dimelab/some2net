"""
GPU Acceleration Test Script
Tests CUDA/MPS availability and spaCy GPU usage
"""

import sys
import subprocess

print("=" * 70)
print("GPU ACCELERATION DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Python and system info
print("\n1. SYSTEM INFORMATION")
print("-" * 70)
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Test 2: Check PyTorch
print("\n2. PYTORCH")
print("-" * 70)
try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")

    print(f"  MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS built: {torch.backends.mps.is_built()}")

    # Test actual tensor creation
    print("\n  Testing tensor creation:")
    cpu_tensor = torch.randn(3, 3)
    print(f"    CPU tensor: {cpu_tensor.device}")

    if torch.cuda.is_available():
        cuda_tensor = torch.randn(3, 3).cuda()
        print(f"    CUDA tensor: {cuda_tensor.device}")
    elif torch.backends.mps.is_available():
        try:
            mps_tensor = torch.randn(3, 3).to('mps')
            print(f"    MPS tensor: {mps_tensor.device}")
        except Exception as e:
            print(f"    ✗ MPS tensor failed: {e}")

except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")

# Test 3: Check spaCy
print("\n3. SPACY")
print("-" * 70)
try:
    import spacy
    print(f"✓ spaCy installed: {spacy.__version__}")

    # Check spaCy's GPU preference
    print(f"  spaCy prefer_gpu(): {spacy.prefer_gpu()}")
    print(f"  spaCy require_gpu(): ", end="")
    try:
        spacy.require_gpu()
        print("True")
    except Exception as e:
        print(f"False ({e})")

    # Try to load a model and check device
    print("\n  Testing model loading:")
    try:
        # Try to load a small model
        nlp = spacy.load("en_core_web_sm")
        print(f"    ✓ Loaded en_core_web_sm")

        # Check if model uses GPU
        if nlp.has_pipe("ner"):
            ner = nlp.get_pipe("ner")
            print(f"    NER component: {type(ner)}")
            if hasattr(ner, 'model'):
                print(f"    NER model type: {type(ner.model)}")
                # Try to get device info
                try:
                    import thinc
                    print(f"    Thinc version: {thinc.__version__}")
                    from thinc.api import get_current_ops
                    ops = get_current_ops()
                    print(f"    Current ops: {type(ops).__name__}")
                    print(f"    Using GPU: {hasattr(ops, 'xp') and ops.xp.__name__ == 'cupy'}")
                except Exception as e:
                    print(f"    Could not check ops: {e}")
    except OSError:
        print("    ✗ en_core_web_sm not installed")
        print("    Run: python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"    ✗ Error loading model: {e}")

except ImportError as e:
    print(f"✗ spaCy not installed: {e}")

# Test 4: Check Thinc (spaCy's ML library)
print("\n4. THINC (spaCy's ML backend)")
print("-" * 70)
try:
    import thinc
    print(f"✓ Thinc installed: {thinc.__version__}")

    from thinc.api import prefer_gpu, require_gpu
    print(f"  Thinc prefer_gpu(): {prefer_gpu()}")
    print(f"  Thinc require_gpu(): ", end="")
    try:
        require_gpu()
        print("True")
    except Exception as e:
        print(f"False ({e})")

except ImportError as e:
    print(f"✗ Thinc not installed: {e}")

# Test 5: Check CuPy (CUDA array library)
print("\n5. CUPY (CUDA arrays for Python)")
print("-" * 70)
try:
    import cupy as cp
    print(f"✓ CuPy installed: {cp.__version__}")
    print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

    # Test array creation
    try:
        arr = cp.array([1, 2, 3])
        print(f"  ✓ Can create CuPy arrays")
        print(f"  Device: {arr.device}")
    except Exception as e:
        print(f"  ✗ Cannot create CuPy arrays: {e}")

except ImportError:
    print("✗ CuPy not installed")
    print("  For CUDA GPU: pip install cupy-cuda11x  (or cupy-cuda12x)")
    print("  Note: CuPy is not needed for Apple Silicon (MPS)")

# Test 6: Environment variables
print("\n6. ENVIRONMENT VARIABLES")
print("-" * 70)
import os
gpu_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PYTORCH_ENABLE_MPS_FALLBACK']
for var in gpu_vars:
    value = os.environ.get(var, 'Not set')
    print(f"  {var}: {value}")

# Test 7: Recommendations
print("\n7. RECOMMENDATIONS")
print("-" * 70)

# Determine system type
if sys.platform == "darwin":
    print("  System: macOS (Apple Silicon or Intel)")
    if torch.backends.mps.is_available():
        print("  ✓ MPS is available - you have Apple Silicon GPU!")
        print("\n  To enable in spaCy:")
        print("    1. Make sure you're using spaCy 3.5+")
        print("    2. Install PyTorch with MPS support:")
        print("       pip install torch torchvision torchaudio")
        print("    3. Configure spaCy to use GPU in your code:")
        print("       import spacy")
        print("       spacy.prefer_gpu()")
        print("       nlp = spacy.load('your_model')")
    else:
        print("  ✗ MPS not available")
        print("    - If you have Apple Silicon (M1/M2/M3), update PyTorch")
        print("    - If you have Intel Mac, CUDA is not available")
else:
    print("  System: Linux/Windows")
    if torch.cuda.is_available():
        print("  ✓ CUDA is available!")
        print("\n  To enable in spaCy:")
        print("    1. Install CuPy:")
        cuda_version = torch.version.cuda
        if cuda_version:
            major = cuda_version.split('.')[0]
            print(f"       pip install cupy-cuda{major}x")
        else:
            print("       pip install cupy-cuda11x  # or cuda12x depending on your CUDA version")
        print("    2. Configure spaCy in your code:")
        print("       import spacy")
        print("       spacy.require_gpu()")
        print("       nlp = spacy.load('your_model')")
    else:
        print("  ✗ CUDA not available")
        print("    - Install NVIDIA drivers")
        print("    - Install CUDA toolkit")
        print("    - Reinstall PyTorch with CUDA support:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Test 8: Quick benchmark
print("\n8. QUICK PERFORMANCE TEST")
print("-" * 70)
try:
    import torch
    import time

    size = 1000
    iterations = 100

    # CPU test
    start = time.time()
    for _ in range(iterations):
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        c = torch.mm(a, b)
    cpu_time = time.time() - start
    print(f"  CPU: {cpu_time:.3f}s for {iterations} matrix multiplications")

    # GPU test
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = None

    if device:
        start = time.time()
        for _ in range(iterations):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.mm(a, b)
        if device == 'mps':
            torch.mps.synchronize()  # Wait for MPS operations to complete
        elif device == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU ({device}): {gpu_time:.3f}s for {iterations} matrix multiplications")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("  GPU: Not available for testing")

except Exception as e:
    print(f"  Error during benchmark: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
