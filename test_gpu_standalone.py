#!/usr/bin/env python3
"""
GPU Acceleration Diagnostic Tool for spaCy NER Pipeline
Run this script in your virtual environment on your remote server

Usage:
    python test_gpu_standalone.py

Or with your virtual environment:
    /path/to/venv/bin/python test_gpu_standalone.py
"""

import sys
import os

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subsection(title):
    """Print a formatted subsection"""
    print(f"\n{title}")
    print("-" * 70)

def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    print_subsection("1. PyTorch Configuration")

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  CUDA device {i}: {torch.cuda.get_device_name(i)}")

        # Check MPS (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f"  MPS (Apple Silicon) available: {mps_available}")

        # Determine best device
        if cuda_available:
            device = 'cuda'
            print(f"\n  ✓ Recommended device: CUDA")
        elif mps_available:
            device = 'mps'
            print(f"\n  ✓ Recommended device: MPS (Apple Silicon)")
        else:
            device = 'cpu'
            print(f"\n  ⚠ No GPU available - will use CPU only")

        # Quick benchmark
        print(f"\n  Running quick benchmark...")
        import time

        size = 1000
        iterations = 50

        # CPU benchmark
        start = time.time()
        for _ in range(iterations):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            c = torch.mm(a, b)
        cpu_time = time.time() - start
        print(f"    CPU: {cpu_time:.3f}s")

        # GPU benchmark
        if device != 'cpu':
            start = time.time()
            for _ in range(iterations):
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.mm(a, b)
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
            gpu_time = time.time() - start
            print(f"    GPU ({device}): {gpu_time:.3f}s")
            print(f"    ⚡ Speedup: {cpu_time/gpu_time:.2f}x")

        return device

    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Install with: pip install torch")
        return None

def check_spacy(device):
    """Check spaCy installation and GPU support"""
    print_subsection("2. spaCy Configuration")

    try:
        import spacy
        print(f"✓ spaCy version: {spacy.__version__}")

        # Test GPU functions
        print(f"\n  Testing GPU activation:")

        # Try prefer_gpu
        gpu_activated = spacy.prefer_gpu()
        print(f"    spacy.prefer_gpu() returned: {gpu_activated}")

        # Try require_gpu
        print(f"    spacy.require_gpu(): ", end="")
        try:
            spacy.require_gpu()
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed ({e})")

        return True

    except ImportError:
        print("✗ spaCy not installed!")
        print("  Install with: pip install spacy")
        return False

def check_thinc():
    """Check Thinc (spaCy's ML backend)"""
    print_subsection("3. Thinc (spaCy's ML Backend)")

    try:
        import thinc
        print(f"✓ Thinc version: {thinc.__version__}")

        from thinc.api import prefer_gpu, require_gpu, get_current_ops

        # Test GPU activation
        print(f"\n  Testing GPU activation:")
        gpu_activated = prefer_gpu()
        print(f"    thinc.api.prefer_gpu() returned: {gpu_activated}")

        print(f"    thinc.api.require_gpu(): ", end="")
        try:
            require_gpu()
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed ({e})")

        # Check current operations backend
        ops = get_current_ops()
        print(f"\n  Current operations backend: {type(ops).__name__}")

        if hasattr(ops, 'xp'):
            print(f"    Array library: {ops.xp.__name__}")

        return True

    except ImportError:
        print("✗ Thinc not installed!")
        return False

def check_cupy():
    """Check CuPy (needed for CUDA GPU support in spaCy)"""
    print_subsection("4. CuPy (CUDA Array Library)")

    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")

        # Test array creation
        try:
            arr = cp.array([1, 2, 3])
            print(f"  ✓ Can create CuPy arrays")
            print(f"  Device: {arr.device}")

            # Test computation
            result = cp.sum(arr)
            print(f"  ✓ Can perform computations")

            return True
        except Exception as e:
            print(f"  ✗ CuPy installed but cannot use GPU: {e}")
            return False

    except ImportError:
        print("✗ CuPy not installed")
        print("  Note: CuPy is REQUIRED for CUDA GPU support in spaCy")
        print("  Note: CuPy is NOT needed for Apple Silicon (MPS)")
        return False

def test_spacy_model(device):
    """Test loading and using a spaCy model with GPU"""
    print_subsection("5. spaCy Model GPU Test")

    try:
        import spacy

        # Activate GPU
        if device == 'cuda' or device == 'mps':
            print(f"  Activating GPU ({device})...")
            spacy.prefer_gpu()

        # Try to load a model
        print(f"\n  Loading spaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            print(f"    ✓ Loaded en_core_web_sm")
        except OSError:
            print(f"    ✗ en_core_web_sm not found")
            print(f"    Install with: python -m spacy download en_core_web_sm")
            return False

        # Check if model has NER
        if not nlp.has_pipe("ner"):
            print(f"    ✗ Model doesn't have NER component")
            return False

        # Get NER component
        ner = nlp.get_pipe("ner")
        print(f"    ✓ NER component found: {type(ner).__name__}")

        # Check device
        if hasattr(ner, 'model') and hasattr(ner.model, 'ops'):
            ops = ner.model.ops
            print(f"    Model ops: {type(ops).__name__}")
            if hasattr(ops, 'xp'):
                print(f"    Using array library: {ops.xp.__name__}")
                if ops.xp.__name__ == 'cupy':
                    print(f"    ✓ NER is using GPU (CUDA via CuPy)")
                elif ops.xp.__name__ == 'numpy':
                    print(f"    ⚠ NER is using CPU (NumPy)")
                else:
                    print(f"    ? Unknown array library")

        # Benchmark NER performance
        print(f"\n  Running NER benchmark...")
        import time

        test_text = "Apple Inc. is looking at buying U.K. startup for $1 billion. " * 10

        # Warmup
        _ = nlp(test_text)

        # Benchmark
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            doc = nlp(test_text)
        elapsed = time.time() - start

        print(f"    Processed {iterations} documents in {elapsed:.3f}s")
        print(f"    Rate: {iterations/elapsed:.1f} docs/sec")

        return True

    except Exception as e:
        print(f"  ✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_recommendations(device):
    """Print recommendations based on detected hardware"""
    print_section("RECOMMENDATIONS")

    if device == 'cuda':
        print("  Your system has NVIDIA GPU (CUDA)")
        print("\n  To enable GPU in your spaCy pipeline:")
        print("    1. Install CuPy (if not already installed):")
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            major = cuda_version.split('.')[0] if cuda_version else "11"
            print(f"       pip install cupy-cuda{major}x")

        print("\n    2. Add this to your pipeline code:")
        print("       import spacy")
        print("       spacy.require_gpu()  # Or spacy.prefer_gpu()")
        print("       nlp = spacy.load('your_model')")

    elif device == 'mps':
        print("  Your system has Apple Silicon GPU (MPS)")
        print("\n  To enable GPU in your spaCy pipeline:")
        print("    1. Make sure you have spaCy 3.5+ and PyTorch 2.0+")
        print("    2. No need to install CuPy for MPS")
        print("\n    3. Add this to your pipeline code:")
        print("       import spacy")
        print("       spacy.prefer_gpu()  # This will use MPS")
        print("       nlp = spacy.load('your_model')")
        print("\n  ⚠ Note: MPS support in spaCy may be limited")
        print("    Check spaCy docs for MPS compatibility")

    else:
        print("  No GPU detected")
        print("\n  Your pipeline will use CPU only")
        print("  Consider using a machine with:")
        print("    - NVIDIA GPU for CUDA acceleration")
        print("    - Apple Silicon (M1/M2/M3) for MPS acceleration")

def main():
    """Main diagnostic function"""
    print_section("GPU DIAGNOSTIC FOR SPACY NER PIPELINE")

    print(f"\nPython: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

    # Run diagnostics
    device = check_pytorch()
    spacy_installed = check_spacy(device) if device else False
    check_thinc()

    if device == 'cuda':
        cupy_available = check_cupy()
        if not cupy_available:
            print("\n  ⚠ WARNING: CUDA detected but CuPy not working!")
            print("    spaCy CANNOT use GPU without CuPy")

    if spacy_installed and device:
        test_spacy_model(device)

    print_recommendations(device)

    print_section("DIAGNOSTIC COMPLETE")
    print()

if __name__ == "__main__":
    main()
