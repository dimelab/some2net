# Troubleshooting CUDA/GPU Issues

## Error: "CUDA unknown error - this may be due to an incorrectly set up environment"

This error occurs when CUDA initialization fails, usually due to environment configuration issues.

### Quick Solution

The application will automatically fall back to CPU mode. It will work but be slower than GPU mode.

### Understanding the Warning

The warning indicates:
```
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment,
e.g. changing env variable CUDA_VISIBLE_DEVICES after program start.
```

This typically happens when:
1. CUDA environment variables are changed after Python starts
2. CUDA drivers are not properly installed
3. CUDA version mismatch with PyTorch
4. GPU is being used by another process

### Solutions

#### Solution 1: Set CUDA Variables Before Starting

```bash
# Set CUDA variables before running Streamlit
export CUDA_VISIBLE_DEVICES=0
streamlit run src/cli/app.py
```

#### Solution 2: Force CPU Mode

If you don't need GPU acceleration:

```bash
# Disable CUDA completely
export CUDA_VISIBLE_DEVICES=""
streamlit run src/cli/app.py
```

Or modify the code to always use CPU - edit `src/cli/app.py` and find where `NEREngine` is initialized, add `device="cpu"`.

#### Solution 3: Check CUDA Installation

```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

#### Solution 4: Reinstall PyTorch with Correct CUDA Version

If you have CUDA 11.8:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If you have CUDA 12.1:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only (no GPU):
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Solution 5: Restart Streamlit Fresh

Sometimes the issue is that Streamlit/Python was started with different environment variables:

```bash
# Kill all Python processes
pkill -f streamlit
pkill -f python

# Clear any cached state
rm -rf ~/.streamlit/

# Start fresh
export CUDA_VISIBLE_DEVICES=0
streamlit run src/cli/app.py
```

### Performance Impact

**GPU vs CPU Performance:**
- **GPU**: ~200-300 posts/second
- **CPU**: ~20-50 posts/second

For small datasets (< 1000 posts), CPU is usually fine. For larger datasets, GPU significantly speeds up processing.

### Verifying GPU Usage

When the model loads, you should see:
```
✅ GPU (CUDA) detected and initialized
🔄 Loading NER model: Davlan/xlm-roberta-base-ner-hrl
📱 Device: GPU (CUDA)
```

If using CPU:
```
ℹ️  No GPU detected, using CPU
🔄 Loading NER model: Davlan/xlm-roberta-base-ner-hrl
📱 Device: CPU
```

### Common Scenarios

#### Scenario 1: Running in Docker
If running in Docker, you need NVIDIA Container Toolkit:
```bash
# Install NVIDIA Container Toolkit
# Then run with GPU support
docker run --gpus all ...
```

#### Scenario 2: Remote Server
If SSH'd into a remote server:
```bash
# Check GPU availability
nvidia-smi

# Ensure CUDA variables are set in your session
echo $CUDA_VISIBLE_DEVICES
```

#### Scenario 3: Virtual Environment
Make sure PyTorch in your venv has CUDA support:
```bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True` if CUDA is available.

### Still Having Issues?

1. **Check system requirements:**
   - NVIDIA GPU with CUDA Compute Capability >= 3.5
   - CUDA Toolkit installed (version matching PyTorch)
   - NVIDIA drivers updated

2. **Try CPU mode explicitly:**
   In the code, force CPU by setting `device=-1` when initializing NEREngine

3. **Check logs:**
   Look for additional error messages in the Streamlit output

4. **Report issue:**
   If none of the above works, report at: https://github.com/dimelab/some2net/issues
   Include:
   - Output of `nvidia-smi`
   - Output of `python -c "import torch; print(torch.cuda.is_available())"`
   - Full error traceback

## Recommendations

For most users:
- **Development/Small datasets**: CPU mode is fine
- **Production/Large datasets**: Fix CUDA setup for better performance
- **Cloud/Docker**: Use CPU-only PyTorch for simpler deployment

The application works perfectly fine on CPU - it's just slower for large datasets.
