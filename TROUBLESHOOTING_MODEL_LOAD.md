# Troubleshooting Model Loading Issues

## Error: AttributeError: 'NoneType' object has no attribute 'endswith'

This error occurs when the transformers library's tokenizer cache is corrupted or incomplete.

### Quick Fix

The application now automatically tries to use the slow tokenizer (`use_fast=False`) which should resolve this issue in most cases.

### If the Error Persists

If you still get this error, try these solutions:

#### Solution 1: Clear Transformers Cache

```bash
# Remove the transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Or if using custom cache directory
rm -rf ./models/
```

Then restart the application - it will re-download the model.

#### Solution 2: Manually Download the Model

```bash
# Run this Python command to pre-download the model
python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
AutoTokenizer.from_pretrained('Davlan/xlm-roberta-base-ner-hrl'); \
AutoModelForTokenClassification.from_pretrained('Davlan/xlm-roberta-base-ner-hrl')"
```

#### Solution 3: Upgrade Transformers

```bash
pip install --upgrade transformers
```

#### Solution 4: Use CPU Instead of GPU

Sometimes GPU-specific model files get corrupted. Try CPU mode:

In the Streamlit sidebar, or when initializing the pipeline, ensure you're not forcing GPU usage if you're having issues.

### Understanding the Error

This error happens when:
1. The model download was interrupted
2. The tokenizer files are incomplete or corrupted
3. There's a version mismatch between transformers and the cached model files

### Prevention

To avoid this issue in the future:
- Ensure stable internet connection when first downloading models
- Use the latest version of transformers
- Avoid manually modifying the `~/.cache/huggingface/` directory

### Still Having Issues?

If none of the above works:

1. Check your transformers version:
   ```bash
   pip show transformers
   ```
   Should be >= 4.30.0

2. Try a different model in the dropdown (e.g., "Babelscape/wikineural-multilingual-ner")

3. Check disk space - model downloads require several GB

4. Report the issue with full error trace at: https://github.com/dimelab/some2net/issues
