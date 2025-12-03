# Troubleshooting Guide

## File Upload Issues

### Error 413: Request Entity Too Large

**Problem:** Getting "Request failed with status code 413" when uploading files.

**Cause:** Your file exceeds Streamlit's upload size limit.

**Solutions:**

1. **Increase the upload limit** (Recommended):
   - Edit `.streamlit/config.toml`
   - Change `maxUploadSize = 1000` to a higher value (in MB)
   - Restart the Streamlit app

   ```toml
   [server]
   maxUploadSize = 2000  # 2GB limit
   maxMessageSize = 2000
   ```

2. **Sample your data**:
   - Use only a subset of your data for testing
   - Process large files in chunks

3. **Compress your data**:
   - Remove unnecessary columns before upload
   - Use NDJSON instead of CSV (often smaller)

### File Upload Times Out

**Problem:** Upload takes forever or fails partway through.

**Solutions:**
- Increase timeout in `.streamlit/config.toml`:
  ```toml
  [server]
  fileWatcherType = "none"
  ```
- Check your network connection
- Try a smaller file first

## Processing Issues

### Out of Memory Errors

**Problem:** App crashes with memory errors during processing.

**Solutions:**
- Reduce chunk size (try 5,000 or 1,000)
- Reduce batch size (try 16 or 8)
- Close other applications
- Use a simpler extraction method (Hashtags instead of NER)

### Processing is Very Slow

**Problem:** Processing takes hours for medium-sized datasets.

**Solutions:**
- Check if GPU is being used (startup messages)
- Increase batch size if using GPU (try 64 or 128)
- Enable caching (for NER)
- Use a faster extraction method

## Visualization Issues

### Error 413 in Visualization

**Problem:** Getting 413 error after processing completes.

**Solutions:**
- The app automatically reduces visualization size
- Check "Giant Component Only" to show smaller network
- Download GEXF file and use Gephi instead

### Blank or Missing Visualization

**Problem:** Network doesn't appear after processing.

**Solutions:**
- Scroll down - it appears below the stats
- Check browser console for errors (F12)
- Try a different browser
- Download GEXF and open in Gephi

## Configuration Issues

### Streamlit Won't Start

**Problem:** Error when running `streamlit run src/cli/app.py`

**Solutions:**
- Check if Streamlit is installed: `pip list | grep streamlit`
- Reinstall: `pip install -U streamlit`
- Check Python version: `python --version` (need 3.9+)

### Port Already in Use

**Problem:** "Port 8501 is already in use"

**Solutions:**
- Use a different port: `streamlit run src/cli/app.py --server.port 8502`
- Kill the existing process
- Check what's using the port: `lsof -i :8501` (Mac/Linux) or `netstat -ano | findstr :8501` (Windows)

## Data Issues

### No Entities Extracted

**Problem:** Processing completes but finds 0 entities.

**Solutions:**
- Check text column actually has text
- For NER: Lower confidence threshold (try 0.5)
- Try a different extraction method
- Check the data preview - is it what you expected?

### Wrong Columns Selected

**Problem:** Processing fails or gives weird results.

**Solutions:**
- Check the data preview carefully
- Make sure author column has usernames/IDs
- Make sure text column has actual text content
- Re-select columns and try again

## Export Issues

### Download Doesn't Work

**Problem:** Clicking download button does nothing.

**Solutions:**
- Check browser allows downloads
- Try right-click → "Save Link As"
- Check browser's download settings
- Try a different browser

### File Won't Open in Gephi

**Problem:** Gephi shows error when opening GEXF file.

**Solutions:**
- Update Gephi to latest version
- Try GraphML format instead
- Check file isn't empty (should be >1KB)

## Platform-Specific Issues

### Windows

**Issues:**
- Path separators (use forward slashes in config)
- Encoding errors (make sure CSV is UTF-8)

**Solutions:**
- Save CSV as UTF-8 in Excel
- Use WSL2 for better compatibility

### Mac

**Issues:**
- M1/M2 chip compatibility
- CUDA not available

**Solutions:**
- Use CPU mode (will work, just slower)
- Install via conda for better M1/M2 support

### Linux

**Issues:**
- Missing system libraries

**Solutions:**
- Install dependencies: `apt-get install python3-dev`
- Check CUDA drivers if using GPU

## Getting Help

If none of these solutions work:

1. Check the error message carefully
2. Look in browser console (F12) for details
3. Restart the Streamlit app
4. Try with a very small test file (100 rows)
5. Report the issue with:
   - Error message
   - File size and format
   - Extraction method used
   - What you were doing when it happened
