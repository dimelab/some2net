#!/usr/bin/env python3
"""
Check Streamlit configuration and server settings.
Run this to diagnose 413 upload errors.
"""

import streamlit as st
import os
import sys
from pathlib import Path

st.title("🔍 Streamlit Configuration Checker")

st.header("1. Environment Information")
st.write(f"**Python Version:** {sys.version}")
st.write(f"**Streamlit Version:** {st.__version__}")
st.write(f"**Working Directory:** {os.getcwd()}")

st.header("2. Configuration File")
config_path = Path(".streamlit/config.toml")
if config_path.exists():
    st.success(f"✅ Config file exists at: {config_path.absolute()}")
    st.code(config_path.read_text(), language="toml")
else:
    st.error(f"❌ Config file NOT found at: {config_path.absolute()}")

st.header("3. Streamlit Server Config")
st.write("**Note:** These values are read at server startup.")

# Try to read config
try:
    from streamlit import config as st_config

    st.write("**Server Configuration:**")
    config_dict = {
        "server.maxUploadSize": st_config.get_option("server.maxUploadSize"),
        "server.maxMessageSize": st_config.get_option("server.maxMessageSize"),
        "server.port": st_config.get_option("server.port"),
        "server.headless": st_config.get_option("server.headless"),
    }

    for key, value in config_dict.items():
        st.write(f"- **{key}:** {value}")

    # Check if limits are reasonable
    max_upload = st_config.get_option("server.maxUploadSize")
    if max_upload < 500:
        st.warning(f"⚠️ Upload limit is only {max_upload}MB. This may cause 413 errors for large files.")
    else:
        st.success(f"✅ Upload limit is {max_upload}MB - should handle large files.")

except Exception as e:
    st.error(f"❌ Could not read Streamlit config: {e}")

st.header("4. Session State Size")
total_size = 0
large_items = []

for key, value in st.session_state.items():
    try:
        import sys
        size = sys.getsizeof(value)
        total_size += size
        if size > 1_000_000:  # > 1MB
            large_items.append((key, size / (1024*1024)))
    except:
        pass

st.write(f"**Total session state size:** {total_size / (1024*1024):.2f} MB")

if large_items:
    st.warning("⚠️ Large items in session state (may cause 413 on subsequent uploads):")
    for key, size_mb in sorted(large_items, key=lambda x: x[1], reverse=True):
        st.write(f"- `{key}`: {size_mb:.2f} MB")
else:
    st.success("✅ No large items in session state")

st.header("5. Test Upload")
st.write("Try uploading a small file to test:")
test_file = st.file_uploader("Upload test file", type=['txt', 'csv'])
if test_file:
    size = len(test_file.getvalue())
    st.success(f"✅ Upload successful! File size: {size / (1024*1024):.2f} MB")

st.header("6. Recommended Actions")

st.markdown("""
### If you're getting 413 errors:

1. **Stop and restart Streamlit completely**
   ```bash
   pkill -f streamlit
   streamlit run src/cli/app.py --server.maxUploadSize=2000
   ```

2. **Check for reverse proxy** (nginx, apache, traefik)
   - If using nginx: Add `client_max_body_size 2000M;` to nginx config
   - If using apache: Add `LimitRequestBody 2097152000` to apache config

3. **Clear browser cache**
   - Hard refresh (Ctrl+Shift+R / Cmd+Shift+R)
   - Or try incognito mode

4. **Check session state**
   - If large items shown above, clear session state before uploading
   - Click "Clear" button in the main app

5. **Try smaller file first**
   - Upload just 1000 rows to test
   - Then try full file
""")

st.header("7. Environment Variables")
st.write("Relevant environment variables:")
env_vars = {
    "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": os.getenv("STREAMLIT_SERVER_MAX_UPLOAD_SIZE"),
    "STREAMLIT_SERVER_MAX_MESSAGE_SIZE": os.getenv("STREAMLIT_SERVER_MAX_MESSAGE_SIZE"),
}

for key, value in env_vars.items():
    if value:
        st.write(f"- `{key}`: {value}")
    else:
        st.write(f"- `{key}`: (not set)")

st.header("8. How Streamlit Was Started")
st.info("""
The config is only loaded at startup. If you:
1. Created/modified `.streamlit/config.toml`
2. But didn't restart Streamlit

Then the old limits are still in effect!

**Solution:** Completely stop and restart Streamlit.
""")
