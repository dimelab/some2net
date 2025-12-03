# Fix Nginx 413 Error for File Uploads

## Problem
Nginx is rejecting file uploads larger than 1MB with a 413 error, even though Streamlit is configured to accept larger files.

## Solution

You need to increase Nginx's `client_max_body_size` setting.

### Step 1: Find Your Nginx Config

The config file is usually in one of these locations:
- `/etc/nginx/nginx.conf` (main config)
- `/etc/nginx/sites-available/default` (site-specific)
- `/etc/nginx/sites-enabled/your-site` (site-specific)
- `/etc/nginx/conf.d/default.conf` (some systems)

### Step 2: Edit the Config

Add this line to your Nginx configuration:

```nginx
# Inside http, server, or location block
client_max_body_size 2000M;  # Allow up to 2GB uploads
```

#### Example 1: In the http block (applies to all sites)

```nginx
http {
    # ... other settings ...

    client_max_body_size 2000M;

    server {
        listen 80;
        # ... rest of config ...
    }
}
```

#### Example 2: In a specific server block (for this app only)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 2000M;  # Add this line

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Also add timeout settings for large file uploads
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

#### Example 3: In a specific location block

```nginx
location /myapp {
    client_max_body_size 2000M;  # Add this line

    proxy_pass http://localhost:8501;
    # ... rest of proxy settings ...
}
```

### Step 3: Test the Config

Before restarting Nginx, test that your config is valid:

```bash
sudo nginx -t
```

You should see:
```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

### Step 4: Reload Nginx

```bash
# Reload (preferred - no downtime)
sudo nginx -s reload

# OR restart (brief downtime)
sudo systemctl restart nginx

# OR (on some systems)
sudo service nginx restart
```

### Step 5: Verify

1. Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)
2. Try uploading your file again

## Additional Settings for Large Files

If you're uploading very large files (>500MB), you might also need:

```nginx
location / {
    # Upload size
    client_max_body_size 2000M;

    # Timeout settings (in seconds)
    client_body_timeout 300;      # Time to read client request body
    proxy_connect_timeout 300;    # Time to connect to backend
    proxy_send_timeout 300;       # Time to send to backend
    proxy_read_timeout 300;       # Time to read from backend

    # Buffer sizes for large uploads
    client_body_buffer_size 128k;
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;

    proxy_pass http://localhost:8501;
    # ... other proxy settings ...
}
```

## Quick Fix (Temporary)

If you don't have sudo access or want a quick test, you can try accessing Streamlit directly:

```
http://your-server:8501
```

This bypasses Nginx and connects directly to Streamlit (which already has 2GB limit configured).

**Note:** This only works if port 8501 is open in your firewall.

## Troubleshooting

### Still getting 413 after changes?

1. **Check you edited the right config file:**
   ```bash
   sudo nginx -T | grep client_max_body_size
   ```
   This shows the active config.

2. **Make sure Nginx reloaded:**
   ```bash
   sudo systemctl status nginx
   ```

3. **Clear browser cache:**
   - Hard refresh (Ctrl+Shift+R)
   - Or use incognito mode

4. **Check Nginx error log:**
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```
   Upload a file and see what error appears.

### Different values in different places?

If you have `client_max_body_size` in multiple places, the **most specific** location wins:
- `location` block overrides `server` block
- `server` block overrides `http` block

So make sure you add it in the right place for your setup!

## Recommended Settings for This App

```nginx
# In your Nginx config
http {
    # Global default (optional)
    client_max_body_size 100M;

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            # Override for this app - allow 2GB
            client_max_body_size 2000M;

            # Timeouts for large uploads
            client_body_timeout 600;
            proxy_connect_timeout 600;
            proxy_send_timeout 600;
            proxy_read_timeout 600;

            # Streamlit WebSocket settings
            proxy_pass http://localhost:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Done!

After making these changes and reloading Nginx, your file uploads should work up to 2GB.
