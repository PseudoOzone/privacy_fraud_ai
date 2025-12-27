# Deployment Guide: Quick Start for Team Sharing

## 🚀 Fastest Way: Streamlit Cloud (Recommended - 5 minutes)

### Step 1: Create GitHub Repository
```bash
# In your project directory
git init
git add .
git commit -m "Privacy Fraud AI - Initial commit"
git branch -M main

# Create new repo on GitHub.com
# Then:
git remote add origin https://github.com/YOUR_USERNAME/privacy_fraud_ai.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Branch: `main`
6. Main file path: `notebooks/ui.py`
7. Click **"Deploy"**

### Step 3: Share URL
Once deployed, you get a permanent URL:
```
https://privacy-fraud-ai-YOUR_USERNAME.streamlit.app
```

Share this with collaborators! ✅

---

## 🐳 Docker Deployment (For Teams/Enterprise)

### Prerequisites
- Docker installed (https://www.docker.com/products/docker-desktop)

### Build & Run Locally
```bash
# Build image
docker build -t privacy-fraud-ai:latest .

# Run container
docker run -p 8506:8501 privacy-fraud-ai:latest
```

Access at: http://localhost:8506

### Push to Docker Hub (for team)
```bash
# Create account at hub.docker.com

# Login
docker login

# Tag image
docker tag privacy-fraud-ai:latest YOUR_USERNAME/privacy-fraud-ai:latest

# Push
docker push YOUR_USERNAME/privacy-fraud-ai:latest

# Share this command with team:
# docker run -p 8506:8501 YOUR_USERNAME/privacy-fraud-ai:latest
```

### Using Docker Compose (Easier)
```bash
# Single command to run everything
docker-compose up

# Access at http://localhost:8506
```

---

## 🌐 Public Tunnel: Ngrok (Quick Demo)

### Setup
```bash
# Install
winget install ngrok

# Create account at ngrok.com & get authtoken
ngrok config add-authtoken YOUR_TOKEN

# Run tunnel (in new terminal)
ngrok http 8506
```

### Share
Copy the forwarding URL:
```
https://abc123def456.ngrok.io
```

⚠️ **Note:** URL changes each time you restart. Keep app running.

---

## 📊 Sharing Options Comparison

| Option | Setup Time | Availability | Cost | Users |
|--------|-----------|--------------|------|-------|
| **Streamlit Cloud** | 5 min | Always on | Free (3 apps) | Unlimited |
| **Docker Hub** | 10 min | You decide | Free | Unlimited |
| **Ngrok** | 2 min | While running | Free (limits) | Limited |
| **Network URL** | 0 min | Local only | Free | Same network |

---

## 🔐 Security for Shared Access

### Input Validation
Already implemented in pipeline:
- ✅ CSV validation
- ✅ PII removal
- ✅ Data sanitization

### Rate Limiting (Optional)
Add to `ui.py` for public deployments:
```python
import time

# Limit file uploads
max_file_size = 50 * 1024 * 1024  # 50MB
max_upload_per_hour = 10

# Check file size
if uploaded_file.size > max_file_size:
    st.error("File too large")
```

### User Authentication (Optional)
Install Streamlit authentication:
```bash
pip install streamlit-authenticator
```

---

## 📋 Checklist for Sharing

- [ ] Choose deployment method
- [ ] Configure security settings
- [ ] Test with a collaborator
- [ ] Document usage instructions
- [ ] Set up update procedures
- [ ] Monitor performance/logs

---

## 👥 Team Access Instructions

**For Streamlit Cloud:**
1. Share URL: `https://privacy-fraud-ai-yourname.streamlit.app`
2. Users open link → Upload data → Run pipeline
3. No installation needed!

**For Docker:**
1. Install Docker
2. Run: `docker run -p 8506:8501 yourname/privacy-fraud-ai:latest`
3. Access: `http://localhost:8506`

**For Ngrok:**
1. Share public URL
2. Users can access from anywhere
3. URL changes on restart

---

## 🔄 Continuous Updates

### Streamlit Cloud (Auto-updates)
Changes pushed to GitHub automatically deploy:
```bash
git add .
git commit -m "Update fraud patterns"
git push origin main
```

### Docker (Manual updates)
Rebuild and push:
```bash
docker build -t yourname/privacy-fraud-ai:latest .
docker push yourname/privacy-fraud-ai:latest
```

### Local (Testing)
Keep running locally during development:
```bash
streamlit run notebooks/ui.py
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Port already in use" | Change port: `docker run -p 8507:8501 ...` |
| "Module not found" | Update `requirements.txt` and rebuild |
| "Out of memory" | Increase Docker memory allocation |
| "Slow performance" | Optimize GPU usage or upgrade tier |

---

**Recommended: Start with Streamlit Cloud for instant team access!** 🚀
