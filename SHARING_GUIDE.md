# Sharing Guide: Privacy-Fraud-AI with Collaborators

## Option 1: Local Network Access (Immediate - No Setup)

**Currently Available:**
- Local: `http://localhost:8506`
- Network: `http://10.244.61.97:8506`

Collaborators on the same network can access directly using the Network URL.

**Requirements:**
- Same office/VPN network
- No firewall blocking port 8506

---

## Option 2: Public Tunnel with Ngrok (Quick 5-min Setup)

Share via a public URL that anyone can access from anywhere.

### Setup:

1. **Install Ngrok**
```bash
# Download from https://ngrok.com/download
# Or via winget (Windows)
winget install ngrok
```

2. **Create Ngrok Account**
```bash
# Sign up free at https://ngrok.com
# Get your authtoken from dashboard
ngrok config add-authtoken <YOUR_TOKEN>
```

3. **Run Ngrok with Streamlit**

In a new terminal:
```bash
# First, make sure Streamlit is running (should already be)
# Then tunnel it
ngrok http 8506
```

This generates a public URL like:
```
Forwarding  https://abc123def456.ngrok.io -> http://localhost:8506
```

4. **Share the URL**
Send `https://abc123def456.ngrok.io` to collaborators

**Pros:**
- Instant public access
- No deployment needed
- Anyone can use it

**Cons:**
- URL changes each time you restart
- Requires you to keep app running
- Free tier has usage limits

---

## Option 3: Deploy to Streamlit Cloud (Free - 10 min Setup)

**Permanent hosting** with automatic updates from GitHub.

### Setup:

1. **Push to GitHub**
```bash
cd c:\Users\anshu\privacy_fraud_ai
git init
git add .
git commit -m "Initial commit: Privacy Fraud AI System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/privacy_fraud_ai.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
- Go to https://streamlit.io/cloud
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file: `notebooks/ui.py`
- Click "Deploy"

3. **Share the Public URL**
Your app gets a permanent URL like:
```
https://privacy-fraud-ai-yourname.streamlit.app
```

**Pros:**
- Always available
- Free tier (3 apps)
- Auto-updates from GitHub
- Professional URL
- Easy to manage

**Cons:**
- Requires GitHub account
- Uploads code to public repo (use .gitignore for secrets)
- Slight cold start delay

**Important:** Create `.streamlit/secrets.toml` for sensitive data (locally):
```toml
# Don't commit this file!
api_keys = "your_secrets_here"
```

---

## Option 4: Docker Container (For Teams)

**Containerize the app** for reproducible environments.

### Create Dockerfile:

Already created in the project. Run:

```bash
# Build Docker image
docker build -t privacy-fraud-ai:latest .

# Run container
docker run -p 8506:8506 privacy-fraud-ai:latest

# Push to Docker Hub for team
docker tag privacy-fraud-ai:latest YOUR_DOCKERHUB_USERNAME/privacy-fraud-ai:latest
docker push YOUR_DOCKERHUB_USERNAME/privacy-fraud-ai:latest
```

Collaborators can then:
```bash
docker run -p 8506:8506 YOUR_DOCKERHUB_USERNAME/privacy-fraud-ai:latest
```

**Pros:**
- Identical environment for all users
- Works on any OS (Windows, Mac, Linux)
- Easy version management
- Scalable

**Cons:**
- Requires Docker installation
- More setup for users

---

## Option 5: Executable (.EXE) for Windows

Create a standalone Windows executable.

### Setup:

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile ^
  --add-data "notebooks:notebooks" ^
  --add-data "data:data" ^
  --add-data "models:models" ^
  -n privacy_fraud_ai ^
  notebooks/ui.py
```

Find executable in `dist/privacy_fraud_ai.exe`

**Pros:**
- No Python/dependencies needed
- Easy for non-technical users
- Portable

**Cons:**
- Large file (~300-500MB)
- Windows only (similar process for Mac/Linux)
- Harder to update

---

## Recommended Approach for Your Team

### For Internal Team (Same Office):
1. **Immediate**: Share Network URL (no setup)
2. **Better**: Set up Ngrok for demos/temporary access

### For Permanent/Public Access:
1. **Best**: Deploy to Streamlit Cloud
   - Free & easy
   - Professional hosting
   - Auto-updates from GitHub

### For Enterprise/Offline:
1. **Best**: Docker container
   - Reproducible
   - Works offline
   - Scalable

---

## Quick Start: Streamlit Cloud Deployment (Recommended)

**5-minute deployment:**

1. Create/use GitHub account
2. Push code to GitHub:
```bash
git init
git add .
git commit -m "Privacy Fraud AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/privacy_fraud_ai.git
git push -u origin main
```

3. Go to https://streamlit.io/cloud → "New app"
4. Select repo → `notebooks/ui.py` → Deploy
5. Share permanent URL with team

---

## Sharing Checklist

- [ ] Choose sharing method
- [ ] Set up authentication/access control if needed
- [ ] Test with collaborator
- [ ] Document data upload requirements
- [ ] Share README with setup instructions
- [ ] Monitor usage/logs

---

## Security Considerations

⚠️ **For Public/Shared Instances:**

1. **Data Privacy**
   - Users upload real datasets → ensure privacy
   - Consider data retention policy
   - Implement input validation

2. **Authentication** (if needed)
   - Add Streamlit authentication
   - Restrict to authorized users

3. **Rate Limiting**
   - Limit concurrent users
   - Set upload size limits

4. **Audit Logging**
   - Log who accesses what
   - Monitor resource usage

---

## Support

For each deployment method:

| Method | Support | Limits | Cost |
|--------|---------|--------|------|
| Network URL | Team only | Same network | Free |
| Ngrok | Public | 40 req/min | Free (tier) |
| Streamlit Cloud | Public | 1GB RAM, 3 apps | Free |
| Docker | Custom | Your infra | Cost of hosting |
| .EXE | Windows only | Single machine | Free |

---

**Next Step:** Which option would you like to set up?
