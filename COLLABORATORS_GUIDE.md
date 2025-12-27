# 👥 Collaborators: Step-by-Step Getting Started Guide

**Copy-paste everything below to get started!**

---

## 🎯 Option 1: Use the Hosted App (Easiest - No Installation)

**This is the fastest way. No coding knowledge needed.**

### Steps:

1. **Get the URL from your project lead:**
   ```
   https://privacy-fraud-ai-pseudoozone.streamlit.app
   ```

2. **Open it in your browser** → Click the link

3. **Go to the 🔗 "End-to-End Pipeline" tab**

4. **Upload your datasets:**
   - Click "Bank A Data" → Select your CSV file
   - Click "Bank B Data" → Select your CSV file
   - Wait for both to load (shows "✅ Both datasets loaded")

5. **Configure settings:**
   - **DP Noise Sigma**: Set to 0.5 (default is good)
   - **Attack Scenarios**: Set to 3-5
   - Leave other settings as default

6. **Click "▶️ Run Full Pipeline"**

7. **Wait 30-60 seconds** for results to appear

8. **See your results:**
   - PII Removal Summary
   - Fraud Statistics
   - Federated Learning models trained
   - **Fraud-GPT Analysis** (Summary + Attack Patterns combined)
   - Final report with metrics

9. **That's it!** 🎉

### What Your Data Should Look Like:

```
transaction_id,amount,merchant_category,timestamp,device_type,is_fraud
1,150.50,Grocery,2025-12-27 10:30:00,Mobile,0
2,1500.00,Electronics,2025-12-27 10:35:00,Unknown,1
3,50.25,Gas Station,2025-12-27 10:40:00,Desktop,0
...
```

**Required columns:** `amount`, `merchant_category`, `device_type`, `is_fraud` (optional)

---

## 💻 Option 2: Run Locally (If You Want to Develop/Contribute)

### A. Windows Users

**Total time: 5 minutes**

#### Step 1: Install Git
```bash
# Download and install from: https://git-scm.com/download/win
# Accept all defaults
```

#### Step 2: Clone the Repository
```bash
# Open PowerShell and run:
git clone https://github.com/PseudoOzone/privacy_fraud_ai.git
cd privacy_fraud_ai
```

#### Step 3: Run Setup Script
```bash
# Double-click on setup.bat
# OR run in PowerShell:
.\setup.bat
```

This automatically:
- Creates Python virtual environment
- Installs all dependencies
- Creates necessary folders

#### Step 4: Start the App
```bash
streamlit run notebooks/ui.py
```

#### Step 5: Open in Browser
```
http://localhost:8501
```

Then use exactly like the hosted version (see Option 1 above).

---

### B. Mac/Linux Users

**Total time: 5 minutes**

```bash
# Clone repository
git clone https://github.com/PseudoOzone/privacy_fraud_ai.git
cd privacy_fraud_ai

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run notebooks/ui.py
```

Then open: `http://localhost:8501`

---

### C. VS Code (Recommended for Development)

If you use VS Code:

```bash
# Clone and open in VS Code
git clone https://github.com/PseudoOzone/privacy_fraud_ai.git
cd privacy_fraud_ai
code .

# In VS Code terminal:
streamlit run notebooks/ui.py
```

Virtual environment auto-activates! ✅

---

## 🔄 Contributing Code (For Developers)

### Step 1: Clone & Setup
```bash
git clone https://github.com/PseudoOzone/privacy_fraud_ai.git
cd privacy_fraud_ai
./setup.bat  # Windows OR source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Step 2: Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# Example: git checkout -b feature/improve-fraud-detection
```

### Step 3: Make Changes
Edit files in `notebooks/` or wherever needed

### Step 4: Test Locally
```bash
streamlit run notebooks/ui.py
# Test your changes in the app
```

### Step 5: Commit Changes
```bash
git add .
git commit -m "Description of your changes"
# Example: git commit -m "Add improved fraud pattern detection"
```

### Step 6: Push & Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then:
1. Go to GitHub: https://github.com/PseudoOzone/privacy_fraud_ai
2. Click "Pull Requests"
3. Click "New Pull Request"
4. Select your branch
5. Add description
6. Click "Create Pull Request"

**Project lead will review and merge!**

---

## 📊 Using the Different Tabs

### Tab 1: 🔗 End-to-End Pipeline (RECOMMENDED)
- Upload Bank A & Bank B CSVs
- One-click to run everything
- Get fraud summary + attack patterns
- **Best for non-technical users**

### Tab 2: 🏋️ Generate Data
- Create synthetic test datasets
- Useful for testing without real data

### Tab 3: 🤝 Federated Training
- Train models separately
- Advanced users only

### Tab 4: 🔐 DP Model Test
- Compare regular vs privacy-protected predictions
- See differential privacy in action

### Tab 5: 📊 Fraud Summary
- Analyze specific fraud stats
- Advanced fraud analysis

### Tab 6: ⚔️ Attack Simulation
- Generate attack scenarios
- Red-team testing

---

## 🐛 Troubleshooting

### "Port 8501 already in use"
```bash
# Use different port:
streamlit run notebooks/ui.py --server.port 8502
```

### "ModuleNotFoundError"
```bash
# Reinstall requirements:
pip install -r requirements.txt
```

### "Streamlit not found"
```bash
# Make sure virtual environment is activated:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Then install:
pip install streamlit
```

### App running slow
- Reduce dataset size (<5k rows)
- Reduce attack scenario count
- Use the hosted version instead (faster server)

### Data upload fails
- Check CSV format (see "What Your Data Should Look Like" above)
- Max file size: 100MB
- Make sure columns exist

---

## 📁 Project Structure (What Files Do What)

```
📁 notebooks/
  ├── ui.py                    ← Main Streamlit app (what you see)
  ├── dataset_generator.py     ← Creates synthetic data
  ├── federated_training.py    ← Trains models across banks
  ├── fraud_gpt.py            ← Fraud analysis with AI
  └── fraud_gpt_training.py   ← Fine-tunes fraud model

📁 data/
  ├── bankA.csv               ← Bank A transactions
  ├── bankB.csv               ← Bank B transactions
  └── full_merged.csv         ← Combined data

📁 models/
  ├── global_model.pkl        ← Trained federated model
  └── global_model_dp.pkl     ← Privacy-protected version

📄 TEAM_SETUP.md             ← Detailed setup guide
📄 DEPLOYMENT.md             ← How to deploy
📄 requirements.txt          ← Python dependencies
```

---

## 🚀 Common Workflows

### Workflow 1: Analyze Your Company Data
```
1. Download the app (use hosted version)
2. Export your transaction data to CSVs
3. Remove any PII (names, SSNs, emails)
4. Upload both banks' data
5. Run pipeline
6. Get fraud summary & attacks
7. Share results with team
```

### Workflow 2: Test New Fraud Patterns
```
1. Run "Generate Data" tab
2. Create synthetic datasets with new patterns
3. Run "Federated Training" tab
4. Check "DP Model Test" tab
5. Compare results
6. If good, commit to GitHub
```

### Workflow 3: Improve the Code
```
1. Clone repo locally
2. Create feature branch
3. Edit notebooks/fraud_gpt.py or other files
4. Test in Streamlit
5. Push to GitHub
6. Create pull request
7. Team reviews & merges
```

---

## 🔐 Security Notes

✅ **Built-in protections:**
- PII automatically removed
- Data not stored permanently
- Differential privacy applied
- No logs of sensitive data

⚠️ **Best practices:**
- Don't upload real SSNs/credit cards
- Use test data when possible
- Check data columns before upload
- Share results with authorized people only

---

## 📞 Need Help?

### Common Questions:

**Q: Can I use real customer data?**
A: Yes, but remove PII first. Pipeline does it automatically.

**Q: Does the app store my data?**
A: No. Data is processed and discarded. Check data folder to verify.

**Q: Can I modify the code?**
A: Yes! Clone it, make changes, and submit pull requests.

**Q: How do I report bugs?**
A: Create an issue on GitHub or tell the project lead.

**Q: How do I suggest features?**
A: Create an issue on GitHub with "Feature request:" in the title.

---

## ✨ Quick Start Summary

### For Everyone:
1. Go to: https://privacy-fraud-ai-pseudoozone.streamlit.app
2. Upload your CSVs
3. Click "Run Pipeline"
4. Get results!

### For Developers:
1. `git clone https://github.com/PseudoOzone/privacy_fraud_ai.git`
2. `pip install -r requirements.txt`
3. `streamlit run notebooks/ui.py`
4. Make changes
5. Submit pull request

---

## 🎓 Learning Resources

**Understanding the pipeline:**
- Federated Learning: Train separately, aggregate globally
- Differential Privacy: Add noise to protect individuals
- Fraud-GPT: AI analysis of fraud patterns

**Code files:**
- `ui.py` = What users see (Streamlit interface)
- `federated_training.py` = Core ML logic
- `fraud_gpt.py` = AI fraud analysis

**Documentation:**
- README.md = Project overview
- TEAM_SETUP.md = This guide
- DEPLOYMENT.md = Deployment options

---

## ✅ Checklist: Ready to Go

- [ ] Got the app URL (https://privacy-fraud-ai-pseudoozone.streamlit.app)
- [ ] Opened it in browser
- [ ] Have sample CSV data ready
- [ ] Understand the 3 options (Hosted, Local, Development)
- [ ] Can create CSVs with required columns
- [ ] Know how to report issues

**Done? You're ready to start!** 🚀

---

## 🎉 You're All Set!

**Start with the hosted version** → Easiest way to get results

**Questions?** Ask your project lead (Anshu)

**Found a bug?** Create an issue on GitHub

**Want to help?** Fork the repo and submit a pull request!

---

**Last Updated:** December 27, 2025
**App Status:** ✅ Live & Ready
**Version:** 1.0
