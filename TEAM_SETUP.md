# 👥 Team Setup Guide: For Collaborators

## 🎯 Fastest Way: Use the Hosted Version

**Ask your project lead for the Streamlit Cloud URL:**
```
https://privacy-fraud-ai-yourname.streamlit.app
```

No installation needed! Just:
1. Click the link
2. Upload your Bank A & Bank B CSV files
3. Configure settings
4. Click "Run Full Pipeline"
5. Get fraud analysis + attack patterns

---

## 💻 Local Development Setup (5 minutes)

### Windows

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_ORG/privacy_fraud_ai.git
cd privacy_fraud_ai
```

2. **Run setup script:**
```bash
setup.bat
```

This automatically:
- Creates Python virtual environment
- Installs all dependencies
- Creates data/models directories

3. **Start the app:**
```bash
streamlit run notebooks/ui.py
```

4. **Open in browser:**
```
http://localhost:8501
```

---

### Mac/Linux

```bash
# Clone
git clone https://github.com/YOUR_ORG/privacy_fraud_ai.git
cd privacy_fraud_ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run notebooks/ui.py
```

---

## 🐳 Using Docker (Easiest for Teams)

### Prerequisites
- Install Docker: https://www.docker.com/products/docker-desktop

### Run:
```bash
docker run -p 8501:8501 YOUR_DOCKER_USERNAME/privacy-fraud-ai:latest
```

Then open: http://localhost:8501

No Python, no dependencies, works on Windows/Mac/Linux!

---

## 📊 Using the Pipeline

### Step 1: Prepare Your Data
CSV files with transaction data:
- **Required columns**: amount, merchant_category, device_type, is_fraud (optional)
- **No PII allowed**: Remove name, email, phone, SSN before uploading

Example:
```
transaction_id, amount, merchant_category, timestamp, device_type, is_fraud
1, 150.50, Grocery, 2025-12-27 10:30:00, Mobile, 0
2, 1500.00, Electronics, 2025-12-27 10:35:00, Unknown, 1
...
```

### Step 2: Upload & Run Pipeline
1. Go to **🔗 End-to-End Pipeline** tab
2. Upload Bank A CSV
3. Upload Bank B CSV
4. Configure DP noise level (privacy vs accuracy)
5. Select number of attack scenarios
6. Click **▶️ Run Full Pipeline**

### Step 3: Get Results
The pipeline automatically:
- ✅ Removes PII
- ✅ Trains federated models
- ✅ Applies differential privacy
- ✅ Generates fraud summary
- ✅ Creates attack simulations

All in one place!

---

## 📁 Project Files

Key files you'll interact with:

```
notebooks/ui.py              # Main Streamlit app
notebooks/dataset_generator.py    # Generate synthetic data
notebooks/federated_training.py   # Federated ML pipeline
notebooks/fraud_gpt.py            # Fraud analysis module
data/                             # Your CSV files go here
models/                           # Trained models stored here
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Port 8501 already in use"
```bash
streamlit run notebooks/ui.py --server.port 8502
```

### "CUDA out of memory"
App uses CPU by default. To use GPU:
```bash
# GPU support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Slow performance"
- Reduce dataset size (<10k rows per bank)
- Reduce attack scenario count
- Use Streamlit Cloud (hosted on server)

---

## 📞 Getting Help

1. **Check documentation:**
   - [SHARING_GUIDE.md](SHARING_GUIDE.md) - Sharing options
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment methods
   - [QUICKSTART.md](QUICKSTART.md) - Quick usage

2. **Common issues:** See Troubleshooting above

3. **Ask team lead** for:
   - Data format questions
   - API keys/secrets
   - Access to hosted version

---

## 🎓 Learning Resources

### Understanding the Pipeline

1. **Federated Learning**: Train models separately, aggregate globally
2. **Differential Privacy**: Add noise to protect individual records
3. **Fraud-GPT**: Analyze patterns and generate attack scenarios

### Code Overview

- `federated_training.py` - Core ML logic
- `fraud_gpt.py` - LLM-based analysis
- `ui.py` - Streamlit interface

### Data Flow

```
Your CSVs
   ↓
PII Removal & Cleaning
   ↓
Federated Training (Local + Global)
   ↓
Differential Privacy Noise
   ↓
Fraud-GPT Analysis
   ↓
Summary + Attack Patterns
```

---

## ✅ Checklist: Ready to Use

- [ ] Cloned repository or have hosted URL
- [ ] Python 3.13+ installed (if local)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] CSV data prepared (no PII)
- [ ] Ran Streamlit app successfully
- [ ] Uploaded test data
- [ ] Got fraud analysis results

---

## 🚀 Next Steps

1. **Immediate:** Use hosted version to test
2. **Short-term:** Set up local development
3. **Medium-term:** Contribute improvements
4. **Long-term:** Train with real data

---

**Questions?** Ask your team lead or check [SHARING_GUIDE.md](SHARING_GUIDE.md)

Happy fraud hunting! 🔍

---

**Last Updated:** December 27, 2025
**Version:** 1.0
