# 🏦 Privacy-Fraud-AI — Federated + Gen-AI Fraud Detection Suite

This repository contains a research-grade project that demonstrates:

| Module | Description |
|--------|-------------|
| 🔐 Federated ML | Train fraud-detection models across banks without sharing raw data |
| 🔏 Differential Privacy | Wrap models so shared output cannot leak sensitive information |
| 🤖 Gen-AI Fraud Agent | LLM that reads aggregated stats → outputs fraud summaries and synthetic attack patterns |
| 🧪 Synthetic Data Generator | Automates new fraud-like events to expand model training |

---

## 🎯 Goal of the Project

Real-world banks cannot share user data (PII).  
This project showcases a **privacy-preserving fraud intelligence system** that works **without exposing names, emails, phone numbers or accounts.**

It includes:

| Feature | Status |
|--------|--------|
| Federated Learning across institutions | ✅ |
| DP-safe prediction wrapper | ✅ |
| Synthetic Fraud Pattern Generator (Gen-AI) | ✅ |
| Aggregated-only LLM summaries | 🚧 improving |
| Real-time anomaly scoring | 🔜 upcoming |
| Dashboard & API endpoint | 🔜 upcoming |

---

## 🏗️ Architecture

```
Bank A CSV ──┐       ┌─ Aggregated statistics ──► Gen-AI Summary
              │       │
              ▼       ▼
Local RandomForest  Local RandomForest
       │                   │
       └──── Fed-Merge → Global Model (No raw data shared)
                │
                ├── DP Wrapper – noise added
                │
                ├── Save → models/global_model_dp.pkl
                │
                └── Predict on new CSV safely
```

---

## 📌 How to Use

### 1️⃣ Run in Google Colab
```bash
git clone https://github.com/<yourname>/privacy_fraud_ai.git
cd privacy_fraud_ai
```
Open `notebooks/federated_training.ipynb` → Run all  
Outputs:
- `models/global_model.pkl`
- `models/global_model_dp.pkl`

### 2️⃣ Generate Synthetic Fraud Records (LLM-based)
```python
!python src/gen_fraud_ai.py
```
Outputs:
```
/results/synthetic_outputs.csv
```

---

## 🔐 Differential Privacy

Instead of modifying internal RandomForest weights (unsafe),  
we apply **output-side noise**:

```python
pred = real_pred + Normal(0, sigma)
pred = clip(pred,0,1)
```

This ensures:
- bank-to-bank model sharing is safe
- attacker cannot infer original customer values

---

## 📊 Example Synthetic Patterns

```
1) POS: 12 transactions in 24 seconds
2) Mobile: 7 failed attempts then 2 success
3) ATM: 8 withdrawals escalating from 83 to 313
4) Web: 14 retries until 5 approval
```

---

## 🧪 Dataset Disclaimer

All CSVs in `/data` are either:
✔ fake  
✔ synthetic  
✔ or anonymized for research  

No real PII exists.

---

## 🧠 Research Extensions You Can Add

| Idea | Type |
|------|------|
| Adaptive meta-learning fraud model | Research publication |
| New-device fingerprint signal embedding | Patent-potential |
| Real-time feature drift detection | Enterprise deployment |
| Blockchain-signed federated updates | Security research |

---

## 📜 License
MIT License — free to use for academic + portfolio work.

---

## 🧑‍💼 Author
Built by **Anshuman Bakshi** — AI researcher 🌙
