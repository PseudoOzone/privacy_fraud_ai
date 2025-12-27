# Quick Start Guide

## 1. Activate Environment
```bash
cd c:\Users\anshu\privacy_fraud_ai
venv\Scripts\activate
```

## 2. Run Full Pipeline
```bash
# Generate synthetic datasets
python notebooks/dataset_generator.py

# Run federated learning + differential privacy
python notebooks/federated_training.py

# Output: 
#   - data/bankA.csv, bankB.csv, full_merged.csv, fraud_patterns.txt
#   - models/global_model.pkl, global_model_dp.pkl
```

## 3. Launch Interactive Dashboard
```bash
streamlit run notebooks/ui.py
```

Opens at `http://localhost:8501/` with 5 tabs:
- **Generate Data**: Create synthetic datasets
- **Federated Training**: Train models across banks
- **DP Model Test**: Test differential privacy predictions
- **Fraud Summary**: Generate fraud scenario summaries
- **Attack Simulation**: Simulate attack patterns

## 4. Use Trained Models

### Load Global Model
```python
import pickle
from sklearn.ensemble import RandomForestClassifier

model = pickle.load(open('models/global_model.pkl', 'rb'))
predictions = model.predict(new_data)
```

### Load DP-Protected Model
```python
dp_model = pickle.load(open('models/global_model_dp.pkl', 'rb'))
# Predictions include Laplace noise for privacy
dp_predictions = dp_model.predict(new_data)
```

## 5. Dataset Info

### bankA.csv (3% fraud rate)
- 10,000 transactions
- 300 fraud cases
- Features: transaction_id, amount, merchant_category, timestamp, device_type, location_risk, is_fraud

### bankB.csv (7% fraud rate)
- 10,000 transactions
- 700 fraud cases
- Same features as bankA

### full_merged.csv
- 20,000 total transactions
- 1,000 fraud cases
- Combined from both banks

## 6. Key Modules

| Module | Purpose |
|--------|---------|
| `dataset_generator.py` | Create synthetic banking datasets |
| `federated_training.py` | Train local models, federate, apply DP |
| `fraud_gpt_training.py` | Fine-tune gpt2 on fraud patterns (LoRA) |
| `fraud_gpt.py` | Generate fraud summaries & attack scenarios |
| `ui.py` | Streamlit interactive dashboard |

## 7. Architecture

```
Multiple Banks → Local Training → Global Federation → Differential Privacy
     ↓                  ↓                ↓                    ↓
  bankA.csv      RandomForest      Global RF          DP-wrapped RF
  bankB.csv      (per bank)        (aggregated)       (noised output)
```

## 8. Troubleshooting

**Issue**: Module not found
```bash
# Ensure you're in the correct directory
cd c:\Users\anshu\privacy_fraud_ai
```

**Issue**: Streamlit command not found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Issue**: Models file not found
```bash
# Run federated training first
python notebooks/federated_training.py
```

## 9. Performance Metrics (Current)

- **Global Model Accuracy**: 100% (on synthetic data)
- **Training Time**: ~5 seconds (10k rows per bank)
- **Model Size**: ~50 MB per pickle file
- **Privacy Budget**: epsilon-DP with sigma=0.5

---

**Ready to use!** Start with: `streamlit run notebooks/ui.py`
