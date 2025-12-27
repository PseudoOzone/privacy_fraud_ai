# Privacy Fraud AI

A machine learning project for fraud detection in financial data with privacy-preserving techniques and federated learning capabilities. This repository includes datasets, synthetic data generation, model training pipelines, and fraud detection workflows.

**🚀 [Share with Collaborators](SHARING_GUIDE.md) | 📦 [Deployment Guide](DEPLOYMENT.md)**

## Project Overview

This project aims to develop and evaluate fraud detection models while preserving user privacy through:
- **Federated Learning**: Distributed training across multiple data sources without centralizing sensitive data
- **Synthetic Data Generation**: Using generative AI to create privacy-respecting synthetic datasets
- **Multiple Data Sources**: Integration with heterogeneous financial datasets (Bank A, Bank B)

## Project Structure

```
privacy_fraud_ai/
├── data/                          # Raw datasets
│   ├── bankA.csv                  # Bank A transaction data
│   ├── bankB.csv                  # Bank B transaction data
│   └── full_merged.csv            # Combined dataset
├── generated/                      # Synthetic datasets
│   ├── synthetic_genai.csv        # Generated synthetic data (raw)
│   └── synthetic_genai_clean.csv  # Generated synthetic data (processed)
├── models/                         # Trained models (checkpoints, weights)
├── notebooks/                      # Analysis and training notebooks
│   ├── dataset_generator.py       # Synthetic data generation pipeline
│   ├── fraud_gpt.py              # GPT-based fraud detection model
│   ├── fraud_gpt_training.py     # Training script for fraud_gpt
│   ├── federated_training.py     # Federated learning trainer
│   └── run_fraud_summary.py      # Evaluation and reporting
├── fraud_gpt/                      # Fraud GPT module (core library)
├── requirements.txt                # Python dependencies
├── .vscode/                        # VS Code workspace settings
└── README.md                       # This file
```

## Setup & Installation

### Prerequisites

- **Windows 10/11** with NVIDIA GPU (RTX 3050 or equivalent) for CUDA support
- **NVIDIA Driver**: CUDA 12.5+ compatible (check with `nvidia-smi`)
- **Git** (for version control)

### Quick Start

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd privacy_fraud_ai
```

#### 2. Set Up Python Environment

**Option A: Using the pre-configured venv (Python 3.13, CPU)**

```powershell
# Activate the existing venv
.\venv\Scripts\Activate.ps1

# Verify packages
python -m pip list | findstr "torch transformers pandas"
```

**Option B: Using Conda with CUDA Support (Recommended for GPU)**

```powershell
# Create conda env (Python 3.11 + CUDA 12.1)
C:\Users\anshu\miniconda3\_conda.exe create -y -p C:\Users\anshu\miniconda3\envs\torch311 python=3.11

# Install PyTorch with CUDA support
C:\Users\anshu\miniconda3\_conda.exe install -y -p C:\Users\anshu\miniconda3\envs\torch311 -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# Activate the environment
C:\Users\anshu\miniconda3\envs\torch311\Scripts\activate.bat
```

#### 3. Install Dependencies

Once activated, install remaining requirements:

```powershell
pip install -r requirements.txt
```

#### 4. Verify Installation

Check that core packages are available:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import pandas; import transformers; print('All dependencies installed!')"
```

### VS Code Integration

The workspace is pre-configured with `.vscode/settings.json` to auto-activate the `venv` in PowerShell terminals. Simply open a new terminal in VS Code and it will activate automatically.

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1+ | Deep learning framework |
| `transformers` | 4.57.3+ | Hugging Face transformer models |
| `pandas` | 2.3.3+ | Data manipulation & analysis |
| `scikit-learn` | 1.8.0+ | ML models & utilities |
| `huggingface_hub` | Latest | Model & dataset repository |
| `sentencepiece` | 0.2.1+ | Tokenization for transformers |
| `matplotlib` | 3.10.8+ | Visualization |
| `tqdm` | 4.67.1+ | Progress bars |

## Usage

### Running Notebooks

Execute Python notebooks directly:

```powershell
# Activate environment first
C:\Users\anshu\miniconda3\envs\torch311\Scripts\Activate.ps1

# Run fraud_gpt training
python notebooks\fraud_gpt_training.py

# Run federated training
python notebooks\federated_training.py

# Generate datasets
python notebooks\dataset_generator.py
```

### Testing GPU Access

Verify CUDA support:

```powershell
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
"
```

### Checking Dependencies

View installed packages:

```powershell
pip list
```

List all available PyTorch CUDA wheels (for CUDA 12.1):

```powershell
pip index versions torch --extra-index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
privacy_fraud_ai/
├── notebooks/
│   ├── fraud_gpt.py                 # Fraud pattern generator
│   ├── fraud_gpt_training.py        # Training pipeline
│   ├── federated_training.py        # Federated learning
│   ├── dataset_generator.py         # Synthetic data creation
│   └── run_fraud_summary.py         # Analysis & reporting
├── data/
│   ├── bankA.csv                    # Bank A transaction dataset
│   ├── bankB.csv                    # Bank B transaction dataset
│   └── full_merged.csv              # Merged dataset
├── generated/
│   ├── synthetic_genai.csv          # Generated synthetic data
│   ├── synthetic_genai_clean.csv    # Cleaned synthetic data
├── models/                          # Saved model artifacts
├── .vscode/
│   └── settings.json                # VS Code environment config
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── start.bat                        # Windows batch launcher
```

## Dependencies

Core ML/DL Stack:
- `torch==2.9.1+cu121` – PyTorch with CUDA 12.1 support
- `transformers==4.57.3` – HuggingFace transformers for NLP/sequence models
- `pandas==2.3.3` – Data manipulation
- `numpy==2.4.0` – Numerical computation
- `scikit-learn==1.8.0` – Classical ML algorithms
- `matplotlib==3.10.8` – Visualization

Additional Libraries:
- `huggingface_hub==0.36.0` – Model hub integration
- `sentencepiece==0.2.1` – Tokenization for transformer models
- `tqdm` – Progress bars
- `joblib` – Parallel computing

See [requirements.txt](requirements.txt) for complete details.

## Environment Variables

No special environment variables required. GPU support is automatic if:
- NVIDIA drivers are installed (check with `nvidia-smi`)
- PyTorch is built with CUDA support (conda install handles this)
- CUDA 12.1 libs are in conda environment

## Troubleshooting

### GPU Not Detected

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA driver: `nvidia-smi`
2. Ensure PyTorch is CUDA-enabled: `python -c "import torch; print(torch.version.cuda)"`
3. If using venv, reinstall PyTorch with CUDA support from conda
4. Check VS Code is using the correct Python interpreter

### PowerShell Activation Issues

If `Activate.ps1` doesn't run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Memory Errors with Large Models

If training runs out of memory:
- Reduce batch size in training scripts
- Use mixed precision training (`torch.cuda.amp`)
- Consider gradient checkpointing for transformer models

## Development

### Adding New Dependencies

1. Install the package:
   ```powershell
   pip install package_name
   ```
2. Update `requirements.txt`:
   ```powershell
   pip freeze > requirements.txt
   ```
3. Commit both files to version control

### Running Tests

(No formal test suite yet. Add pytest/unittest tests here as needed.)

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contributors

- Project lead: [Your name/organization]

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Federated Learning (TensorFlow Federated)](https://www.tensorflow.org/federated)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [NVIDIA CUDA Support for RTX 3050](https://developer.nvidia.com/cuda-toolkit)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review notebook comments for detailed explanations
3. Open an issue on GitHub with error details and environment info

---

**Last Updated**: December 27, 2025

## GitHub Workflow

### Initial Setup

`ash
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git remote add origin https://github.com/yourusername/privacy-fraud-ai.git
git branch -M main
`

### .gitignore Template

Create .gitignore to exclude large files and sensitive data:

```
# Virtual environments
venv/
miniconda3/
*.egg-info/
__pycache__/
*.pyc
.Python

# Data (sensitive)
data/*.csv
generated/*.csv
models/*.pt
models/*.pth
*.zip

# IDEs & editors
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### Commit & Push Workflow

`ash
# Check status
git status

# Stage changes
git add .

# Or specific files
git add notebooks/fraud_gpt_training.py requirements.txt

# Commit
git commit -m "Implement GPU-accelerated fraud detection"

# Push to GitHub
git push origin main
`

### Feature Branches

`ash
# Create feature branch
git checkout -b feature/gpu-optimization

# Make changes and commit
git add .
git commit -m "Optimize GPU memory usage"

# Push and create Pull Request
git push origin feature/gpu-optimization
`

## Troubleshooting

### CUDA Not Available

**Problem**: 	orch.cuda.is_available() returns False

**Solution**:
1. Verify NVIDIA driver:
   `powershell
   nvidia-smi
   `
2. Use conda PyTorch with bundled CUDA libraries (recommended)
3. Or install CUDA 12.1 Toolkit: https://developer.nvidia.com/cuda-12-1-0-download-archive

### Import Errors

**Problem**: ModuleNotFoundError: No module named 'transformers'

**Solution**:
`powershell
pip install --upgrade transformers huggingface_hub
`

### Out of Memory (OOM)

**Problem**: CUDA out of memory during training

**Solutions**:
- Reduce batch size in training script
- Use CPU device: device = torch.device("cpu")
- Clear cache: 	orch.cuda.empty_cache()

### Conda Install Slow/Fails

**Problem**: Network timeouts during conda installs

**Solution** - Retry with longer timeout:
`powershell
C:\Users\anshu\miniconda3\_conda.exe install -y -p C:\Users\anshu\miniconda3\envs\torch311 
  -c pytorch -c nvidia pytorch pytorch-cuda=12.1 --timeout=600
`

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Federated Learning Frameworks](https://flower.dev/)
- [CUDA Development Tools](https://developer.nvidia.com/cuda-toolkit)

## License

MIT License � See LICENSE file for details

## Authors

- Development Team

---

**Last Updated**: December 27, 2025
