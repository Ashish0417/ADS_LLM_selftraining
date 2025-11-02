#!/usr/bin/env bash
# ADS Framework - CPU-Only Setup Script for Linux

set -e

echo "=========================================="
echo "ADS Framework - CPU Environment Setup"
echo "=========================================="
echo ""

# Create project structure
mkdir -p data/wikipedia_cache data/magpie_subset data/benchmarks cache cache/embeddings models results/logs

echo "[1/5] Creating Python virtual environment..."
python3 -m venv ads_env
source ads_env/bin/activate

echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[3/5] Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "[4/5] Installing NLP dependencies..."
pip install \
    transformers>=4.36.0 \
    datasets>=2.14.0 \
    tokenizers>=0.14.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    rank-bm25>=0.2.2 \
    sentence-transformers>=2.2.0 \
    peft>=0.7.0 \
    accelerate>=0.24.0

echo "[5/5] Installing optional dependencies..."
pip install \
    tqdm>=4.66.0 \
    pyyaml>=6.0 \
    tensorboard>=2.14.0 \
    wandb>=0.15.0 \
    requests>=2.31.0

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source ads_env/bin/activate"
echo ""
echo "To start training, run:"
echo "  python main.py"
echo ""
