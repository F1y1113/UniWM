#!/usr/bin/env bash
# install.sh — One-click environment setup for UniWM.
# Usage: bash install.sh
#
# Run this *inside* an activated conda env (e.g., after `conda activate uniwm`).

set -e

echo "[1/4] Pinning legacy build tooling..."
pip install --upgrade "pip<25" "setuptools<81" wheel

echo "[2/4] Installing build-time dependencies..."
pip install numpy==1.24.3 psutil==6.0.0 setuptools_scm seqeval==1.2.2

echo "[3/4] Installing PyTorch 2.4.0..."
pip install torch==2.4.0

echo "[4/4] Installing remaining requirements (no build isolation)..."
pip install --no-build-isolation -r requirements.txt

echo ""
echo "[DONE] UniWM environment is ready."
