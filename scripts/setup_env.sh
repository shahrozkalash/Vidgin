#!/usr/bin/env bash
# Quick environment bootstrapper for Vidgin
set -euo pipefail

echo "==> Checking Python..."
python3 --version

if ! command -v pip3 >/dev/null 2>&1; then
    echo "==> Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages
fi

echo "==> Installing requirements..."
pip3 install --break-system-packages -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> Verifying core dependencies..."
python3 -c "import torch, cv2, PIL, scipy, soundfile, google.genai; print('All core modules verified successfully!')"
echo "==> Done!"
