#!/bin/bash
# DESCARTES Neural ODE - Vast.ai Setup & Run Script
# Run this after uploading rung3_data.tar.gz to /workspace/

set -e

echo "============================================"
echo "DESCARTES Neural ODE - Setup"
echo "============================================"

# 1. Extract data
cd /workspace
if [ -f rung3_data.tar.gz ]; then
    echo "[1/4] Extracting data..."
    tar xzf rung3_data.tar.gz
    echo "  Found $(ls /workspace/rung3_data/*.h5 2>/dev/null | wc -l) HDF5 files"
else
    echo "ERROR: /workspace/rung3_data.tar.gz not found!"
    echo "Upload it first: scp -P PORT rung3_data.tar.gz root@HOST:/workspace/"
    exit 1
fi

# 2. Clone/update repo
echo "[2/4] Setting up code..."
if [ -d /workspace/Le-Masson-2002 ]; then
    cd /workspace/Le-Masson-2002
    git pull
else
    cd /workspace
    git clone https://github.com/nimallansa937/Le-Masson-2002.git
    cd /workspace/Le-Masson-2002
fi

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install -q torch torchdiffeq numpy scipy scikit-learn matplotlib h5py

# 4. Verify GPU
echo "[4/4] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "============================================"
echo "Setup complete! Now run experiments:"
echo "============================================"
echo ""
echo "cd /workspace/Le-Masson-2002/descartes_neural_ode"
echo ""
echo "# RUN 1: Full DESCARTES suite (DreamCoder + Balloon + 9 archs)"
echo "python run_descartes_neural_ode.py --data_dir /workspace/rung3_data --device cuda --max_iterations 9 --max_hours_per_model 0.5 --subsample 200 --output descartes_results.json"
echo ""
echo "# RUN 2: Hidden dimension sweep"
echo "python -m experiments.hidden_sweep --data-dir /workspace/rung3_data --device cuda --max-hours 0.3 --max-epochs 100 --output hidden_sweep_results.json --architectures ltc_network gru_ode hybrid_lstm_ode"
echo ""
echo "# RUN 3: GABA interpolation hold-out"
echo "python -m experiments.gaba_interpolation --data-dir /workspace/rung3_data --device cuda --max-hours 0.5 --output gaba_interpolation_results.json --architectures ltc_network gru_ode hybrid_lstm_ode"
echo ""
echo "# RUN 4: Generate figures (after 1-3 complete)"
echo "python -m figures.plot_comparison --results-dir . --output-dir /workspace/figures"
