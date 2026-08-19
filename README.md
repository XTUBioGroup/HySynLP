# HySynLP: Synergistic Drug Combination  

---

## 1. Environment Setup

```bash
# Recommended: Python 3.10, PyTorch 1.11.0 with CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install PyTorch Geometric and RDKit
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp310-cp310-win_amd64.whl
pip install torch-geometric==2.0.4 rdkit
```

---

## 2. Run Experiment

python main_pretrained.py

