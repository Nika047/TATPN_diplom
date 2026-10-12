"""
models/ — Архитектурные компоненты TATPN.

Подмодули:
  - cnn.py        : Multi-Scale CNN (Conv1d 3/7/15, без MaxPool)
  - mamba.py      : Custom Mamba (Selective State Space Model) на чистом PyTorch
  - pooling.py    : Adaptive Context Pooling (attention-взвешенная сумма)
  - attention.py  : Cross-Attention Target Head
  - decoder.py    : Авторегрессивный GRU-декодер траектории
  - heads.py      : MLP-головы (target classification + ETA regression)
  - tatpn.py      : Главный класс TATPN, собирающий всё вместе

Использование:
    from models import TATPN
    model = TATPN()
    out = model(observed, mask, target_embeddings)
"""

from .tatpn import TATPN

__all__ = ["TATPN"]
