"""
tests/test_model_shapes.py — Smoke-тест архитектуры TATPN.

Не требует обработанных данных — генерирует синтетический батч с правильными
размерностями и проверяет:
  1) Что forward работает и возвращает все ожидаемые тензоры.
  2) Что размерности выходов совпадают с описанием в config.
  3) Что backward проходит и градиенты текут во все параметры.
  4) Что отдельные компоненты (CNN, Mamba, Pooling, Cross-Attention,
     Decoder, ETA) тоже выдают корректные размеры.

Запуск:
    python -m tests.test_model_shapes
или
    python tests/test_model_shapes.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import config
from models import TATPN
from models.cnn import MultiScaleCNN
from models.mamba import MambaEncoder
from models.pooling import AdaptiveContextPooling
from models.attention import CrossAttentionTargetHead, TargetEmbedding
from models.decoder import TrajectoryDecoder
from models.heads import ETAHead
from losses import TATPNLoss
from metrics import compute_all_metrics


def _make_batch(B=4, L_real=128, L_max=None, H=None, F=None, T=None, device="cpu"):
    """
    Создаёт синтетический батч в формате DataLoader.
    """
    L_max = L_max or config.MAX_SEQ_LEN
    H = H or config.PRED_HORIZON
    F = F or config.INPUT_FEATURES
    T = T or config.NUM_TARGETS

    observed = torch.randn(B, L_max, F, device=device)
    # реальные L_real шагов, далее padding
    mask = torch.zeros(B, L_max, device=device)
    mask[:, :L_real] = 1.0
    # обнуляем padding в observed
    observed = observed * mask.unsqueeze(-1)

    future_points = torch.randn(B, H, 3, device=device) * 100.0
    future_mask   = torch.ones(B, H, device=device)
    target_id     = torch.randint(0, T, (B,), device=device)
    eta           = torch.rand(B, device=device) * 100.0

    return {
        "observed":      observed,
        "mask":          mask,
        "future_points": future_points,
        "future_mask":   future_mask,
        "target_id":     target_id,
        "eta":           eta,
    }


def test_cnn():
    print("\n[1] MultiScaleCNN")
    cnn = MultiScaleCNN()
    B, L, F = 2, 64, config.INPUT_FEATURES
    x = torch.randn(B, L, F)
    mask = torch.ones(B, L)
    out = cnn(x, mask)
    assert out.shape == (B, L, config.MAMBA_D_MODEL), \
        f"CNN: {out.shape} != {(B, L, config.MAMBA_D_MODEL)}"
    print(f"    OK  in=({B},{L},{F})  out={tuple(out.shape)}")


def test_mamba():
    print("\n[2] MambaEncoder")
    enc = MambaEncoder()
    B, L, D = 2, 32, config.MAMBA_D_MODEL  # короткая L, чтобы было быстро
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    out = enc(x, mask)
    assert out.shape == (B, L, D), f"Mamba: {out.shape} != {(B, L, D)}"
    print(f"    OK  in={tuple(x.shape)}  out={tuple(out.shape)}")


def test_pooling():
    print("\n[3] AdaptiveContextPooling")
    pool = AdaptiveContextPooling()
    B, L, D = 3, 200, config.MAMBA_D_MODEL
    h = torch.randn(B, L, D)
    mask = torch.zeros(B, L)
    # разные длины
    mask[0, :180] = 1
    mask[1, :100] = 1
    mask[2, :10]  = 1  # короткая
    ctx = pool(h, mask)
    assert ctx.shape == (B, D), f"Pool: {ctx.shape} != {(B, D)}"
    assert torch.isfinite(ctx).all(), "Pool: NaN/Inf в выходе"
    print(f"    OK  in={tuple(h.shape)}  out={tuple(ctx.shape)}")


def test_cross_attention():
    print("\n[4] CrossAttentionTargetHead + TargetEmbedding")
    tgt_emb = TargetEmbedding()
    head = CrossAttentionTargetHead()
    B, D, T = 4, config.MAMBA_D_MODEL, config.NUM_TARGETS

    types = torch.randint(0, config.NUM_TARGET_TYPES, (T,))
    coords = torch.randn(T, 3)
    embs = tgt_emb(types, coords)
    assert embs.shape == (T, config.TARGET_EMBED_DIM), embs.shape

    context = torch.randn(B, D)
    tgt_aware, logits = head(context, embs)
    assert tgt_aware.shape == (B, D), tgt_aware.shape
    assert logits.shape == (B, T), logits.shape
    print(f"    OK  target_aware={tuple(tgt_aware.shape)}  logits={tuple(logits.shape)}")


def test_decoder():
    print("\n[5] TrajectoryDecoder")
    dec = TrajectoryDecoder()
    B, D, H = 2, config.MAMBA_D_MODEL, config.PRED_HORIZON
    context = torch.randn(B, D)
    tgt_aware = torch.randn(B, D)
    last_point = torch.randn(B, 3) * 100
    future = torch.randn(B, H, 3) * 100
    dec.train()
    out = dec(context, tgt_aware, last_point, future_truth=future, teacher_forcing_ratio=0.5)
    assert out.shape == (B, H, 3), out.shape
    print(f"    OK  out={tuple(out.shape)}")


def test_eta_head():
    print("\n[6] ETAHead")
    head = ETAHead(in_dim=2 * config.MAMBA_D_MODEL)
    B = 5
    x = torch.randn(B, 2 * config.MAMBA_D_MODEL)
    out = head(x)
    assert out.shape == (B,), out.shape
    assert (out >= 0).all(), "ETA должен быть неотрицательным"
    print(f"    OK  out={tuple(out.shape)}, min={out.min():.3f}")


def test_full_forward_backward():
    print("\n[7] TATPN полный forward + backward")
    # Используем короткую L_max для скорости smoke-теста
    L_short = 64
    H_short = 16
    # Подменим временно config (через локальные параметры конструктора)

    model = TATPN(pred_horizon=H_short)
    model.train()

    B = 2
    F_in = config.INPUT_FEATURES
    T = config.NUM_TARGETS

    observed = torch.randn(B, L_short, F_in)
    mask = torch.ones(B, L_short)
    last_point = torch.randn(B, 3) * 100
    future = torch.randn(B, H_short, 3) * 100

    out = model(
        observed=observed,
        mask=mask,
        last_point=last_point,
        future_points=future,
        teacher_forcing_ratio=0.5,
    )

    assert out["pred_future"].shape == (B, H_short, 3)
    assert out["target_logits"].shape == (B, T)
    assert out["eta"].shape == (B,)
    print(f"    forward OK  pred_future={tuple(out['pred_future'].shape)}  "
          f"target_logits={tuple(out['target_logits'].shape)}  "
          f"eta={tuple(out['eta'].shape)}")

    # === backward через loss ===
    criterion = TATPNLoss()
    batch = {
        "observed":      observed,
        "mask":          mask,
        "future_points": future,
        "future_mask":   torch.ones(B, H_short),
        "target_id":     torch.randint(0, T, (B,)),
        "eta":           torch.rand(B) * 50,
    }
    loss, loss_dict = criterion(out, batch)
    print(f"    loss={loss.item():.4f}  "
          f"(traj={loss_dict['loss_traj']:.3f}, "
          f"target={loss_dict['loss_target']:.3f}, "
          f"eta={loss_dict['loss_eta']:.3f})")

    loss.backward()

    # Проверяем что градиенты протекли в большинстве параметров
    n_total = 0
    n_with_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            n_with_grad += 1
    coverage = n_with_grad / max(n_total, 1)
    print(f"    backward OK  параметров с градиентом: {n_with_grad}/{n_total} "
          f"({coverage:.1%})")
    assert coverage > 0.8, f"Только {coverage:.1%} параметров получили градиент"

    # === Метрики ===
    metrics = compute_all_metrics(out, batch)
    print(f"    metrics: ADE={metrics['ADE']:.2f}  FDE={metrics['FDE']:.2f}  "
          f"Acc@1={metrics['Acc@1']:.3f}  MAE_ETA={metrics['MAE_ETA']:.2f}")


def main():
    print("=" * 70)
    print("  TATPN — smoke-тест архитектуры")
    print("=" * 70)
    print(f"  Device: {config.DEVICE}")
    print(f"  INPUT_FEATURES={config.INPUT_FEATURES}, "
          f"MAX_SEQ_LEN={config.MAX_SEQ_LEN}, "
          f"PRED_HORIZON={config.PRED_HORIZON}")
    print(f"  NUM_TARGETS={config.NUM_TARGETS}")

    torch.manual_seed(0)
    np.random.seed(0)

    test_cnn()
    test_mamba()
    test_pooling()
    test_cross_attention()
    test_decoder()
    test_eta_head()
    test_full_forward_backward()

    print("\n" + "=" * 70)
    print("  Все тесты пройдены ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
