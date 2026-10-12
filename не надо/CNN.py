import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ConvTrajectoryNet(nn.Module):
    def __init__(self, pred_horizon=20):
        super().__init__()
        self.pred_horizon = pred_horizon

        # Блок сверток по времени
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Глобальный пуллинг по времени
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)

        # Полносвязный блок для прогноза
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, pred_horizon * 3)  # 3 координаты на каждый шаг
        )

    def forward(self, x):
        # x: (B, T, 4) → (B, 4, T)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)  # (B, C, T)
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        x = self.fc(x)  # (B, pred_horizon*3)
        x = x.view(-1, self.pred_horizon, 3)  # (B, pred_horizon, 3)
        return x

from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for inp, tgt in loop:
            inp, tgt = inp.to(device), tgt.to(device)

            pred = model(inp)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=np.mean(train_losses))

        avg_train_loss = np.mean(train_losses)

        # Валидация
        val_loss = evaluate_model(model, val_loader, device, criterion)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_conv_model.pt")
            print(f"✅ Лучшая модель сохранена (Val Loss: {val_loss:.6f})")

@torch.no_grad()
def evaluate_model(model, loader, device, criterion):
    model.eval()
    losses = []

    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        pred = model(inp)
        loss = criterion(pred, tgt)
        losses.append(loss.item())

    return np.mean(losses)

