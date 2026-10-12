import torch
from torch.utils.data import Dataset, TensorDataset

import os
import globals
import generate_syntetic_data as FakeData
import generate_data as RealData

def dataset_to_tensors(ds: Dataset):
    inputs, targets = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        inputs.append(x)
        targets.append(y)
    X = torch.stack(inputs, dim=0)  # (N, T, 4)
    Y = torch.stack(targets, dim=0)  # (N, H, 3)
    return X, Y


def save_dataset(ds: Dataset, path: str, meta: dict = None):
    X, Y = dataset_to_tensors(ds)
    payload = {
        "inp": X, "tgt": Y,
        "meta": meta or {}
    }
    torch.save(payload, path)


def load_dataset(path: str) -> TensorDataset:
    obj = torch.load(path, map_location="cpu")
    return TensorDataset(obj["inp"], obj["tgt"])


def generate_data(real_data = True):
    if real_data:
        # RealData.create_datasets_with_generated_data()
        train_ds, val_ds, test_ds = RealData.load_datasets_from_folders()

    else:
        train_ds, val_ds, test_ds = FakeData.create_datasets()

    save_dataset(train_ds, os.path.join(globals.save_dir, "train.pt"), meta={"seq_len": globals.seq_len, "pred_horizon": globals.pred_horizon, "dt": globals.dt, "split": "train"})
    save_dataset(val_ds, os.path.join(globals.save_dir, "val.pt"), meta={"seq_len": globals.seq_len, "pred_horizon": globals.pred_horizon, "dt": globals.dt, "split": "val"})
    save_dataset(test_ds, os.path.join(globals.save_dir, "test.pt"), meta={"seq_len": globals.seq_len, "pred_horizon": globals.pred_horizon, "dt": globals.dt, "split": "test"})

    return train_ds, val_ds, test_ds
