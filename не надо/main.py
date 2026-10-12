import os
from torch.utils.data import DataLoader
import pandas as pd
import glob

import globals
import CNN_transform_network as Network
import dataloader
import visualization
import visualize_trajectories as visualize


train = False
test = False
real_data = True


for file in glob.glob(os.path.join("flight_logs_source", "*.csv")):
    df = pd.read_csv(file)
    # print(file, df["GPS"].head())
visualize.plot_all_trajectories_from_folders()

for file in glob.glob(os.path.join("flight_logs_generated/data", "*.csv")):
    df = pd.read_csv(file)
visualize.plot_all_trajectories_from_folders()

train_ds, val_ds, test_ds = dataloader.generate_data(real_data)

pin = (globals.device.type == "cuda")
train_loader = DataLoader(train_ds, batch_size=globals.batch_size, shuffle=True,  drop_last=True,  pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=globals.batch_size, shuffle=False, drop_last=False, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=globals.batch_size, shuffle=False, drop_last=False, pin_memory=pin)

N = 15

if train:
    Network.train_model(train_loader, val_loader)

if test:
    all_predictions, all_probs, pred_xyz = Network.test_model(test_loader)
    visualization.plot_trajectories_with_projections(train_loader, N, all_predictions, all_probs, pred_xyz)

# visualization.plot_train_samples(train_loader)
# visualization.plot_trajectories_with_projections(test_loader, N)
