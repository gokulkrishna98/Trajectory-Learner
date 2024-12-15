import numpy as np
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tr
import torch.optim as optim
from tqdm import tqdm

from dataset import TrajectoryDataset
from models import MeowMeowModel

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def save_model(model, epoch, save_path="checkpoint", file_name="jepa"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


def train(model, train_loader, num_epochs=50, learning_rate=1e-4, device=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        epoch_loss = 0.0
        mse_loss_ttl, std_loss_ttl, cov_loss_ttl, env_std_loss_ttl = 0.0, 0.0, 0.0, 0.0

        # VICReg Coefficients for computing the total loss
        mse_loss_coef, std_loss_coef, cov_loss_coef = 25.0, 25.0, 1.0
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            encoded_wall, predicted_states, encoded_states = model(states, actions)
            
            mse_loss, std_loss, cov_loss, env_std_loss = model.loss(predicted_states, encoded_states, encoded_wall)
            loss = (mse_loss * mse_loss_coef) + (std_loss * std_loss_coef) + (cov_loss * cov_loss_coef) + (env_std_loss * std_loss_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mse_loss_ttl += mse_loss.item()
            std_loss_ttl += std_loss.item()
            cov_loss_ttl += cov_loss.item()
            env_std_loss_ttl += env_std_loss.item()
        save_model(model, epoch)
        print(f"Epoch {epoch+1}, MSE Loss: {mse_loss_ttl / len(train_loader):.10f}")
        print(f"Epoch {epoch+1}, STD Loss: {std_loss_ttl / len(train_loader):.10f}")
        print(f"Epoch {epoch+1}, COV Loss: {cov_loss_ttl / len(train_loader):.10f}")
        print(f"Epoch {epoch+1}, STD ENV Loss: {env_std_loss_ttl / len(train_loader):.10f}")
        print(f"Epoch {epoch+1}, TOTAL Loss: {epoch_loss / len(train_loader):.10f}")
        
    return predicted_states, encoded_states


if __name__ == "__main__":
    print("Training main function")
    device = get_device()

    traj_dataset = TrajectoryDataset(
        data_dir = "/scratch/DL24FA/train",
        states_filename = "states.npy",
        actions_filename = "actions.npy",
        s_transform = None,
        a_transform = None
    )


    dataloader = DataLoader(traj_dataset, batch_size=64, shuffle=True)

    print("Dataset loaded successfully")

    model = MeowMeowModel(training_mode=True)
    model.to(device)
    epochs = 20
    lr = 1e-4
    _, _ = train(model, dataloader, epochs, lr, device)

    # Optionally, save the final model
    save_model(model, "model")
