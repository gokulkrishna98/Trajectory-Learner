import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as tr

from torch import nn
from torch.utils.data import Dataset 

class TrajectoryDataset(Dataset):
    """
    Arguments:
        data_dir: Absolute path to the dataset directory.
        states_filename: Name of states dataset file.
        actions_filename: Name of the actions dataset file.
        s_transform: Transformation for states.
        a_transform: Transformation for actions.
    what does it contain ?
        states is a numpy array - (num of data points, trajectory_length, 2, 65, 65)
        actions is a numpy array - (num of data_points, trajectory_length, 2)
        transforms should be image transformations
    """
    def __init__(self, data_dir, 
                 states_filename, 
                 actions_filename, 
                 s_transform=None, 
                 a_transform=None,
                 length=None):
        self.states = np.load(f"{data_dir}/{states_filename}", mmap_mode="r")
        self.actions = np.load(f"{data_dir}/{actions_filename}")
        if length is None:
            length = len(self.states)
        
        self.states = self.states[:length]
        self.actions = self.actions[:length]

        self.state_transform = s_transform
        self.action_transform = a_transform
    
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        
        if self.state_transform:
            for i in range(state.shape[0]):
                state[i] = self.state_transform(state[i])
        
        if self.action_transform:
            for i in range(action[i].shape[0]):
                action[i] = self.action_transform(action[i])
        
        return state, action

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_byol_transforms(mean, std):
    # Define the first augmentation pipeline
    transformT = tr.Compose([
        tr.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        tr.RandomRotation(degrees=90),  # Random rotation
        tr.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),  # Gaussian blur
        tr.Normalize(mean, std),  # Normalize for 2 channels
    ])

    # Define a slightly different second augmentation pipeline
    transformT1 = tr.Compose([
        tr.RandomVerticalFlip(p=0.5),  # Random vertical flip
        tr.RandomRotation(degrees=45),  # Different random rotation
        tr.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 1.5)),  # Gaussian blur with smaller kernel
        tr.Normalize(mean, std),  # Normalize for 2 channels
    ])
    
    return transformT, transformT1


def off_diagonal(matrix):
    # Create a mask for off-diagonal elements
    n = matrix.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=bool, device=matrix.device)
    
    # Use the mask to extract off-diagonal elements
    off_diag_elements = matrix[off_diag_mask]
    return off_diag_elements


def criterion(x, y, invar = 25, mu = 25, nu = 1, epsilon = 1e-5):
    bs = x.size(0)
    emb = x.size(1)

    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    invar_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_z_a = (x.T @ x) / (bs - 1)
    cov_z_b = (y.T @ y) / (bs - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / emb + off_diagonal(cov_z_b).pow_(2).sum() / emb

    loss = invar*invar_loss + mu*var_loss + nu*cov_loss
    return loss

def compute_mean_and_std(dataloader, is_channelsize3 = True):
    num_channels = 2  # Assuming you have 2 channels
    pixel_sum = [0] * num_channels
    pixel_squared_sum = [0] * num_channels
    total_pixels = 0

    # Iterate through the dataset
    for state, _ in dataloader:
        # Iterate through each channel
        for channel in range(num_channels):
            channel_data = state[:, :, channel, :, :].reshape(-1)  # Flatten the current channel
            pixel_sum[channel] += channel_data.sum().item()
            pixel_squared_sum[channel] += (channel_data ** 2).sum().item()
        
        # Total number of pixels per channel (all images combined)
        total_pixels += state.size(0) * state.size(1) * state.size(3) * state.size(4)

    # Calculate mean and std for each channel
    mean = [pixel_sum[c] / total_pixels for c in range(num_channels)]
    std = [
        np.sqrt((pixel_squared_sum[c] / total_pixels) - (mean[c] ** 2))
        for c in range(num_channels)
    ]
    
    # Adding a 3rd dimension
    if is_channelsize3:
        mean.append(mean[1])
        std.append(std[1])

    return mean, std

def save_model(model, epoch, save_path="checkpoints", file_name="encoder_"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

def get_vicreg_loss(model, img, transformation1, transformation2, criterion):
    x0 = transformation1(img)
    x1 = transformation2(img)
    _, (_, z0) = model(state=x0)
    _, (_, z1) = model(state=x1)

    loss = criterion(z0, z1)
    return loss
