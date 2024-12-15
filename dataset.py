from typing import NamedTuple, Optional
import torch
import numpy as np
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

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
