from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


# Main models developed by our team
class MeowMeowEnvironmentEncoder(nn.Module):
    def __init__(self,
                wall_shape=(1, 65, 65),
                embedding_dim=128,
                stride=2
            ):
        super().__init__()

        self.stride = stride
        channels, height, width = wall_shape

        for _ in range(2):
            height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1

        self.cnn_model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
        )
        self.fully_connected = nn.Linear(height * width * 64, embedding_dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.squeeze(x, dim=1) # batch_size, ch, height, width
        x = self.cnn_model(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return x
    
class MeowMeowObservationEncoder(nn.Module):
    def __init__(self,
                obs_shape=(1, 65, 65),
                embedding_dim=128,
                stride=2
            ):
        super().__init__()

        self.stride = stride
        self.embedding_dim = embedding_dim
        channels, height, width = obs_shape

        for _ in range(2):
            height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1

        self.cnn_model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
        )
        self.fully_connected = nn.Linear(height * width * 64, self.embedding_dim)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.contiguous().view(batch_size * time_steps, channels, height, width)
        x = self.cnn_model(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        x = x.view(batch_size, time_steps, -1)
        return x

class MeowMeowParentEncoder(nn.Module):
    def __init__(self, 
                environment_embedding_dim=128,
                embedding_dim=128
            ):
        super().__init__()

        self.observation_encoder = MeowMeowObservationEncoder()
        self.fully_connected = nn.Linear(
            self.observation_encoder.embedding_dim + environment_embedding_dim, 
            embedding_dim
        )
    
    def forward(self, observation, environment_embedding):
        observation_embedding = self.observation_encoder(observation)
        
        batch_size, time_steps, _ = observation_embedding.shape

        environment_embedding = environment_embedding.unsqueeze(1)
        environment_embedding = environment_embedding.repeat(1, time_steps, 1)

        x = torch.cat([observation_embedding, environment_embedding], dim=2)
        x = x.contiguous().view(batch_size * time_steps, x.shape[-1])
        x = self.fully_connected(x)
        x = x.view(batch_size, time_steps, -1)
        return x
    
class MeowMeowPredictor(nn.Module):
    def __init__(self, embedding_dim=128, encoding_dim=128):
        super().__init__()
        action_encoding_dim = 16
        fc_embedding_dim = action_encoding_dim + encoding_dim
        self.fully_connected = nn.Sequential(
            nn.Linear(fc_embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.a_embed = nn.Sequential(
            nn.Linear(2, action_encoding_dim),
            nn.ReLU()
        )
    
    def forward(self, state_encoding, action):
        action = self.a_embed(action) # Encode actions
        x = torch.cat([state_encoding, action], dim=1)
        x = self.fully_connected(x)
        return x
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MeowMeowModel(nn.Module):
    def __init__(self, 
                    repr_dim=128, 
                    training_mode=False
                ):
        super().__init__()
        self.repr_dim = repr_dim
        self.training_mode = training_mode
        self.env_encoder = MeowMeowEnvironmentEncoder(embedding_dim=self.repr_dim) # Encodes wall
        self.parent_encoder = MeowMeowParentEncoder(embedding_dim=self.repr_dim) # Encodes wall + observation
        self.predictor = MeowMeowPredictor(embedding_dim=self.repr_dim, encoding_dim=self.repr_dim) # predicts state representation
    
    def forward(self, states, actions):
        _, time_steps, _ = actions.shape
        predicted_state_embeddings = []
        encoded_state_embeddings = None
        path, wall = (states[:, :, i:i+1, :, :].clone() for i in range(2))

        # Time step 0 (Compute initial encodings for both environment and agent)
        env_encoding = self.env_encoder(wall[:, :1])
        inital_state_encoding = self.parent_encoder(path[:, :1], env_encoding)

        # If training, then pre compute for all remaning timestamps, excluding first
        if self.training_mode:
            encoded_state_embeddings = self.parent_encoder(path[:, 1:], env_encoding)

        # Loop for all time steps and generate predictions
        predicted_state_embeddings.append(inital_state_encoding[:, 0])
        for i in range(time_steps):
            pred_embedding = self.predictor(
                                predicted_state_embeddings[i],
                                actions[:, i]
                            )
            predicted_state_embeddings.append(pred_embedding)

        predicted_state_embeddings = torch.stack(predicted_state_embeddings, dim=1)
        return env_encoding, predicted_state_embeddings, encoded_state_embeddings
    
    def loss(self, predicted_states, encoded_states, env_encoding):
        # Invariance Loss or D loss between predicted and encoded states
        predicted_states = predicted_states[:, 1:]
        batch_size, time_steps = predicted_states.shape[0], predicted_states.shape[1]
        mse_loss = F.mse_loss(predicted_states, encoded_states)

        # Variance Loss calculation for predictions and encodings
        x = predicted_states.contiguous().view(batch_size * time_steps, predicted_states.shape[-1])
        y = encoded_states.contiguous().view(batch_size * time_steps, encoded_states.shape[-1])
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        env_encoding = env_encoding - env_encoding.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Variance loss calculation for walls
        std_env = torch.sqrt(env_encoding.var(dim=0) + 0.0001)
        env_std_loss = torch.mean(F.relu(1 - std_env)) / 2 

        # Covariance loss calculation for predictions and encodings
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.repr_dim) + off_diagonal(cov_y).pow_(2).sum().div(self.repr_dim)

        return mse_loss, std_loss, cov_loss, env_std_loss