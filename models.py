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
                input_shape=(1, 65, 65),
                embedding_dim=128,
                stride=2
            ):
        super().__init__()

        channels, height, width = input_shape
        self.stride = stride

        height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1
        height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(height * width * 64, embedding_dim)

    def forward(self, x):
        x = torch.squeeze(x, dim=1) # batch_size, ch, height, width
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
    
class MeowMeowObservationEncoder(nn.Module):
    def __init__(self,
                input_shape=(1, 65, 65),
                embedding_dim=128,
                stride=2
            ):
        super().__init__()

        self.embedding_dim = embedding_dim

        channels, height, width = input_shape
        self.stride = stride

        height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1
        height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(height * width * 64, self.embedding_dim)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.contiguous().view(batch_size * time_steps, channels, height, width)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(batch_size, time_steps, -1)
        return x

class MeowMeowParentEncoder(nn.Module):
    def __init__(self, 
                environment_embedding_dim=128,
                embedding_dim=128
            ):
        super().__init__()

        self.observation_encoder = MeowMeowObservationEncoder()
        self.fc = nn.Linear(
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
        x = self.fc(x)
        x = x.view(batch_size, time_steps, -1)
        return x
    
class MeowMeowPredictor(nn.Module):
    def __init__(self, embedding_dim=128, action_dim=32, encoding_dim=128):
        super().__init__()
        self.action_embedding = nn.Sequential(
            nn.Linear(2, action_dim),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(encoding_dim + action_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
    
    def forward(self, state_encoding, action):
        action = self.action_embedding(action) # Encode actions
        x = torch.cat([state_encoding, action], dim=1)
        x = self.fc(x)
        return x
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MeowMeowModel(nn.Module):
    def __init__(self, 
                    n_steps=17, 
                    repr_dim=128, 
                    training=False
                ):
        super().__init__()
        self.n_steps = n_steps
        self.repr_dim = repr_dim
        self.training = training
        
        self.env_encoder = MeowMeowEnvironmentEncoder(embedding_dim=self.repr_dim) # Encodes wall
        self.parent_encoder = MeowMeowParentEncoder(embedding_dim=self.repr_dim) # Encodes wall + observation
        self.predictor = MeowMeowPredictor(embedding_dim=self.repr_dim, encoding_dim=self.repr_dim) # predicts state representation
    
    def forward(self, states, actions):
        batch_size, time_steps, action_dim = actions.shape

        path = states[:, :, 0:1, :, :].clone()
        wall = states[:, :, 1:, :, :].clone()

        # Time step 0
        env_encoding = self.env_encoder(wall[:, :1])
        inital_state_embedding = self.parent_encoder(path[:, :1], env_encoding)

        # If training, then pre compute for all remaning timestamps, excluding first
        target_state_embeddings = None
        if self.training:
            target_state_embeddings = self.parent_encoder(path[:, 1:], env_encoding)

        # Loop for all time steps
        predicted_state_embeddings = []
        predicted_state_embeddings.append(inital_state_embedding[:, 0])
        for i in range(time_steps):
            pred_embedding = self.predictor(
                                predicted_state_embeddings[i],
                                actions[:, i]
                            )
            predicted_state_embeddings.append(pred_embedding)

        predicted_state_embeddings = torch.stack(predicted_state_embeddings, dim=1)

        return predicted_state_embeddings, target_state_embeddings, env_encoding
    
    def loss(self, predicted_states, target_states, env_encoding):
        # Invariance Loss or D loss
        predicted_states = predicted_states[:, 1:]
        batch_size, time_steps = predicted_states.shape[0], predicted_states.shape[1]
        
        mse_loss = F.mse_loss(predicted_states, target_states)

        # Variance Loss
        x = predicted_states.contiguous().view(batch_size * time_steps, predicted_states.shape[-1])
        y = target_states.contiguous().view(batch_size * time_steps, target_states.shape[-1])
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        env_encoding = env_encoding - env_encoding.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        std_env = torch.sqrt(env_encoding.var(dim=0) + 0.0001)
        env_std_loss = torch.mean(F.relu(1 - std_env)) / 2 

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.repr_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.repr_dim)

        return mse_loss, std_loss, cov_loss, env_std_loss