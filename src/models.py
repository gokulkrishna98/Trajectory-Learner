import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr
from lightly.models.modules.heads import VICRegProjectionHead

class SimpleEncoder(nn.Module):
    def __init__(self, embed_size, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 12, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((5, 5), stride=2)
        self.pool2 = nn.MaxPool2d((5, 5), stride=5)
        self.fc1 = nn.Linear(432, 4096)
        self.fc2 = nn.Linear(4096, embed_size)

    def forward(self, x):
        # h,w = 65
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x1
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x3 + x2
        x3 = self.pool2(x3)

        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        return x3

class VICRegModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=256,
            hidden_dim=512,
            output_dim=512,
            num_layers=2,
        )
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.h = None
        self.c = None

    def set_hc(self, h, c):
        self.h = h
        self.c = c 
    
    def reset_hc(self):
        self.h = self.h.zero_() 
        self.c = self.c.zero_()

    def forward(self, action):
        self.h, self.c = self.lstm_cell(action, (self.h, self.c))
        return self.h

class SimpleEncoderv2(nn.Module):
    def __init__(self, embed_size, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 12, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((5, 5), stride=2)
        self.pool2 = nn.MaxPool2d((5, 5), stride=5)
        self.fc1 = nn.Linear(432, 1024)
        self.fc2 = nn.Linear(1024, embed_size)

    def forward(self, x):
        # h,w = 65
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x1
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x3 + x2
        x3 = self.pool2(x3)

        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        return x3
    
class Predictorv2(nn.Module):
    def __init__(self, hidden_dim=1024, embed_dim=512, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.prev_embed = None

        self.linear1 = nn.Linear(embed_dim+action_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def set_init_embedding(self, init_embed):
        self.prev_embed = init_embed

    def forward(self, action):
        x = torch.cat((self.prev_embed, action), dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class JEPAModelv2(nn.Module):
    def __init__(self, embed_size, hidden_size, action_size=2, input_channel_size=2):
        super().__init__()
        self.encoder = SimpleEncoderv2(embed_size, input_channel_size)
        self.predictor = Predictorv2(hidden_size, embed_size, action_size)
        self.repr_dim = embed_size
        
    def set_init_embedding(self, state):
        inp = self.encoder(state)
        self.predictor.set_init_embedding(inp)

    def forward(self, action=None, state=None):
        sy_hat, sy = None, None
        if action is not None:
            sy_hat = self.predictor(action)
        if state is not None:
            sy = self.encoder(state)    
        return sy_hat, sy

    def forward_inference(self, actions, states):
        B, L, D = states.shape[0], actions.shape[1], self.repr_dim

        o = states[:, 0, :, :, :]
        self.set_init_embedding(o)

        result = torch.empty((B, L+1, D)).to(o.device)
        result[:, 0, :] = (self.predictor.h)
        for i in range(1, L+1):
            sy_hat, _ = self.forward(actions[:, i-1, :], None)
            result[:, i, :] = sy_hat

        return result


class JEPAModel(nn.Module):
    def __init__(self, embed_size, input_channel_size):
        super().__init__()
        self.encoder = VICRegModel(SimpleEncoder(embed_size, input_channel_size))
        self.predictor = Predictor(input_channel_size, embed_size)
        self.repr_dim = 1024
        
    def set_predictor(self, o, co, use_expander=False):
        x, z = self.encoder.forward(o)
        so = z if use_expander else x
        self.predictor.set_hc(so, co)
        return so
    
    def reset_predictor(self):
        self.predictor.reset_hc()

    def forward(self, action=None, state=None):
        sy_hat, sy = None, None
        if action is not None:
            sy_hat = self.predictor(action)
        if state is not None:
            sy = self.encoder(state)

        return sy_hat, sy

    def forward_inference(self, actions, state):
        B, L, D = state.shape[0], actions.shape[1], self.repr_dim

        o = state 
        co = torch.zeros((B, D)).to(o.device)
        self.set_predictor(o, co, use_expander=False)

        result = torch.empty((B, L+1, D))
        result[:, 0, :], _ = self.encoder(state) 
        for i in range(1, L+1):
            sy_hat, _ = self.forward(actions[:, i-1, :], state=None)
            result[:, i, :] = sy_hat

        return result
