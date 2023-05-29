import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ImitateModel(nn.Module):
    def __init__(self, win_len, use_state=False, state_dim=10, emb_dim=128):
        super().__init__()
        self.model_conv = models.resnet18(pretrained=True)
        self.use_state = use_state
        self.state_dim = state_dim
        self.win_len = win_len
        self.emb_dim = emb_dim
        self.num_ftrs = self.model_conv.fc.in_features  # 512
        self.encoder_in_ftrs = 128
        self.model_conv.fc = Identity()
        if self.use_state:
            self.encoder_in_ftrs += 128
            self.state_encoder = nn.Sequential(
                nn.Linear(self.state_dim, 128),
                nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(self.num_ftrs * win_len, 128),
            nn.ReLU(),
            # ('lin2', nn.Linear(128, 32)),
            # ('relu2', nn.ReLU()),
            # ('lin3', nn.Linear(32, 4)),
        )
        self.encoder = nn.Linear(self.encoder_in_ftrs, 4)

    def forward(self, x):
        if self.use_state:
            state = x['state']
            x = x['images']
        b, win, c, h, w = x.shape
        x = torch.reshape(x, [-1, c, h, w])
        x = self.model_conv(x)  # [b*win, num_ftrs]
        x = torch.reshape(x, [b, win, self.num_ftrs])
        x = torch.reshape(x, [b, -1])
        x = self.fc(x)
        if self.use_state:
            state_feat = self.state_encoder(state)
            x = torch.cat([x, state_feat], 1)
        x = self.encoder(x)
        return x


def create_model(window_len, device, loss_type='mse', use_state=False, state_dim=10):
    imitate_model = ImitateModel(
        window_len, use_state=use_state, state_dim=state_dim)
    imitate_model = imitate_model.to(device)
    criterion = nn.MSELoss() if 'mse' in loss_type else nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(
        imitate_model.parameters(), lr=0.0003, weight_decay=0)
    # Decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=2000, gamma=0.5)
    return imitate_model, criterion, optimizer_conv, exp_lr_scheduler
