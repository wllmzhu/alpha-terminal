import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .. import constants 
from .. import gamelib

class ActionTypeHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = torch.where(mask.to(self.device), logits, torch.tensor(-1e+8).to(self.device))
        
        dist = Categorical(logits=logits)
        action_type = dist.sample()
        logp = dist.log_prob(action_type)
        return action_type.item(), logits, logp

class LocationHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(256, 210)

    def forward(self, x, mask):
        logits = self.fc1(x)
        logits = torch.where(mask.to(self.device), logits, torch.tensor(-1e+8).to(self.device))
        
        dist = Categorical(logits=logits)
        location = dist.sample()
        logp = dist.log_prob(location)

        return constants.MY_LOCATIONS[location.item()], logits, logp
