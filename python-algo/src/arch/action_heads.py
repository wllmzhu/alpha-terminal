import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants 

class ActionTypeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mask = valid_action_type_mask()
        x = torch.where(mask, x, torch.tensor(-1e+8))
        probs = F.softmax(x, dim=-1)
        action_type = torch.multinomial(probs, num_samples=1)
        return action_type.item()

class LocationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 210)

    def forward(self, x, action_type):
        x = self.fc1(x)
        mask = valid_location_mask()
        x = torch.where(mask, x, torch.tensor(-1e+8))
        probs = F.softmax(x, dim=-1)
        location = torch.multinomial(probs, num_samples=1)
        return constants.MY_LOCATIONS[location.item()]
        
def valid_action_type_mask():
    # TODO: affordable 
    return torch.tensor([1] * 9, dtype=torch.bool)

def valid_location_mask():
    # FIXME
    # if action_type == 1:
    #     # NOOP
    #     mask = torch.tensor([1] * 210, dtype=torch.bool)
    # elif action_type in [2, 3, 4, 5, 6, 7]:
    #     mask = [game_state.can_spawn(ALL_ACTIONS[action_type], [loc]) for loc in MY_LOCATIONS]
    #     mask = torch.tensor(mask, dtype=torch.bool)
    # elif action_type == 8:
    #     mask = []
    #     for loc in MY_LOCATIONS:
    #         upgradable = game_state.contains_stationary_unit(loc)
    return torch.tensor([1] * 210, dtype=torch.bool)