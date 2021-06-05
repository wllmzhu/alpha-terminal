import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants 

class ActionTypeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x, game_state):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mask = valid_action_type_mask(game_state)
        x = torch.where(mask, x, torch.tensor(-1e+8))
        probs = F.softmax(x, dim=-1)
        action_type = torch.multinomial(probs, num_samples=1)
        return action_type.item()

class LocationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 210)

    def forward(self, x, game_state, action_type):
        x = self.fc1(x)
        mask = valid_location_mask(game_state, action_type)
        x = torch.where(mask, x, torch.tensor(-1e+8))
        probs = F.softmax(x, dim=-1)
        location = torch.multinomial(probs, num_samples=1)
        return constants.MY_LOCATIONS[location.item()]
        
def valid_action_type_mask(game_state):
    # NOOP is always available
    # structure and mobile units must be affordable
    # upgrade and remove are valid if there exists structure
    
    # NOTE: gamestate=None for SL
    if game_state is None:
        return torch.tensor([True] * 9)

    mask = [True] * 9
    # check affordability
    for i_unit in range(1, 7):
        mask[i_unit] = mask[i_unit] and game_state.number_affordable(constants.ALL_ACTIONS[i_unit]) > 0
    if not any(map(game_state.contains_stationary_unit, constants.MY_LOCATIONS)):
        mask[7] = False
        mask[8] = False
    mask = torch.tensor(mask)
    return mask

def valid_location_mask(game_state, action_type):
    # NOTE: gamestate=None for SL
    if game_state is None or action_type == 0:  # SL or NOOP
        mask = [True] * 210
    elif action_type in range(1, 7):            # structure or mobile
        mask = [game_state.can_spawn(constants.ALL_ACTIONS[action_type], loc) for loc in constants.MY_LOCATIONS]
    else:                                       # remove or upgrade
        mask = [game_state.contains_stationary_unit(loc) is not False for loc in constants.MY_LOCATIONS]
    mask = torch.tensor(mask)
    return mask