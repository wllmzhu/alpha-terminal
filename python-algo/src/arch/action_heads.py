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

    def forward(self, x, game_state):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        mask = self.valid_action_type_mask(game_state)
        logits = torch.where(mask.to(self.device), logits, torch.tensor(-1e+8).to(self.device))
        
        dist = Categorical(logits=logits)
        action_type = dist.sample()
        logp = dist.log_prob(action_type)
        return action_type.item(), logits, logp

    def valid_action_type_mask(self, game_state):
        # NOOP is always available
        # structure and mobile units must be affordable
        # upgrade and remove are valid if there exists structure
        
        # NOTE: gamestate=None for SL
        if game_state is None:
            return torch.tensor([True] * 9)

        mask = [True] * 9
        # check affordability
        for unit in constants.STRUCTURES + constants.MOBILES:
            mask[unit] = mask[unit] and game_state.number_affordable(constants.ACTION_SHORTHAND[unit]) > 0
        # if there is any structure not pending removal
        mask[constants.REMOVE]  = any(map(
            lambda loc: game_state.contains_stationary_unit(loc) and not game_state.game_map[loc][0].pending_removal, 
            constants.MY_LOCATIONS))
        # if there is any structure
        mask[constants.UPGRADE] = any(map(game_state.contains_stationary_unit, constants.MY_LOCATIONS))
        mask = torch.tensor(mask)
        return mask


class LocationHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(256, 210)
        self.already_removing = [False] * 210
        self.already_upgrading = [False] * 210

    def flush_queue_mask(self):
        self.already_removing = [False] * 210
        self.already_upgrading = [False] * 210

    def forward(self, x, game_state, action_type):
        logits = self.fc1(x)
        mask = self.valid_location_mask(game_state, action_type)
        logits = torch.where(mask.to(self.device), logits, torch.tensor(-1e+8).to(self.device))
        
        dist = Categorical(logits=logits)
        location = dist.sample()
        logp = dist.log_prob(location)

        if action_type == constants.REMOVE:
            self.already_removing[location.item()] = True
        elif action_type == constants.UPGRADE:
            self.already_upgrading[location.item()] = True
        
        return constants.MY_LOCATIONS[location.item()], logits, logp

    def valid_location_mask(self, game_state, action_type):
        # NOTE: gamestate=None for SL
        if game_state is None or action_type == constants.NOOP:             # SL or NOOP
            mask = [True] * 210
        elif action_type in constants.STRUCTURES + constants.MOBILES:     # structure or mobile
            mask = [game_state.can_spawn(constants.ACTION_SHORTHAND[action_type], loc) for loc in constants.MY_LOCATIONS]
        elif action_type == constants.REMOVE:                               # remove
            mask = [bool(game_state.contains_stationary_unit(loc)) 
                    and not game_state.game_map[loc][0].pending_removal
                    for loc in constants.MY_LOCATIONS]
            mask = mask and not self.already_removing
        else:                                                               # upgrade
            mask = [bool(game_state.contains_stationary_unit(loc)) 
                    # and not self.already_upgrading[loc[0],loc[1]] == 1
                    for loc in constants.MY_LOCATIONS]
            mask = mask and not self.already_removing
        mask = torch.tensor(mask)
        return mask