import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import SpatialEncoder, ScalarEncoder
from .action_heads import ActionTypeHead, LocationHead
from .. import constants

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.scalar_encoder = ScalarEncoder()
    
    def forward(self, spatial_features, scalar_features):
        latent_spatial = self.spatial_encoder(spatial_features)
        latent_scalar = self.scalar_encoder(scalar_features)
        return torch.cat((latent_spatial, latent_scalar), -1) 

class PolicyNet(nn.Module): 
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTMCell(input_size=256, hidden_size=256)
        self.action_type_head = ActionTypeHead(device=device)
        self.location_head = LocationHead(device=device)
        self.flush_queue_mask()
    
    def forward(self, observation_feature, game_state, hidden_and_cell_states):
        hidden_state, cell_state = self.lstm(observation_feature, hidden_and_cell_states)
        
        action_type_mask = get_action_type_mask(game_state, self.already_removing)
        action_type, action_type_logits, action_type_logp = self.action_type_head(hidden_state, action_type_mask)
        
        location_mask = get_location_mask(game_state, action_type, self.already_removing)
        location, location_logits, location_logp = self.location_head(hidden_state, location_mask)
        
        if action_type == constants.REMOVE:
            self.already_removing.add(tuple(location))
        
        return action_type, action_type_logits, action_type_logp, \
            location, location_logits, location_logp, \
            (hidden_state, cell_state)
    
    def init_hidden_state(self, batch_size=1):
        return (torch.zeros(batch_size, 256).to(self.device), torch.zeros(batch_size, 256).to(self.device))

    def flush_queue_mask(self):
        self.already_removing = set()

def get_action_type_mask(game_state, already_removing):
    # NOOP is always available
    # structure and mobile units must be affordable and spawnable
    # upgrade is valid if there exists structure
    # remove is valid if there exists structure which is not pending removal 
    #   (otherwise agent learns to exploit this invalid action)
    
    # NOTE: gamestate=None for SL
    if game_state is None:
        return torch.tensor([True] * 9)

    mask = [True] * 9
    for unit in constants.STRUCTURES + constants.MOBILES:
        # affordable
        mask[unit] = mask[unit] and game_state.number_affordable(constants.ACTION_SHORTHAND[unit]) > 0
        # spawnable
        mask[unit] = mask[unit] and any(map(lambda loc: game_state.can_spawn(constants.ACTION_SHORTHAND[unit], loc), constants.MY_LOCATIONS))
    # if there is any structure not pending removal
    mask[constants.REMOVE]  = any(map(lambda loc: game_state.contains_stationary_unit(loc) and tuple(loc) not in already_removing, constants.MY_LOCATIONS))
    # if there is any structure
    mask[constants.UPGRADE] = any(map(game_state.contains_stationary_unit, constants.MY_LOCATIONS))
    mask = torch.tensor(mask)
    return mask

def get_location_mask(game_state, action_type, already_removing):
    # NOTE: gamestate=None for SL
    if game_state is None or action_type == constants.NOOP:             # SL or NOOP
        mask = [True] * 210
    elif action_type in constants.STRUCTURES + constants.MOBILES:       # structure or mobile
        mask = [game_state.can_spawn(constants.ACTION_SHORTHAND[action_type], loc) for loc in constants.MY_LOCATIONS]
    elif action_type == constants.REMOVE:                               # remove
        mask = [bool(game_state.contains_stationary_unit(loc)) 
                and tuple(loc) not in already_removing
                for loc in constants.MY_LOCATIONS]
    else:                                                               # upgrade
        mask = [bool(game_state.contains_stationary_unit(loc)) for loc in constants.MY_LOCATIONS]
    mask = torch.tensor(mask)
    return mask