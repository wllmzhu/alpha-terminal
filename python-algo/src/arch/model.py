import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoders import SpatialEncoder, ScalarEncoder
from .action_heads import ActionTypeHead, LocationHead
from .. import constants
from .. import gamelib

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.scalar_encoder = ScalarEncoder()
    
    def forward(self, spatial_features, scalar_features):
        latent_spatial = self.spatial_encoder(spatial_features)
        latent_scalar = self.scalar_encoder(scalar_features)
        latent = torch.cat((latent_spatial, latent_scalar), -1) 
        return latent

class PolicyNet(nn.Module): 
    def __init__(self, device):
        super().__init__()
        embed_dim = 16
        self.device = device
        self.lstm = nn.LSTMCell(input_size=256+embed_dim*2, hidden_size=256)
        self.action_type_head = ActionTypeHead(device=device)
        self.location_head = LocationHead(device=device)
        self.action_embed = nn.Linear(10, embed_dim)
        self.loc_embed = nn.Linear(2, embed_dim)
        self.flush_queue_mask()
    
    def forward(self, observation_feature, last_action, last_loc, game_state, hidden_and_cell_states):
        last_action = torch.tensor(last_action)
        last_loc = torch.tensor(last_loc)

        last_action = nn.functional.one_hot(last_action, 10).float()
        last_action = self.action_embed(last_action)
        last_action = torch.unsqueeze(last_action, 0)
        last_loc = last_loc.float()
        last_loc = self.loc_embed(last_loc)
        last_loc = torch.unsqueeze(last_loc, 0)

        # gamelib.debug_write('=========================================================')
        # gamelib.debug_write(last_action.shape)
        # gamelib.debug_write(last_loc.shape)
        # gamelib.debug_write(observation_feature.shape)
        # gamelib.debug_write('=========================================================')        

        lstm_input = torch.cat((observation_feature, last_action, last_loc), -1)

        hidden_state, cell_state = self.lstm(lstm_input, hidden_and_cell_states)

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
        return torch.tensor([True] * 10)

    mask = [True] * 10
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


class State2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, spatial_features, scalar_features, target_action_seq, target_loc_seq, teacher_forcing_ratio = 0.5):
        #spatial_features: (N, 8, 28, 28)
        #scalar_features: (N, 7)
        #target_action_seq: (N, num_actions)
        #target_loc_seq: (N, num_actions, 2)

        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = spatial_features.shape[0]
        num_actions = target_action_seq.shape[1]
        
        #tensor to store decoder outputs
        action_seq = torch.zeros((num_actions, batch_size, 1))
        loc_seq = torch.zeros((num_actions, batch_size, 2))
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        spatial_features = spatial_features.to(self.device)
        scalar_features = scalar_features.to(self.device)
        observation_features = self.encoder(spatial_features, scalar_features).to(self.device)
        
        last_action = torch.tensor(9)
        last_loc = torch.tensor([9, 9])
        hidden_and_cell_states = (torch.zeros(1, 256), torch.zeros(1, 256))
        
        for t in range(1, num_actions):
            last_action = last_action.to(self.device)
            last_loc = last_loc.to(self.device)
            hidden_and_cell_states = (hidden_and_cell_states[0].to(self.device), hidden_and_cell_states[1].to(self.device))

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            last_action, _, _, last_loc, _, _, hidden_and_cell_states = self.decoder(observation_features, last_action, last_loc, None, hidden_and_cell_states)
            
            #place predictions in a tensor holding predictions for each token
            action_seq[t] = last_action
            loc_seq[t] = last_loc
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            if teacher_force:
                last_action = target_action_seq[t]
                last_loc = target_loc_seq[t]
        
        return action_seq, loc_seq