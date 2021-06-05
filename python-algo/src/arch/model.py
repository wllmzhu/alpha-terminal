import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import SpatialEncoder, ScalarEncoder
from .action_heads import ActionTypeHead, LocationHead

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
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=256, hidden_size=256)
        self.action_type_head = ActionTypeHead()
        self.location_head = LocationHead()
    
    def forward(self, observation_feature, game_state, hidden_and_cell_states):
        hidden_state, cell_state = self.lstm(observation_feature, hidden_and_cell_states)
        action_type, action_type_logits, action_type_logp = self.action_type_head(hidden_state, game_state)
        location, location_logits, location_logp = self.location_head(hidden_state, game_state, action_type)
        return action_type, location, (hidden_state, cell_state)
    
    def init_hidden_state(self, batch_size=1):
        device = next(self.parameters()).device
        hidden = (torch.zeros(batch_size, 256).to(device), 
                  torch.zeros(batch_size, 256).to(device))
        return hidden