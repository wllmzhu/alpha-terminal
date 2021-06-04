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
    
    def forward(self, observation_feature, game_state, hidden=None):
        if hidden == None:
            hidden = self.init_hidden_state()
        lstm_out, hidden = self.lstm(observation_feature, hidden)
        action_type = self.action_type_head(lstm_out, game_state)
        location = self.location_head(lstm_out, game_state, action_type)
        return action_type, location
    
    def init_hidden_state(self, batch_size=1):
        device = next(self.parameters()).device
        hidden = (torch.zeros(batch_size, 256).to(device), 
                  torch.zeros(batch_size, 256).to(device))
        return hidden
            
if __name__ == '__main__':
    feature_encoder = FeatureEncoder()
    policy = PolicyNet()

    spatial_features = torch.rand(1, 7, 28, 28)
    scalar_features = torch.rand(1, 7)
    observation_features = feature_encoder(spatial_features, scalar_features)
    action_type, location = policy(observation_features)