import random
from sys import maxsize
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from . import gamelib
from . import constants
from .arch.model import FeatureEncoder, PolicyNet

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        # TODO: gpu support
        self.device = 'cpu'
        gamelib.debug_write('Using {}'.format(self.device))

        self.is_learning = True
        self.lr = 0.01

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP, STRUCTURE_ONEHOT
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        STRUCTURE_ONEHOT = {WALL: [1, 0, 0], SUPPORT: [0, 1, 0], TURRET: [0, 0, 1]}
        # This is a good place to do initial setup
        self.scored_on_locations = []
        
        self.setup_policy_net()
        if self.is_learning:
            self.setup_vanila_policy_gradient()

    def setup_policy_net(self):
        self.feature_encoder = FeatureEncoder()
        self.policy = PolicyNet()
        self.memory_state = self.policy.init_hidden_state()

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        self.policy_net_strategy(game_state)
        game_state.submit_turn()

    def policy_net_strategy(self, game_state):
        if self.is_learning:
            action_type_logps, location_logps = [], []

        spatial_features, scalar_features = self.game_state_to_features(game_state)
        observation_features = self.feature_encoder(spatial_features, scalar_features)
        action_type = None
        while action_type != 0:
            action_type, _, action_type_logp, location, _, location_logp, self.memory_state \
                = self.policy(observation_features, game_state, self.memory_state)
            # gamelib.debug_write(action_type, location)
            if action_type in [1, 2, 3, 4, 5, 6]:
                game_state.attempt_spawn(constants.ALL_ACTIONS[action_type], [location])
            elif action_type == 7:
                game_state.attempt_upgrade([location])
            elif action_type == 8:
                game_state.attempt_remove([location])
            
            if self.is_learning:
                action_type_logps.append(action_type_logp)
                location_logps.append(location_logp)
        
        if self.is_learning:
            reward_prev_turn = self.compute_reward(game_state)
            self.ep_rews.append(reward_prev_turn)
            self.ep_action_type_logps.append(action_type_logps) 
            self.ep_location_logps.append(location_logps)

    def game_state_to_features(self, game_state):
        # spatial features
        spatial_features = deepcopy(game_state.game_map._GameMap__map)
        for x in range(len(spatial_features)):
            for y in range(len(spatial_features[x])):
                if len(spatial_features[x][y]) > 0:
                    unit = spatial_features[x][y][0]
                    feature = []
                    feature +=  STRUCTURE_ONEHOT[unit.unit_type]
                    feature +=  [unit.health, unit.player_index]
                    feature +=  list(map(int, [unit.pending_removal, unit.upgraded]))
                    # TODO: add features
                    spatial_features[x][y] = feature
                else:
                    spatial_features[x][y] = [0] * 7
        spatial_features = torch.tensor(spatial_features, dtype=torch.float).to(self.device)
        spatial_features = spatial_features.permute(2, 0, 1)
        spatial_features = torch.unsqueeze(spatial_features, 0)

        # scalar features
        # TODO: should data whitening be in model or here?
        scalar_features =   [game_state.my_health] + game_state.get_resources(0) 
        scalar_features +=  [game_state.enemy_health] + game_state.get_resources(1)
        scalar_features +=  [game_state.turn_number] 
        scalar_features = torch.tensor(scalar_features, dtype=torch.float).to(self.device)
        scalar_features = torch.unsqueeze(scalar_features, 0)

        return spatial_features, scalar_features

    # Methods for reinforcement learning

    def on_game_end(self):
        if self.is_learning:
            self.action_lengths = [len(logps) for logps in self.ep_action_type_logps]
            self.print_statistics()
            self.optimizer.zero_grad()
            episode_loss = self.compute_loss()
            episode_loss.backward()
            self.optimizer.step()

            # TODO: save model and optimizer state_dict for another day
    
    def setup_vanila_policy_gradient(self):
        params = list(self.feature_encoder.parameters()) + list(self.policy.parameters())
        # TODO: load state_dict from disk if possible to continue training
        self.optimizer = Adam(params, lr=self.lr)
        self.my_health = self.enemy_health = self.config['resources']['startingHP']

        self.ep_action_type_logps = []
        self.ep_location_logps = []
        self.ep_rews = []

    def compute_reward(self, game_state):
        reward = game_state.my_health - self.my_health
        reward += self.enemy_health - game_state.enemy_health
        self.my_health, self.enemy_health = game_state.my_health, game_state.enemy_health
        return reward

    def compute_loss(self):
        # TODO: check if need to shift rewards by 1
        ep_weights = list(self.reward_to_go())
        batch_weights = []
        for action_len, weight in zip(self.action_lengths, ep_weights):
            batch_weights.extend([weight] * action_len)
        batch_action_type_logps = [logp for logps in self.ep_action_type_logps for logp in logps]
        batch_location_logps = [logp for logps in self.ep_location_logps for logp in logps]
        
        batch_weights = torch.tensor(batch_weights, dtype=torch.float32)
        batch_action_type_logps = torch.cat(batch_action_type_logps)
        batch_location_logps = torch.cat(batch_location_logps)

        action_type_loss = -(batch_action_type_logps * batch_weights).mean()
        location_loss = -(batch_location_logps * batch_weights).mean() 
        
        return action_type_loss + location_loss

    def reward_to_go(self):
        n = len(self.ep_rews)
        rtgs = np.zeros_like(self.ep_rews)
        for i in reversed(range(n)):
            rtgs[i] = self.ep_rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def print_statistics(self):
        # reward and return
        gamelib.debug_write('EPISODE LENGTH', len(self.ep_rews))
        gamelib.debug_write('EPISODE RETURN', sum(self.ep_rews))
        gamelib.debug_write('REWARD MEAN', np.mean(self.ep_rews))
        gamelib.debug_write('REWARD STD', np.std(self.ep_rews))
        gamelib.debug_write('REWARD MAX', max(self.ep_rews))
        gamelib.debug_write('REWARD MIN', min(self.ep_rews))

        # actions
        gamelib.debug_write('ACTION LEN MEAN', np.mean(self.action_lengths))
        gamelib.debug_write('ACTION LEN STD', np.std(self.action_lengths))
        gamelib.debug_write('ACTION LEN MAX', max(self.action_lengths))
        gamelib.debug_write('ACTION LEN MIN', min(self.action_lengths)) 