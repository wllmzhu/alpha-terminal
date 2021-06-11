import random
from sys import maxsize
from copy import deepcopy
import json

import numpy as np
import torch
from torch.optim import Adam

from . import gamelib
from . import constants
from . import utils
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
    def __init__(self, args):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        self.device = 'cpu'
        gamelib.debug_write('Using {}'.format(self.device))

        self.is_learning = args.is_learning
        if self.is_learning:
            gamelib.debug_write("I'm LEARNING!")
        self.lr = 0.01
        
        self.is_enemy = args.is_enemy
        if self.is_enemy:
            gamelib.debug_write("I'm EVIL so I'm not LEARNING! Ignoring is_learning...")
            self.is_learning = False

        self.is_prod = args.is_prod
        if self.is_prod:
            gamelib.debug_write("Production mode. Ignoring is_learning, is_enemy...")
            self.is_learning = False
            self.is_enemy = False

        self.checkpoint_manager = utils.CheckpointManager(self.is_enemy, self.is_prod)

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
        self.feature_encoder = FeatureEncoder().to(self.device)
        self.policy = PolicyNet(self.device).to(self.device)
        
        if self.is_learning:
            params = list(self.feature_encoder.parameters()) + list(self.policy.parameters())
            self.optimizer = Adam(params, lr=self.lr)
        
        if self.checkpoint_manager.checkpoint_exists():
            gamelib.debug_write('Loading model weights...')
            feature_encoder_path, policy_path, optimizer_path = self.checkpoint_manager.get_latest_model_path()
            self.feature_encoder.load_state_dict(torch.load(feature_encoder_path))
            self.policy.load_state_dict(torch.load(policy_path))

            if self.is_learning:
                self.optimizer.load_state_dict(torch.load(optimizer_path))

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
        action_type = torch.tensor(-1)
        location = torch.tensor([-1,-1])
        while action_type != constants.NOOP:
            action_type, _, action_type_logp, location, _, location_logp, self.memory_state \
                = self.policy(observation_features, action_type, location, game_state, self.memory_state)

            if action_type in constants.STRUCTURES + constants.MOBILES:
                game_state.attempt_spawn(constants.ACTION_SHORTHAND[action_type], [location])
            elif action_type == constants.UPGRADE:
                game_state.attempt_upgrade([location])
            elif action_type == constants.REMOVE:
                game_state.attempt_remove([location])
            
            if self.is_learning:
                action_type_logps.append(action_type_logp)
                location_logps.append(location_logp)
        
        # when all actions has been taken, flush the queue mask in policy net's action head
        self.policy.flush_queue_mask()
        
        if self.is_learning:
            if game_state.turn_number > 0: # skip turn 0 reward
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
                    feature +=  [unit.health]
                    feature +=  [unit.player_index==1, unit.player_index==2]
                    feature +=  list(map(int, [unit.pending_removal, unit.upgraded]))
                    # TODO: add features
                    spatial_features[x][y] = feature
                else:
                    spatial_features[x][y] = [0] * 8
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

    def on_final_reward(self,game_state_string):
        if self.is_learning:
            turn_state = json.loads(game_state_string)
            
            # change of health
            my_health = float(turn_state.get('p1Stats')[0])
            enemy_health = float(turn_state.get('p2Stats')[0])
            reward = my_health - self.my_health
            reward += self.enemy_health - enemy_health
            
            # additional reward of winner or loser
            self.winner = int(turn_state.get('endStats')['winner'])
            win_or_not_reward = constants.WINNER_REWARD if self.winner == 1 else constants.LOSER_REWARD
            reward += win_or_not_reward
            
            # append the last reward 
            self.ep_rews.append(reward)
            
            # get turn number in total
            self.total_turn = int(turn_state.get('turnInfo')[1])

    def on_game_end(self):
        if self.is_learning:
            # train
            gamelib.debug_write('Optimizing policy network...')
            self.optimizer.zero_grad()
            episode_loss = self.compute_loss()
            episode_loss.backward()
            self.optimizer.step()
            
            # log metrics
            self.loss = episode_loss.item()
            stats_dict = self.get_statistics()
            gamelib.debug_write(stats_dict)

            # checkpoint
            self.checkpoint_manager.save_model(self.feature_encoder, self.policy, self.optimizer)
            self.checkpoint_manager.save_stats(stats_dict)
    
    def setup_vanila_policy_gradient(self):
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
        self.action_lengths = [len(logps) for logps in self.ep_action_type_logps]
        ep_weights = list(self.reward_to_go())
        batch_weights = []
        for action_len, weight in zip(self.action_lengths, ep_weights):
            batch_weights.extend([weight / action_len] * action_len)
        batch_action_type_logps = [logp for logps in self.ep_action_type_logps for logp in logps]
        batch_location_logps = [logp for logps in self.ep_location_logps for logp in logps]
        self.ep_ret = batch_weights
        
        batch_weights = torch.tensor(batch_weights, dtype=torch.float32).to(self.device)
        batch_action_type_logps = torch.cat(batch_action_type_logps)
        batch_location_logps = torch.cat(batch_location_logps)

        action_type_loss = -(batch_action_type_logps * batch_weights).mean()
        location_loss = -(batch_location_logps * batch_weights).mean() 
        
        return action_type_loss + location_loss

    def reward_to_go(self):
        n = len(self.ep_rews)        
        rtgs = np.zeros_like(self.ep_rews)
        gamma = constants.GAMMA
        for i in reversed(range(n)):
            rtgs[i] = self.ep_rews[i] + gamma*(rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def get_statistics(self):
        stats = dict()
        # winner and turn_number
        stats['winner']         = self.winner
        stats['total_turn']     = self.total_turn
        # policy gradient loss
        stats['policy_gradient_loss'] = self.loss
        # reward and return in theory
        stats['episode_length'] = len(self.ep_rews)
        stats['episode_return'] = sum(self.ep_rews) 
        stats['reward_mean']    = np.mean(self.ep_rews)
        stats['reward_std']     = np.std(self.ep_rews)
        stats['reward_max']     = max(self.ep_rews)
        stats['reward_min']     = min(self.ep_rews)
        # actual return experienced by the agent
        stats['return_cumulative']  = sum(self.ep_ret)
        stats['return_mean']        = np.mean(self.ep_ret)
        stats['return_std']         = np.std(self.ep_ret)
        stats['return_max']         = max(self.ep_ret)
        stats['return_min']         = min(self.ep_ret)
        # actions
        stats['action_length_cumulative']   = sum(self.action_lengths)
        stats['action_length_mean']         = np.mean(self.action_lengths)
        stats['action_length_std']          = np.std(self.action_lengths) 
        stats['action_length_max']          = max(self.action_lengths)
        stats['action_length_min']          = min(self.action_lengths)
        return stats
