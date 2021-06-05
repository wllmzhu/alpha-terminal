#!/usr/bin/env python
import torch
import json
import numpy as np
from torch.utils.data import Dataset
import os

cost_dict = {'WALL':1, 'FACTORY':4, 'TURRET':2, 'SCOUT':1, 'DEMOLISHER':3, 'INTERCEPTOR':1} 
upgrade_cost_dict = {'WALL':1, 'FACTORY':4, 'TURRET':4}
idx_dict = {'WALL':0, 'FACTORY':1, 'TURRET':2, 'to_remove':3, 'upgraded':4, 'belongs_to_1':5, 'belongs_to_2':6, 'health':7,
            'sd_damage':8, 'sd_belongs_to_1':9, 'sd_belongs_to_2':10, 'bch_damage':11, 'bch_belongs_to_1':12, 'bch_belongs_to_2':13,
            'dmg_damage':14, 'dmg_belongs_to_1':15, 'dmg_belongs_to_2':16, 'sld_hp':17, 'sld_belongs_to_1':18, 'sld_belongs_to_2':19,
            'dth_number':20, 'dth_belongs_to_1':21, 'dth_belongs_to_2':22, 'atk_damage':23, 'atk_belongs_to_1':24, 'atk_belongs_to_2':25}
rev_idx_dict = {v: k for k, v in idx_dict.items()}
env_dict = {'player1_health': 0, 'player1_SP': 1, 'player1_MP': 2, 'player2_health': 3, 'player2_SP': 4, 'player2_MP': 5, 'elapsed_time': 6}


class Terminal_Replay_Dataset_SL(Dataset):
    def __init__(self, replay_file_list, max_seq_len):
        self.replay_file_list = replay_file_list
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.replay_file_list)
    
    def __getitem__(self, idx):
        filename = self.replay_file_list[idx]
        state_list = []
        env_list = []
        actions_list = []

        with open(filename) as f:
            #print(filename)
            player1_sa, player2_sa = self._process_replay(f)

            for sa in player1_sa + player2_sa:
                state, env, actions = sa
                state = torch.from_numpy(state)
                env = torch.from_numpy(env)
                if len(actions) == 0:
                    actions = np.zeros((28,28,8,1))                    
                else:
                    actions = actions[:self.max_seq_len]
                    actions = np.stack(actions, axis=3)
                actions = torch.from_numpy(actions)
                state_list.append(state)
                env_list.append(env)
                actions_list.append(actions)   

        state = torch.stack(state_list, axis=0)
        env = torch.stack(env_list, axis=0)

        return {'state': state, 'env': env, 'actions': actions}

    def _process_replay(self, f):
        player1_state_action = []
        player2_state_action = []
        event_list = []

        for idx,item in enumerate(f):
            if item == '\n':
                continue
            item_dict = json.loads(item)
            if 'debug' in item_dict.keys():
                continue
            turn_info = item_dict['turnInfo']
            action_phase_num = turn_info[2]
            
            if action_phase_num == -1:
                p1Stats = item_dict['p1Stats']
                p2Stats = item_dict['p2Stats']
                turn_num = turn_info[1]
                #create dataset from player1's perspective
                env1 = self._construct_env(p1Stats, p2Stats, turn_num, from_player_view=1)
                #create dataset from player2's perspective
                env2 = self._construct_env(p1Stats, p2Stats, turn_num, from_player_view=2)                
            
            elif action_phase_num == 0: 
                p1Units = item_dict['p1Units']
                p2Units = item_dict['p2Units']
                events = item_dict['events']
                spawn = events['spawn']
                #create dataset from player1's perspective
                start_state1 = self._construct_state(p1Units, p2Units, event_list, from_player_view=1)
                state_action1 = self._construct_state_action(start_state1, env1, spawn, from_player_view=1)
                player1_state_action.append(state_action1)
                #create dataset from player2's perspective
                start_state2 = self._construct_state(p1Units, p2Units, event_list, from_player_view=2)
                state_action2 = self._construct_state_action(start_state2, env2, spawn, from_player_view=2)                
                player2_state_action.append(state_action2)
                event_list = []
                
            else:
                events = item_dict['events'] 
                event_of_interest = [events['selfDestruct'], events['breach'], events['damage'], 
                                    events['shield'], events['death'], events['attack']]
                event_list.append(event_of_interest)

        return player1_state_action, player2_state_action

    def _construct_state_action(self, start_state, env, spawn, from_player_view):
        action_list = []
        for idx, unit in enumerate(spawn):
            #initialize curr_state
            if idx==0:
                state = start_state.copy()
            #if spawning enemy unit, ignore
            if unit[3] != from_player_view:
                continue
                    
            #construct action tensor
            action_tensor = np.zeros((28,28,8))
            x, y = unit[0]
            if from_player_view == 2: #if from player2's view, coordinates are flipped
                x, y = self._flip(x, y)
            unit_type = unit[1]
            action_tensor[x, y, unit_type] = 1
            
            #append to list
            action_list.append(action_tensor) 
            
        return [start_state, env, action_list]

    def _add_units(self, units, state, internal_playerID, flip_coord):
        wall, factory, turret, scout, demolisher, interceptor, to_remove, upgraded = units
            
        for unit in wall:
            x, y, health= unit[0], unit[1], unit[2]
            if flip_coord:
                x, y = self._flip(x, y)
            state[x, y, idx_dict['WALL']] += 1
            state[x, y, idx_dict['health']] += health
            if internal_playerID == 1:
                state[x, y, idx_dict['belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['belongs_to_2']] = 1            
            
        for unit in factory:
            x, y, health= unit[0], unit[1], unit[2]
            if flip_coord:
                x, y = self._flip(x, y)
            state[x, y, idx_dict['FACTORY']] += 1
            state[x, y, idx_dict['health']] += health
            if internal_playerID == 1:
                state[x, y, idx_dict['belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['belongs_to_2']] = 1    

        for unit in turret:
            x, y, health= unit[0], unit[1], unit[2]
            if flip_coord:
                x, y = self._flip(x, y)
            state[x, y, idx_dict['TURRET']] += 1
            state[x, y, idx_dict['health']] += health
            if internal_playerID == 1:
                state[x, y, idx_dict['belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['belongs_to_2']] = 1    

        for unit in upgraded:
            x, y, health = unit[0], unit[1], unit[2]
            if flip_coord:
                x, y = self._flip(x, y)
            state[x, y, idx_dict['upgraded']] += 1
            state[x, y, idx_dict['health']] += health
            
        for unit in to_remove:
            x, y = unit[0], unit[1]
            if flip_coord:
                x, y = self._flip(x, y)
            state[x, y, idx_dict['to_remove']] += 1
                
        return state

    def _add_events(self, events, state, flip_coord):
        self_destruct, breach, damage, shield, death, attack = events
        
        for event in self_destruct:
            x, y = event[0]
            dmg = event[2]
            playerID = event[5]
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['sd_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['sd_belongs_to_2']] = 1
            state[x, y, idx_dict['sd_damage']] += dmg
                
        for event in breach:
            x, y = event[0]
            dmg = event[1]
            playerID = event[4]
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['bch_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['bch_belongs_to_2']] = 1
            state[x, y, idx_dict['bch_damage']] += dmg
        
        for event in damage:
            x, y = event[0]
            dmg = event[1]
            playerID = event[4]        
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['dmg_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['dmg_belongs_to_2']] = 1
            state[x, y, idx_dict['dmg_damage']] += dmg
            
        for event in shield:
            x, y = event[0] #of the receiving unit
            hp = event[2]
            playerID = event[6]        
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['sld_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['sld_belongs_to_2']] = 1
            state[x, y, idx_dict['sld_hp']] += hp
            
        for event in death:
            x, y = event[0]
            playerID = event[4]        
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['dth_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['dth_belongs_to_2']] = 1
            state[x, y, idx_dict['dth_number']] += 1
        
        for event in attack:
            x, y = event[0] #of unit being attacked
            dmg = event[2]
            playerID = event[6] #of player owning unit
            if flip_coord:
                x, y = self._flip(x, y)
            if playerID == 1:
                state[x, y, idx_dict['atk_belongs_to_1']] = 1
            else:
                state[x, y, idx_dict['atk_belongs_to_2']] = 1
            state[x, y, idx_dict['atk_damage']] += dmg

        return state

    def _construct_state(self, p1Units, p2Units, event_list, from_player_view):
        #state as of the end of last turn
        #(type, health, upgraded, demolish, touchdown_type, damage_received, damage_dealt)
        state = np.zeros((28,28,26))

        if from_player_view == 1:
            state = self._add_units(p1Units, state, internal_playerID=1, flip_coord=False)
            state = self._add_units(p2Units, state, internal_playerID=2, flip_coord=False)
            for event in event_list:
                state = self._add_events(event, state, flip_coord=False)
            
        else: #from player2's view, so player2 becomes "player1", and all coord needs to be flipped
            state = self._add_units(p1Units, state, internal_playerID=2, flip_coord=True)
            state = self._add_units(p2Units, state, internal_playerID=1, flip_coord=True)
            for event in event_list:
                state = self._add_events(event, state, flip_coord=True)
        
        return state   

    def _construct_env(self, p1Stats, p2Stats, turn_num, from_player_view):
        #(player1_health, player1_SP, player1_MP, player2_health, player2_SP, player2_MP, elapsed_time)
        env = np.zeros(7)
        if from_player_view == 1:
            env[0:3] = p1Stats[0:3]
            env[3:6] = p2Stats[0:3]
        else: #if from player2's view, internal playerID is flipped
            env[0:3] = p2Stats[0:3]
            env[3:6] = p1Stats[0:3]
        env[6] = turn_num     # elapsed_time
        return env

    def _flip(self, x, y):
        new_y = 27 - y
        return x, new_y



def get_all_filenames(data_dir):
    filename_list = []
    for dir in os.scandir(data_dir):
        if dir.is_dir() and int(dir.name) > 0:
            for replay in os.scandir(dir):
                filename = os.path.join(data_dir, dir, replay.name)
                filename_list.append(filename)
    return filename_list

def split_filenames(filename_list, ratio=(0.7, 0.15, 0.15)):
    length = len(filename_list)
    perm = np.random.permutation(np.arange(0, length))
    split_pt1 = int(ratio[0] * length)
    split_pt2 = int((ratio[0] + ratio[1]) * length)
    train_set = filename_list[perm[:split_pt1]]
    val_set = filename_list[perm[split_pt1:split_pt2]]
    test_set = filename_list[perm[split_pt2:]]
    return train_set, val_set, test_set



def test():
    filename_list = get_all_filenames('/Users/williamzhu/Github/EE239_Terminal_RL/alphaterminal/data/competitions')
    dataset = Terminal_Replay_Dataset_SL(filename_list, 50)
    state_env, actions = dataset[0]
    print(state_env)

if __name__ == '__main__':
    test()
