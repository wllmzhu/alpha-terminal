#!/usr/bin/env python
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import os

torch.manual_seed(0)

cost_dict = {'WALL':1, 'FACTORY':4, 'TURRET':2, 'SCOUT':1, 'DEMOLISHER':3, 'INTERCEPTOR':1} 
upgrade_cost_dict = {'WALL':1, 'FACTORY':4, 'TURRET':4}
idx_dict = {'WALL':0, 'FACTORY':1, 'TURRET':2, 'health':3, 'belongs_to_1':4, 'belongs_to_2':5, 'to_remove':6, 'upgraded':7,
            'sd_damage':8, 'sd_belongs_to_1':9, 'sd_belongs_to_2':10, 'bch_damage':11, 'bch_belongs_to_1':12, 'bch_belongs_to_2':13,
            'dmg_damage':14, 'dmg_belongs_to_1':15, 'dmg_belongs_to_2':16, 'sld_hp':17, 'sld_belongs_to_1':18, 'sld_belongs_to_2':19,
            'dth_number':20, 'dth_belongs_to_1':21, 'dth_belongs_to_2':22, 'atk_damage':23, 'atk_belongs_to_1':24, 'atk_belongs_to_2':25}
rev_idx_dict = {v: k for k, v in idx_dict.items()}
env_dict = {'player1_health': 0, 'player1_SP': 1, 'player1_MP': 2, 'player2_health': 3, 'player2_SP': 4, 'player2_MP': 5, 'elapsed_time': 6}


class Terminal_Replay_Dataset(Dataset):
    def __init__(self, replay_file_list, max_seq_len, with_events=False):
        super(Terminal_Replay_Dataset, self).__init__()

        self.replay_file_list = replay_file_list
        self.max_seq_len = max_seq_len
        self.with_events = with_events

    def __len__(self):
        return len(self.replay_file_list)
    
    def __getitem__(self, idx):
        filename = self.replay_file_list[idx]
        state_list = []
        env_list = []
        actions_list = []   #list of list of int
        locs_list = []      #list of list of [x,y]
        with open(filename) as f:
            player1_sa, player2_sa = self._process_replay(f)
            for sa in player1_sa + player2_sa:
                state, env, actions, locs = sa
                state = torch.from_numpy(state)
                env = torch.from_numpy(env)
                state_list.append(state)
                env_list.append(env)
                actions_list.append(actions)   
                locs_list.append(locs)
        assert(len(state_list) == len(env_list) == len(actions_list) == len(locs_list))
        return {'len': len(state_list), 'state_list': state_list, 'env_list': env_list, 'actions_list': actions_list, 'locs_list':locs_list}

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
        loc_list = []
        for idx, unit in enumerate(spawn):
            #initialize curr_state
            if idx==0:
                state = start_state.copy()
            #if spawning enemy unit, ignore
            if unit[3] != from_player_view:
                continue   
            #construct action tensor
            x, y = unit[0]
            if from_player_view == 2: #if from player2's view, coordinates are flipped
                x, y = self._flip(x, y)
            unit_type = unit[1]
            #append to list
            action_list.append(unit_type + 1)
            loc_list.append([x,y]) 
        #append NOOP
        action_list.append(0)
        loc_list.append([0,0])
        return [start_state, env, action_list, loc_list]

    def _construct_state(self, p1Units, p2Units, event_list, from_player_view):
        #state as of the end of last turn
        #(type, health, upgraded, demolish, touchdown_type, damage_received, damage_dealt)
        if self.with_events == True:
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
        else:
            state = np.zeros((28,28,8))
            if from_player_view == 1:
                state = self._add_units(p1Units, state, internal_playerID=1, flip_coord=False)
                state = self._add_units(p2Units, state, internal_playerID=2, flip_coord=False)
            else: #from player2's view, so player2 becomes "player1", and all coord needs to be flipped
                state = self._add_units(p1Units, state, internal_playerID=2, flip_coord=True)
                state = self._add_units(p2Units, state, internal_playerID=1, flip_coord=True)   

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

    def _flip(self, x, y):
        new_y = 27 - y
        return x, new_y


def get_all_filenames(data_dir):
    filename_list = []
    for dir in os.scandir(data_dir):
        if dir.is_dir() and int(dir.name) > 200: #get only the recent games
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


def Terminal_Replay_Collate(batch):
    # if len(batch) > 1:
    #     print('sorry, only supporting batch size of 1 right now! (since the 20 or so state-action pairs from each replay file is kind of like a batch)')

    max_len_actions = np.max([len(actions) for replay in batch for actions in replay['actions_list']])
    max_len_game = np.max([len(replay['state_list']) for replay in batch])

    NOOP = 0
    NOOP_LOC = [0,0]
    empty_state = torch.zeros_like(batch[0]['state_list'][0])
    empty_env = torch.zeros_like(batch[0]['env_list'][0])
    empty_actions = [NOOP for _ in range(max_len_actions)]
    empty_locs = [NOOP_LOC for _ in range(max_len_actions)]

    state_list = []
    env_list = []
    actions_list = []
    locs_list = []
    
    for replay in batch:
        if replay['len'] <= 4:
            print('ignoring bad replay')
            continue
        
        replay['state_list'] += [empty_state] * (max_len_game - len(replay['state_list']))
        replay['state_list'] = torch.stack(replay['state_list'], axis=0)
        state_list.extend(replay['state_list'])

        replay['env_list'] += [empty_env] * (max_len_game - len(replay['env_list']))
        replay['env_list'] = torch.stack(replay['env_list'], axis=0)
        env_list.extend(replay['env_list'])

        for actions in replay['actions_list']:
            actions += [NOOP] * (max_len_actions - len(actions))
        replay['actions_list'] += [empty_actions] * (max_len_game - len(replay['actions_list']))        
        replay['actions_list'] = torch.tensor(replay['actions_list'])
        actions_list.extend(replay['actions_list'])
            
        for locs in replay['locs_list']:
            locs += [NOOP_LOC] * (max_len_actions - len(locs))
        replay['locs_list'] += [empty_locs] * (max_len_game - len(replay['locs_list']))        
        replay['locs_list'] = torch.tensor(replay['locs_list'])
        locs_list.extend(replay['locs_list'])

    state_list = torch.stack(state_list, axis=0)
    env_list = torch.stack(env_list, axis=0)
    actions_list = torch.stack(actions_list, axis=0)
    locs_list = torch.stack(locs_list, axis=0)
    
    new_batch =  {'state_list': state_list, 'env_list': env_list, 'actions_list': actions_list, 'locs_list': locs_list}    
    return new_batch


def test_dataset():
    filename_list = get_all_filenames('/home/wllmzhu/Documents/Github/C1GamesStarterKit/data/competitions')
    dataset = Terminal_Replay_Dataset(filename_list, max_seq_len=50, with_events=False)
    temp = dataset[0]
    #print(temp['actions_list'][0][0])
    pass

def test_dataloader():
    filename_list = get_all_filenames('/home/wllmzhu/Documents/Github/C1GamesStarterKit/data/competitions')
    dataset = Terminal_Replay_Dataset(filename_list, max_seq_len=50, with_events=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=Terminal_Replay_Collate)
    it = iter(dataloader)
    hi = next(it)
    #print(hi['state_list'].shape)

if __name__ == '__main__':
    test_dataset()
    test_dataloader()