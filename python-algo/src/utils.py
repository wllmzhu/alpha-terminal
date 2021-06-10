import os
from pathlib import Path
import json
import torch

from . import gamelib

class CheckpointManager:
    def __init__(self, is_enemy, is_prod):
        self.is_enemy = is_enemy
        self.is_prod = is_prod

        self.checkpoint_path = self.get_checkpoint_path()
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        if not self.is_prod:
            self.model_id = self.get_latest_model_id()
            self.next_id = self.get_next_model_id()
            gamelib.debug_write('MODEL ID', self.model_id)

    def get_checkpoint_path(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        if self.is_prod:
            parent_dir = os.path.join(file_dir, os.pardir)
            parent_dir = os.path.abspath(parent_dir)
            return os.path.join(parent_dir, 'weights')
        else:
            grandparent_dir = os.path.join(file_dir, os.pardir, os.pardir)
            grandparent_dir = os.path.abspath(grandparent_dir)
            return os.path.join(grandparent_dir, 'model')
    
    def checkpoint_exists(self):
        if not self.is_prod:
            return self.model_id is not None
        else:
            policy_path = os.path.join(self.get_checkpoint_path(), 'policy.pth')
            return Path(policy_path).exists()

    def save_model(self, feature_encoder, policy_net, optimizer):
        if self.is_enemy or self.is_prod:
            gamelib.debug_write("Enemy/Production can't save model. Skipping...")
            return
        model_path = os.path.join(self.checkpoint_path, str(self.next_id))
        os.mkdir(model_path)
        feature_encoder_path = os.path.join(model_path, 'feature.pth')
        torch.save(feature_encoder.state_dict(), feature_encoder_path)
        policy_path = os.path.join(model_path, 'policy.pth')
        torch.save(policy_net.state_dict(), policy_path)
        optimizer_path = os.path.join(model_path, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)

    def save_stats(self, stats_dict):
        stats_path = os.path.join(self.checkpoint_path, 'stats', f'{self.next_id}.json')
        with open(stats_path, 'w') as fp:
            json.dump(stats_dict, fp)

    def get_latest_model_id(self):
        # enemy gets updated every 1,000 games
        model_ids = list(map(int, filter(str.isnumeric, os.listdir(self.checkpoint_path))))
        if len(model_ids) == 0:
            return None
        model_ids.sort()
        if self.is_enemy:
            return model_ids[len(model_ids) // 1000 * 1000]
        else:
            return model_ids[-1]

    def get_next_model_id(self):
        return 0 if self.model_id is None else self.model_id + 1

    def get_latest_model_path(self):
        if self.is_prod:
            feature_encoder_name = os.path.join(self.checkpoint_path, 'feature.pth')
            policy_name = os.path.join(self.checkpoint_path, 'policy.pth')
            return feature_encoder_name, policy_name, ''
        else:
            model_id = str(self.model_id)
            feature_encoder_name = os.path.join(self.checkpoint_path, model_id, 'feature.pth')
            policy_name = os.path.join(self.checkpoint_path, model_id, 'policy.pth')
            optimizer_name = os.path.join(self.checkpoint_path, model_id, 'optimizer.pth')
            return feature_encoder_name, policy_name, optimizer_name