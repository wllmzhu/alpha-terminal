
# -*- coding: utf-8 -*-

import os
import torch
from time import strftime, localtime
from .arch.model import FeatureEncoder, PolicyNet

# model path

FOLDER_PATH = "./model/"
if not os.path.exists(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)


def save_model(the_model,folder_path,model_type):
    # model_type is either 'feature' or 'policy', should be a str
    save_path=os.path.join(folder_path,model_type + "_" + strftime("%y-%m-%d_%H-%M-%S", localtime())+ "" + ".pth")
    torch.save(the_model.state_dict(),save_path)


def save_policy_net(self,folder_path):
    ''' save the parameters of policy net and feature encoder, respectively'''
    save_model(self.policy,folder_path,'policy')
    save_model(self.feature_encoder,folder_path,'feature')


def setup_policy_net(self,path=FOLDER_PATH,restore=True):
    self.feature_encoder = FeatureEncoder()
    self.policy = PolicyNet()

    if restore:
        feature_path=get_latest_model_path('feature',path)
        self.feature_encoder.load_state_dict(torch.load(feature_path))
        policy_path=get_latest_model_path('policy',path)
        self.policy.load_state_dict(torch.load(policy_path))
    # TODO: Not sure how to deal with hidden_state 
    self.memory_state = self.policy.init_hidden_state()


def get_latest_model_path(model_type, path):
    # model_type is either 'feature' or 'policy', should be a str
    models = list(filter(lambda x: model_type in x, os.listdir(path)))
    if len(models) == 0:
        print("None " + model_type + " models found!")
        return None

    models.sort()    
    model_path = os.path.join(path, models[-1])

    return model_path

