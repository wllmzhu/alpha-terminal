import torch
import time
import numpy as np
import os
import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from torch.nn.utils import clip_grad_norm

from src.arch.model import State2Seq, FeatureEncoder, PolicyNet
from src.sl.dataset import Terminal_Replay_Dataset, get_all_filenames, split_filenames

torch.manual_seed(789)
np.random.seed(678)

ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
# torch.autograd.set_detect_anomaly(True)

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, 'LSTM' + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))

CLIP = 1
DRY_RUN = True
BATCH_SIZE = 1
MAX_SEQ_LEN = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10

def train(model, optimizer, train_loader, val_loader):
    model.train()

    epoch_loss = 0
    for sample in tqdm(train_loader):
        state_list = sample['state_list']
        env_list = sample['env_list']
        target_actions_list = sample['actions_list']
        target_locs_list = sample['locs_list']

        optimizer.zero_grad()
        action_seq, loc_seq = model(state_list, env_list)
        loss = nn.CrossEntropyLoss(action_seq, target_actions_seq) \
               + nn.CrossEntropyLoss(action_seq)
        loss.backward()
        clip_grad_norm(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

        if DRY_RUN:
            break
            
    return epoch_loss / len(train_loader)


encoder = FeatureEncoder()
decoder = PolicyNet(DEVICE)
state2seq = State2Seq(encoder, decoder, DEVICE)

filename_list = get_all_filenames('/Users/williamzhu/Github/EE239_Terminal_RL/alphaterminal/data/competitions')
train_filenames, val_filenames, test_filenames = split_filenames(filename_list, ratio=(0.7, 0.15, 0.15))

train_set = Terminal_Replay_Dataset_SL(train_filenames, max_seq_len=MAX_SEQ_LEN)
val_set = Terminal_Replay_Dataset_SL(val_filenames, max_seq_len=MAX_SEQ_LEN)
test_set = Terminal_Replay_Dataset_SL(test_filenames, seq_length=MAX_SEQ_LEN)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

optimizer = Adam(state2seq.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train(state2seq, optimizer, train_loader, val_loader)