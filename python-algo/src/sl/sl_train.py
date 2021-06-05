import torch
import time
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop

from src.arch.model import SL_Agent
from src.sl.dataset import Terminal_Replay_Dataset_SL, get_all_filenames, split_filenames


BATCH_SIZE = 16
MAX_SEQ_LEN = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
CLIP = 0.5

torch.manual_seed(789)
np.random.seed(678)

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
torch.autograd.set_detect_anomaly(True)

# model path
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, 'LSTM' + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))


def sl_train(agent, optimizer, train_loader, val_loader):
    agent.train()
    
    for epoch in range(NUM_EPOCHS):






        # loss_sum = 0.0
        # i = 0
        #     traj = traj.to(DEVICE).float()

        #     loss = agent.get_sl_loss(traj, replay_data)
        #     optimizer.zero_grad()
        #     loss.backward()

        #     # add a grad clip
        #     parameters = [p for p in agent.model.parameters() if p is not None and p.requires_grad]
        #     torch.nn.utils.clip_grad_norm_(parameters, CLIP)

        #     optimizer.step()


        # loss_sum += loss.item()
        # i += 1

        train_loss = loss_sum / (i + 1e-9)
        #val_loss = eval(model, criterion, val_loader, train_set, val_set)
        print("Train loss: {:.6f}.".format(train_loss))
        #print("Train loss: {:.6f}, Val loss: {:.6f}.".format(train_loss, val_loss))

    torch.save(agent.model, SAVE_PATH + "_val" + ".pkl")


def eval(model, criterion, data_loader, train_set, val_set):
    model.eval()

    n_samples = len(val_set)
    loss_sum = 0.0

    for feature, target in data_loader:
        feature = feature.to(DEVICE).float()
        target = target.to(DEVICE).float()

        output = agent.unroll(feature)

        if debug:
            print("feature.size(): ", feature.size())
            print("target.size(): ", target.size())
            print("output.size(): ", output.size())
            break

        loss = criterion(output, target)
        loss_sum += output.size(0) * loss.item()

    return loss_sum / n_samples


if __name__ == '__main__':
    filename_list = get_all_filenames('/Users/williamzhu/Github/EE239_Terminal_RL/alphaterminal/data/competitions')
    train_filenames, val_filenames, test_filenames = split_filenames(filename_list, ratio=(0.7, 0.15, 0.15))

    train_set = Terminal_Replay_Dataset_SL(train_filenames, max_seq_len=MAX_SEQ_LEN)
    val_set = Terminal_Replay_Dataset_SL(val_filenames, max_seq_len=MAX_SEQ_LEN)
    test_set = Terminal_Replay_Dataset_SL(test_filenames, seq_length=MAX_SEQ_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    #model = load_latest_model() if RESTORE else choose_model(MODEL)
    # model.to(DEVICE)
    agent = SL_Agent()
    agent.to(DEVICE)

    optimizer = Adam(agent.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(agent, optimizer, train_loader, val_loader)
