import os
import torch
from torch import nn


def save_model(model, epoch):
    if hasattr(model, 'module'):
        model = model.module
    save_dir = os.path.join("weights", model.args.timestamp)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "epoch{:05}.pt".format(epoch))

    save_dict = {}
    for i, dist in enumerate(model.distributions):
        save_dict.update({"g_net{}".format(i): dist.state_dict()})
    save_dict.update({"g_opt": model.g_optimizer.state_dict()})
    torch.save(save_dict, path)


def load_model(model, epoch=None, model_dir=None):
    if hasattr(model, 'module'):
        model = model.module
    save_dir = os.path.join("weights", model.args.timestamp)
    if model_dir:
        save_dir = os.path.join(model_dir, save_dir)
    path = os.path.join(save_dir, "epoch{:05}.pt".format(epoch))

    checkpoint = torch.load(path)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(checkpoint["g_net{}".format(i)])
    model.g_optimizer.load_state_dict(checkpoint["g_opt"])


# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def init_weights(model):
    if hasattr(model, 'module'):
        model = model.module

    # print("---- init weights ----")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.RNN, nn.RNNCell, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            # print("  ", type(m))
            continue
        # print("ok", type(m))


import re


class Extractor(object):
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        self.lines = {}
        for mode in ["train", "test"]:
            key = "({}) Epoch".format(mode)
            self.lines[mode] = [line for line in lines if key in line]

    def __call__(self, mode, key):
        key = "\'{}\'".format(key)
        lines = self.lines[mode]
        lines = [re.split('[{}]', line)[1].split(', ') for line in lines]
        lines = [[item for item in line if key in item] for line in lines]
        lines = [float(line[0].split()[1]) for line in lines]
        return lines