import torch
from torch import nn
from torch.nn import functional as F


class Prior(nn.Module):
    def __init__(self, s_dim, a_dim, DIM=1024):
        super(Prior, self).__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, DIM)
        self.fc21 = nn.Linear(DIM, s_dim)
        self.fc22 = nn.Linear(DIM, s_dim)

    def forward(self, s_prev, a):
        h = torch.cat([s_prev, a], 1)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return loc, scale + 1e-1


class Posterior(nn.Module):
    def __init__(self, prior, s_dim, a_dim, h_dim, DIM=1024):
        super(Posterior, self).__init__()
        self.prior = prior
        self.fc1 = nn.Linear(s_dim * 2 + h_dim, DIM)
        self.fc21 = nn.Linear(DIM, s_dim)
        self.fc22 = nn.Linear(DIM, s_dim)

    def forward(self, s_prev, a, h):
        loc, scale = self.prior(s_prev, a)
        h = torch.cat([loc, scale, h], 1)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return loc, scale + 1e-1


class Encoder(nn.Module):
    def __init__(self, o_dim, h_dim, DIM=256):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(o_dim, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, h_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class Decoder(nn.Module):
    def __init__(self, s_dim, o_dim, DIM=256):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(s_dim, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, o_dim)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        # x = torch.tanh(self.fc3(h))
        x = self.fc3(h)  # for gym
        return x
