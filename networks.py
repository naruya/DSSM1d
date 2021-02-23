import torch
from torch import nn
from torch.nn import functional as F


DIM = 128


class Prior(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Prior, self).__init__()

        self.fc_loc11 = nn.Linear(s_dim + a_dim, DIM)
        self.fc_loc12 = nn.Linear(DIM, s_dim)

        self.fc_loc21 = nn.Linear(s_dim + DIM, DIM)
        self.fc_loc22 = nn.Linear(DIM, s_dim)

        self.fc_loc31 = nn.Linear(s_dim + DIM, DIM)
        self.fc_loc32 = nn.Linear(DIM, s_dim)

        self.fc_loc41 = nn.Linear(s_dim + DIM, DIM)
        self.fc_loc42 = nn.Linear(DIM, s_dim)

        self.fc_scale11 = nn.Linear(DIM * 4, DIM)
        self.fc_scale12 = nn.Linear(DIM, s_dim)

    def forward_shared(self, s_prev, a):
        h = torch.cat([s_prev, a], 1)
        h1 = F.relu(self.fc_loc11(h))
        s1 = self.fc_loc12(h1)

        h = torch.cat([s1, h1], 1)
        h2 = F.relu(self.fc_loc21(h))
        s2 = s1 + self.fc_loc22(h2)

        h = torch.cat([s2, h2], 1)
        h3 = F.relu(self.fc_loc31(h))
        s3 = s2 + self.fc_loc32(h3)

        h = torch.cat([s3, h3], 1)
        h4 = F.relu(self.fc_loc41(h))
        s4 = s3 + self.fc_loc42(h4)

        loc = s4

        h = torch.cat([h1, h2, h3, h4], 1)
        scale = self.fc_scale12(F.relu(self.fc_scale11(h)))

        return loc, scale

    def forward(self, s_prev, a):
        loc, scale = self.forward_shared(s_prev, a)
        # TODO: loc = F.tanh(loc)?
        return loc, F.softplus(scale)


class Posterior(nn.Module):
    def __init__(self, prior, s_dim, a_dim, h_dim):
        super(Posterior, self).__init__()
        self.prior = prior
        self.fc1 = nn.Linear(s_dim * 2 + h_dim, DIM)
        self.fc21 = nn.Linear(DIM, s_dim)
        self.fc22 = nn.Linear(DIM, s_dim)

    def forward(self, s_prev, a, h):
        if hasattr(self.prior, 'module'):
            prior = self.prior.module
        else:
            prior = self.prior

        loc, scale = prior.forward_shared(s_prev, a)
        h = torch.cat([loc, scale, h], 1)
        h = F.relu(self.fc1(h))
        # TODO: loc = F.tanh(loc)?
        return self.fc21(h), F.softplus(self.fc22(h))


class Encoder(nn.Module):
    def __init__(self, o_dim, h_dim):
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
    def __init__(self, s_dim, o_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(s_dim, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, o_dim)

    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        x = torch.tanh(self.fc3(h))
        return x