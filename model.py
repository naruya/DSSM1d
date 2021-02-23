import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from networks import Prior, Posterior, Encoder, Decoder
from utils import init_weights
import numpy as np


class SSM(nn.Module):
    def __init__(self, args):
        super(SSM, self).__init__()

        self.s_dim = s_dim = args.s_dim
        self.a_dim = a_dim = args.a_dim
        self.o_dim = o_dim = args.o_dim
        self.h_dim = h_dim = args.h_dim
        self.device = args.device
        self.args = args

        self.encoder = torch.nn.DataParallel(
            Encoder(o_dim, h_dim).to(self.device[0]),
            self.device)
        self.decoder = torch.nn.DataParallel(
            Decoder(s_dim, o_dim).to(self.device[0]),
            self.device)
        self.prior = torch.nn.DataParallel(
            Prior(s_dim, a_dim).to(self.device[0]),
            self.device)
        self.posterior = torch.nn.DataParallel(
            Posterior(self.prior, s_dim, a_dim, h_dim).to(self.device[0]),
            self.device)

        self.distributions = nn.ModuleList([
            self.prior, self.posterior, self.encoder, self.decoder])
        init_weights(self.distributions)

        # for s_aux_loss
        self.prior01 = Normal(torch.tensor(0.).to(self.device[0]),
                              scale=torch.tensor(1.).to(self.device[0]),)

        self.g_optimizer = optim.Adam(self.distributions.parameters())


    def forward(self, x_0, x, a, train=True, return_x=False):
        _B, _T = x.size(0), x.size(1)
        x = x.transpose(0, 1)  # T,B,3,64,64
        a = a.transpose(0, 1)  # T,B,1

        _xq, _xp = [], []
        s_loss, x_loss, s_aux_loss = 0, 0, 0

        s_prev = self.sample_s_0(x_0)

        for t in range(_T):
            x_t, a_t = x[t], a[t]
            h_t = self.encoder(x_t)

            q = Normal(*self.posterior(s_prev, a_t, h_t))
            p = Normal(*self.prior(s_prev, a_t))
            sq_t = q.rsample()
            xq_t = self.decoder(sq_t)

            # SSM Losses
            s_loss += torch.sum(
                kl_divergence(q, p), dim=[1,]).mean()
            x_loss += - torch.sum(
                Normal(xq_t, torch.ones(x_t.shape, device=x_0.device)).log_prob(x_t),
                dim=[1]).mean()
            s_aux_loss += kl_divergence(
                q, self.prior01).mean()

            s_prev = sq_t

            if return_x:
                sp_t = p.rsample()
                xp_t = self.decoder(sp_t)
                _xp.append(xp_t)
                _xq.append(xq_t)

        if return_x:
            return x, torch.stack(_xq), torch.stack(_xp)

        g_loss, d_loss = 0., 0.
        g_loss += s_loss + x_loss

        return_dict = {
            "loss": g_loss.item(),
            "s_loss": s_loss.item(),
            "x_loss": x_loss.item(),
            "s_aux_loss": s_aux_loss.item(),
        }
        return g_loss, d_loss, return_dict


    def sample_s_0(self, x_0):
        device = x_0.device

        # dummy
        a_t = torch.zeros(x_0.size(0), self.a_dim).to(device)
        s_prev = torch.zeros(x_0.size(0), self.s_dim).to(device)
        h_t = self.encoder(x_0)
        s_t = Normal(*self.posterior(s_prev, a_t, h_t)).mean
        return s_t


    def sample_x(self, x_0, x, a):
        with torch.no_grad():
            x_list = []
            for _x in self.forward(x_0, x, a, False, return_x=True):
                _x = torch.clamp(_x, 0, 1)
                _x = _x.transpose(0, 1).detach().cpu().numpy()  # BxT
                _x = (np.transpose(_x, [0,1,3,4,2]) * 255).astype(np.uint8)
                x_list.append(_x)
        return x_list


    # for simulation
    def step(self, a_t):
        s_t = self.prior(self.s_t, a_t)[0]
        self.s_t = s_t
        x_t = self.decoder(s_t)
        return x_t


    # for simulation
    def reset(self, x_0):
        s_t = self.sample_s_0(x_0)
        self.s_t = s_t
        x_t = self.decoder(s_t)
        return x_t
