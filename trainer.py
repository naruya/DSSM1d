import os
from logzero import logger
import numpy as np
import torch


class Trainer(object):
    def __init__(self, model, train_loader, test_loader, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.iters_to_accumulate = args.iters_to_accumulate

        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model


    def train(self, epoch):
        self.forward(self.train_loader, self._train, epoch, "train")


    def test(self, epoch):
        self.forward(self.test_loader, self._test, epoch, "test")


    def forward(self, loader, run, epoch, mode):
        self.i, N, summ = 0, 0, None

        for x_0, x, a in loader:
            x_0 = x_0.to(self.device[0])
            x = x.to(self.device[0])
            a = a.to(self.device[0])

            return_dict = run(x_0, x, a)

            if summ is None:
                keys = return_dict.keys()
                summ = dict(zip(keys, [0] * len(keys)))

            # update summary
            for k in summ.keys():
                v = return_dict[k]
                summ[k] += v * x.size(0)

            self.i += 1
            N += x.size(0)

        # write summary
        for k, v in summ.items():
            summ[k] = v / N
        logger.info("({}) Epoch: {} {}".format(mode, epoch, summ))


    def _train(self, x_0, x, a):
        model = self.model
        model.train()

        model.g_optimizer.zero_grad()
        g_loss, d_loss, return_dict = model.forward(x_0, x, a, True)
        g_loss = g_loss / self.iters_to_accumulate
        g_loss.backward()

        if (self.i + 1) % self.iters_to_accumulate == 0:
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     model.distributions.parameters(), 1e+6)
            model.g_optimizer.step()

        return return_dict


    def _test(self, x_0, x, a):
        model = self.model
        model.eval()

        with torch.no_grad():
            g_loss, d_loss, return_dict = model.forward(x_0, x, a, False)

        return return_dict