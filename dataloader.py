import os
import random
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, mode, epoch, args):
        self.T = args.T
        
        data_path = os.path.join(args.data_dir, "{}{}.pkl".format(mode, epoch))
        with open(data_path, mode='rb') as f:
            data = pickle.load(f)

        self.traj_a = np.array([data[i][0] for i in range(len(data))],
                               dtype=np.float32)
        self.traj_o = np.array([data[i][1] for i in range(len(data))],
                               dtype=np.float32)
        self.H = self.traj_a.shape[1]
        print(self.traj_a.shape, self.traj_o.shape)

        param_path = os.path.join(args.data_dir, "param{}.pkl".format(epoch))
        if mode == 'train':
            with open(param_path, mode='wb') as f:
                pickle.dump([
                    self.traj_a.mean(axis=(0,1)),
                    self.traj_a.std(axis=(0,1)),
                    self.traj_o.mean(axis=(0,1)),
                    self.traj_o.std(axis=(0,1)),
                ], f)
        with open(param_path, mode='rb') as f:
            self.a_mean, self.a_std, self.o_mean, self.o_std = pickle.load(f)

        self.traj_a = ((self.traj_a - self.a_mean) / self.a_std)
        self.traj_o = ((self.traj_o - self.o_mean) / self.o_std)

    def __len__(self):
        return len(self.traj_a)

    def __getitem__(self, idx):
        # TODO: k previous frames input
        t = np.random.randint(self.H - (self.T+1))
        x = self.traj_o[idx, t  :t+self.T+1]
        a = self.traj_a[idx, t+1:t+self.T+1]

        x = np.transpose(x, [0,1])
        x_0, x = x[0], x[1:]
        return x_0, x, a


class MyDataLoader(DataLoader):
    def __init__(self, mode, epoch, args):
        self.mode = mode
        SEED = args.seed
        np.random.seed(SEED)

        dataset = MyDataset(
            mode=mode,
            epoch=epoch,
            args=args,
        )
        super(MyDataLoader, self).__init__(dataset,
                                           batch_size=args.B,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=4,
                                           pin_memory=True)