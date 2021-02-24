import argparse
import os
import logzero
import subprocess
from datetime import datetime
import torch

from model import SSM
from trainer import Trainer
from dataloader import MyDataLoader
from utils import save_model, load_model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--H", type=int, default=100)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--s_dim", type=int, default=64)
    parser.add_argument("--a_dim", type=int, default=6)
    parser.add_argument("--o_dim", type=int, default=17)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--load_epoch", type=int, default=None)
    if not args is None:
        args = parser.parse_args(args.split())
    else:
        args = parser.parse_args()
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # logger
    os.makedirs("logzero", exist_ok=True)
    logzero.loglevel(20)
    logzero.logfile(os.path.join("logzero", args.timestamp + ".txt"), loglevel=20)
    logzero.logger.info("args: " + str(args))
    
    # main
    model = SSM(args)
    train_loader = MyDataLoader("train", args.depth, args)
    test_loader = MyDataLoader("test", args.depth, args)
    trainer = Trainer(model,
                      train_loader,
                      test_loader,
                      args)

    if args.load_epoch:
        resume_epoch = args.load_epoch + 1
        load_model(model, args.load_epoch)
    else:
        resume_epoch = 1

    for epoch in range(resume_epoch, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)

        if epoch % 1 == 0:
            save_model(model, epoch)
            load_model(model, epoch)

    return model, os.path.join("weights", model.args.timestamp)

if __name__ == "__main__":
    main()