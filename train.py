# -*- coding: utf-8 -*-

"""train.py
Example:
python train.py --dataset tinyimagenet --alpha 0.5 --beta 1.0 \
  --base-ckpt /path/to/TRADES/tinyimagenet/model_best.pt \
  --save-dir ./runs/trades_fsu_ft
"""

import os
import time
import copy
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from losses.trades import trades_loss
from datasets.load_dataset import data_loader
from models.wideresnet_trades_for_tiny import ResNet18 as ResNet_for_tiny


def num_classes(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    if dataset == "tinyimagenet":
        return 200
    return 10


def strip_module(sd: dict) -> dict:
    # compatible with DataParallel checkpoints
    return {k.replace("module.", ""): v for k, v in sd.items()}


def adjust_lr(args, optimizer, epoch: int):
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_one_epoch(args, model, device, train_loader, optimizer, epoch: int):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        loss = trades_loss(
            model=model,
            x_natural=x,
            y=y,
            optimizer=optimizer,
            step_size=float(args.step_size),
            epsilon=float(args.epsilon),
            perturb_steps=int(args.num_steps),
            beta=float(args.beta_trades),
        )
        loss.backward()
        optimizer.step()

        # if batch_idx % args.log_interval == 0:
        print(
            f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
            f"({100. * batch_idx / len(train_loader):.0f}%)]  Loss: {loss.item():.6f}"
        )


@torch.no_grad()
def eval_acc(model, device, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main():
    p = argparse.ArgumentParser("TRADES fine-tuning")
    p.add_argument("--dataset", type=str, default="tinyimagenet")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)

    # training
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--start-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--log-interval", type=int, default=100)

    # TRADES
    p.add_argument("--epsilon", type=float, default=8/255)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=2/255)
    p.add_argument("--beta-trades", type=float, default=6.0, dest="beta_trades")

    # IO
    p.add_argument("--base-ckpt", type=str, required=True, help="checkpoint to finetune from")
    p.add_argument("--save-dir", type=str, default="./runs/trades_ft")

    # misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-cuda", action="store_true", default=False)
    args = p.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # data
    ds = data_loader()
    train_loader, test_loader = ds(args.dataset, normalize=False, test_batch=args.test_batch_size, train_batch=args.batch_size)

    # model
    alpha_beta = [args.alpha, args.beta]
    model = ResNet_for_tiny(alpha_beta=alpha_beta, num_classes=num_classes(args.dataset))
    sd = strip_module(torch.load(args.base_ckpt, map_location="cpu"))
    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc, best_state = -1.0, None

    t0 = time.time()
    for epoch in range(args.start_epoch+1, args.epochs+1):
        adjust_lr(args, optimizer, epoch)
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)

        _, tr_acc = eval_acc(model, device, train_loader)
        _, te_acc = eval_acc(model, device, test_loader)
        print(f"[Epoch {epoch}] Train acc={tr_acc*100:.2f}%  Test acc={te_acc*100:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()

    out = os.path.join(args.save_dir, "model_best.pt")
    torch.save(best_state, out)
    print(f"[DONE] saved: {out}  best_test_acc={best_acc*100:.2f}%  time={int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
