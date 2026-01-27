import re
import tempfile
import argparse
import torch

from datasets.load_dataset import data_loader
from attacks.attack import attack
from attacks.cw_inf import CWLinf_attack
from models.wideresnet_trades_for_tiny import ResNet18 as ResNet_for_tiny

from advertorch import attacks as adv_attacks
import torchattacks
from autoattack import AutoAttack as AA_bench
from autoattack.fab_pt import FABAttack


def get_num_classes(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    if dataset == "tinyimagenet":
        return 200
    return 10


def strip_module(state: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state.items()}


def load_trades_model(ckpt_path: str, dataset: str, alpha_beta=[0.5, 1.0], device="cuda"):
    model = ResNet_for_tiny(alpha_beta=alpha_beta, num_classes=get_num_classes(dataset))
    state = torch.load(ckpt_path, map_location="cpu")
    state = strip_module(state)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def clean_acc(model, loader, device="cuda"):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def autoattack_acc(model, loader, device="cuda", eps=8 / 255, bs=128):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs, 0).to(device)
    y_all = torch.cat(ys, 0).to(device)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as f:
        log_path = f.name

    adversary = AA_bench(model=model, norm="Linf", eps=eps, log_path=log_path)
    adversary.run_standard_evaluation(x_all, y_all, bs=bs)

    adv_acc = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in reversed(f.readlines()):
            if "robust accuracy:" in line:
                m = re.search(r"robust accuracy:\s*([0-9.]+)%", line)
                if m:
                    adv_acc = float(m.group(1))
                    break
    return adv_acc if adv_acc is not None else float("nan")


def eval_all_attacks(model, test_loader, dataset, device="cuda", eps=8 / 255, eps_iter=2 / 255, bs=128):
    C = get_num_classes(dataset)
    res = {}

    res["CLN"] = clean_acc(model, test_loader, device)

    fgsm = adv_attacks.FGSM(model, eps=eps)
    _, res["FGSM"] = attack(fgsm, model, test_loader, device, fp=None)

    bim = adv_attacks.LinfBasicIterativeAttack(model, eps=eps, eps_iter=eps_iter)
    _, res["BIM"] = attack(bim, model, test_loader, device, fp=None)

    mim = adv_attacks.LinfMomentumIterativeAttack(model, eps=eps, eps_iter=eps_iter)
    _, res["MIM"] = attack(mim, model, test_loader, device, fp=None)

    for it in (20, 40, 100):
        pgd = adv_attacks.LinfPGDAttack(model, eps=eps, nb_iter=it, eps_iter=eps_iter)
        _, res[f"PGD{it}"] = attack(pgd, model, test_loader, device, fp=None)

    cw_inf = CWLinf_attack(model, num_steps=100, eps=eps, setp_size=eps_iter, num_classes=C)
    _, res["CW_inf"] = attack(cw_inf, model, test_loader, device, fp=None, cw=True)

    cw_l2 = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    _, res["CW_l2_s50"] = attack(cw_l2, model, test_loader, device, fp=None, cw=True)

    try:
        fab = FABAttack(model, n_restarts=5, n_iter=100, eps=eps, seed=0, norm="Linf",
                        verbose=False, device=device)
        _, res["FAB"] = attack(fab, model, test_loader, device, fp=None)
    except Exception:
        res["FAB"] = float("nan")

    res["AA"] = autoattack_acc(model, test_loader, device=device, eps=eps, bs=bs)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="tinyimagenet")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    dataset = args.dataset
    ckpt = args.ckpt
    alpha_beta = [args.alpha, args.beta]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = args.batch_size

    eps, eps_iter = (0.3, 2 / 255) if dataset == "mnist" else (8 / 255, 2 / 255)

    data_sel = data_loader()
    _, test_loader = data_sel(dataset, normalize=False, test_batch=bs, train_batch=bs)

    model = load_trades_model(ckpt, dataset, alpha_beta=alpha_beta, device=device)

    results = eval_all_attacks(model, test_loader, dataset, device=device, eps=eps, eps_iter=eps_iter, bs=bs)

    print(f"[TEST] dataset={dataset}, ckpt={ckpt}")
    for k in ["CLN", "FGSM", "BIM", "MIM", "PGD20", "PGD40", "PGD100", "CW_inf", "CW_l2_s50", "FAB", "AA"]:
        v = results.get(k, float("nan"))
        print(f"  {k:10s}: {v:.2f}%")
