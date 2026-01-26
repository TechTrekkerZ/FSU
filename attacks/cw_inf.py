import torch
import torch.nn as nn
import numpy as np

from .step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


class CWLoss(nn.Module):
    def __init__(self, num_classes=10, margin=50, reduced=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduced = reduced
        return

    def forward(self, logits, targets):
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduced:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


class CWLinf_attack(object):
    def __init__(self, model, num_steps=100, eps=8/255, num_classes=10, setp_size=None):
        self.model = model
        self.eps = eps
        self.setp_size = setp_size
        self.num_steps = num_steps
        self.num_classes = num_classes

    def perturb(self, x, y=None):
        """Virtual method for generating the adversarial examples.

        :param x: the model's input tensor.
        :param **kwargs: optional parameters used by child classes.
        :return: adversarial examples.
        """
        # error = "Sub-classes must implement perturb."
        # raise NotImplementedError(error)

        # def batch_adv_attack(args, model, x, target):
        orig_x = x.clone().detach()
        step = STEPS['Linf'](orig_x, self.eps, self.setp_size)

        @torch.enable_grad()
        def get_adv_examples(x):
            for _ in range(self.num_steps):
                x = x.clone().detach().requires_grad_(True)
                logits = self.model(x)
                loss = -1 * CWLoss(self.num_classes)(logits, y)
                grad = torch.autograd.grad(loss, [x])[0]
                with torch.no_grad():
                    x = step.step(x, grad)
                    x = step.project(x)
                    x = torch.clamp(x, 0, 1)
            return x.clone().detach()

        x = step.random_perturb(x)
        x = torch.clamp(x, 0, 1)
        adv = get_adv_examples(x)
        to_ret = adv.detach()

        return to_ret.detach().requires_grad_(False)


    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)





