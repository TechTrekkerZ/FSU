import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class DistributionUncertainty(nn.Module):
    def __init__(self, alpha_beta, plot_path=None, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.cln_mean = None
        self.cln_std = None
        self.plot_path = plot_path
        self.alpha_beta = alpha_beta
        self.factor = 1.0
        if alpha_beta is not None:
            self.alpha = alpha_beta[0]
            self.beta = alpha_beta[1]

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def conditional_sampling(self, deltas):

        if deltas.dim() == 2:
            sampled_values = torch.randn_like(deltas)

            pos_mask = deltas > 0
            neg_mask = deltas < 0

            while True:
                resample = torch.randn_like(deltas)
                sampled_values[pos_mask] = torch.where(resample[pos_mask] > 0, resample[pos_mask], sampled_values[pos_mask])
                if torch.all(sampled_values[pos_mask] > 0):
                    break

            while True:
                resample = torch.randn_like(deltas)
                sampled_values[neg_mask] = torch.where(resample[neg_mask] < 0, resample[neg_mask], sampled_values[neg_mask])
                if torch.all(sampled_values[neg_mask] < 0):
                    break

            return sampled_values

        else:
            batch_size = deltas.shape[0]
            sampled_values = torch.randn(batch_size)
            pos_mask = []
            neg_mask = []

            for i in range(batch_size):
                sample = deltas[i]
                positive_count = (sample > 0).sum().item()
                negative_count = (sample < 0).sum().item()

                if positive_count > negative_count:
                    pos_mask.append(True)
                    neg_mask.append(False)
                else:
                    pos_mask.append(False)
                    neg_mask.append(True)

            while True:
                resample = torch.randn(batch_size)
                sampled_values[pos_mask] = torch.where(resample[pos_mask] > 0, resample[pos_mask], sampled_values[pos_mask])
                if torch.all(sampled_values[pos_mask] > 0):
                    break

            while True:
                resample = torch.randn(batch_size)
                sampled_values[neg_mask] = torch.where(resample[neg_mask] < 0, resample[neg_mask], sampled_values[neg_mask])
                if torch.all(sampled_values[neg_mask] < 0):
                    break

            sampled_values = sampled_values.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            sampled_values = sampled_values.expand(deltas.shape)
            device = deltas.device
            return sampled_values.to(device)

    def _reparameterize_with_cln(self, mu, sstd, t=None):

        if t == 'mean':
            delta = mu - self.cln_mean
            sampled_values = self.conditional_sampling(delta)
            epsilon = sampled_values * self.alpha
        elif t == 'std':
            delta = mu - self.cln_std
            sampled_values = self.conditional_sampling(delta)
            epsilon = sampled_values * self.beta
        else:
            raise ValueError('Please choose tage from mean and std.')

        return mu + epsilon * sstd

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, get_cln_stati=False, plot_paras=None, step=None):
        if np.random.random() > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        if plot_paras is not None:

            mean_1dim = torch.mean(mean, dim=1)
            std_1dim = torch.mean(std, dim=1)

            add_dic = {'tage': plot_paras[0], 'attack': plot_paras[1],
                       'mean_1dim': mean_1dim.detach().cpu().numpy().tolist(),
                       'std_1dim': std_1dim.detach().cpu().numpy().tolist()}

            if step is None:
                plot_path = plot_paras[2]
            elif step == 1 or (step % 2 == 0 and step <= 10) or step % 25 == 0:
                plot_path = f"{plot_paras[2][0]}/pgd100_step_{step}_mean_std.jsonl"
            else:
                return x

            with open(plot_path, 'a', encoding='utf-8') as f:
                json.dump(add_dic, f)
                f.write('\n')

            if step == 100:
                with open(plot_paras[2][1], 'a', encoding='utf-8') as f:
                    json.dump(add_dic, f)
                    f.write('\n')

            return x


        if get_cln_stati:
            self.cln_mean = mean
            self.cln_std = std

            return x

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        if self.alpha_beta is None:
            beta = self._reparameterize(mean, sqrtvar_mu)
            gamma = self._reparameterize(std, sqrtvar_std)
        else:
            beta = self._reparameterize_with_cln(mean, sqrtvar_mu, t='mean')
            gamma = self._reparameterize_with_cln(std, sqrtvar_std, t='std')

        x_temp = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        reshape_x = x_temp * gamma.reshape(x_temp.shape[0], x_temp.shape[1], 1, 1) + beta.reshape(x_temp.shape[0],x_temp.shape[1], 1, 1)
        return reshape_x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, alpha_beta, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.proxy_tage = False
        self.fsu_state = False
        self.alpha_beta = alpha_beta
        # self.resampling_strat = resampling_strat
        self.get_cln_stati = False
        self.plot_paras = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.pertubration = DistributionUncertainty(self.alpha_beta,
                                                    p=1) if DistributionUncertainty else nn.Identity()  # uncertainty = 1


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_parameters(self, parameter, parameter_name):
        if parameter_name == 'fsu_state':
            self.fsu_state = parameter
        elif parameter_name == 'alpha_beta':
            self.alpha_beta = parameter
        elif parameter_name == 'resampling_strat':
            self.resampling_strat = parameter
        elif parameter_name == 'get_cln_stati':
            self.get_cln_stati = parameter
        elif parameter_name == 'plot_paras':
            self.plot_paras = parameter

    def forward(self, x, step=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.fsu_state or self.plot_paras is not None:
            out = self.pertubration(out, self.get_cln_stati, plot_paras=self.plot_paras, step=step)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(alpha_beta, num_classes=10):
    return ResNet(alpha_beta, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
