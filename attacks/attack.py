
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
import os
# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into files
    with open(log_path, 'a+') as f:
        f.write(log_info + '\n')

def attack_trans_black_box_self_generate(proxy, model, device, fp=None, flag_adv=True):
    """
    对抗攻击
    :param draw_adv: 显示对抗样本
    :param adver_method: advertorch中的对抗攻击方法
    :param model: 训练好的模型
    :param data_loader: 数据迭代器
    :param flag_adv: 是否进行对抗攻击，True时攻击
    :return:
    """
    H = 0
    H_adv = 0
    test_clnloss = 0
    clncorrect = 0
    robustness = 0
    nums = 0

    # data_path = "/media/data/zhouxnli/gxd/image_data/trans_blackAttack_advData_cifar10"
    # data_path = "/media/data/zhouxnli/gxd/image_data/trans_blackAttack_advData_cifar10/cifar10"

    if flag_adv:
        test_advloss = 0
        advcorrect = 0

    for batch_id in tqdm(range(10)):

        advdata = torch.load(os.path.join('/media/data/zhouxnli/gxd/image_data/trans_blackAttack_advData_cifar10/cifar10/', f'model_{proxy}_batch_{batch_id}.pt'))
        data, target = advdata['img'].to(device), advdata['label'].to(device)

        with torch.no_grad():
            output_adv = model(data)
        H_adv_i = - F.softmax(output_adv, 1) * F.log_softmax(output_adv, 1)
        H_adv_i = H_adv_i.detach().cpu().numpy()
        H_adv_batch = np.sum(np.sum(H_adv_i, 1))
        H_adv += H_adv_batch

        test_advloss += F.cross_entropy(
            output_adv, target, reduction='sum').item()
        advpred = output_adv.max(1, keepdim=True)[1]
        advcorrect += advpred.eq(target.view_as(advpred)).sum().item()

        nums += len(target)

        # if batch_id % 20 == 0:
        #     print(f"Batch {batch_id} is done")

    test_clnloss /= nums

    H /= nums
    H_adv /= nums

    print_log('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
        test_advloss, advcorrect, nums,
        100. * advcorrect / nums), fp)


def attack_trans_black_box(model, device, fp=None, flag_adv=True):
    """
    对抗攻击
    :param draw_adv: 显示对抗样本
    :param adver_method: advertorch中的对抗攻击方法
    :param model: 训练好的模型
    :param data_loader: 数据迭代器
    :param flag_adv: 是否进行对抗攻击，True时攻击
    :return:
    """
    H = 0
    H_adv = 0
    test_clnloss = 0
    clncorrect = 0
    robustness = 0
    nums = 0

    data_path = "/media/data/zhouxnli/gxd/image_data/trans_blackAttack_advData_cifar10"

    if flag_adv:
        test_advloss = 0
        advcorrect = 0


    for idx in range(313):
        advdata_path = os.path.join(data_path, f'batch_{idx}.pt')
        advdata = torch.load(advdata_path, map_location=device)

        data, target = advdata['img'].to(device), advdata['label'].to(device)

        with torch.no_grad():
            output_adv = model(data)
        H_adv_i = - F.softmax(output_adv, 1) * F.log_softmax(output_adv, 1)
        H_adv_i = H_adv_i.detach().cpu().numpy()
        H_adv_batch = np.sum(np.sum(H_adv_i, 1))
        H_adv += H_adv_batch

        test_advloss += F.cross_entropy(
            output_adv, target, reduction='sum').item()
        advpred = output_adv.max(1, keepdim=True)[1]
        advcorrect += advpred.eq(target.view_as(advpred)).sum().item()

        nums += len(target)

        if idx % 20 == 0:
            print(f"Batch {idx} is done")

    test_clnloss /= nums

    H /= nums
    H_adv /= nums

    print_log('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
        test_advloss, advcorrect, nums,
        100. * advcorrect / nums), fp)


def attack(adver_method, model, data_loader, device, fp=None, flag_adv=True, draw_adv=False, blackbox=False, cw=False, batch_size=None, svhn_tage=False, plot_paras=False):
    """
    对抗攻击
    :param draw_adv: 显示对抗样本
    :param adver_method: advertorch中的对抗攻击方法
    :param model: 训练好的模型
    :param data_loader: 数据迭代器
    :param flag_adv: 是否进行对抗攻击，True时攻击
    :return:
    """
    H = 0
    H_adv = 0
    test_clnloss = 0
    clncorrect = 0
    robustness = 0

    if flag_adv:
        test_advloss = 0
        advcorrect = 0

    for clndata, target in tqdm(data_loader):
        clndata, target = clndata.to(device), target.to(device)


        # model.model.set_parameters(parameter=True, parameter_name='get_cln_stati')
        with torch.no_grad():
            if not flag_adv:
                output = model(clndata)
                # output = model(clndata, plot_paras=plot_paras)
            else:
                output = model(clndata)

        # model.model.set_parameters(parameter=False, parameter_name='get_cln_stati')

        # 计算输出熵
        H_i = - F.softmax(output, 1) * F.log_softmax(output, 1)
        H_i = H_i.detach().cpu().numpy()
        H_batch = np.sum(np.sum(H_i, 1))
        H += H_batch

        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        clnpred = output.max(1, keepdim=True)[1]
        clncorrect += clnpred.eq(target.view_as(clnpred)).sum().item()

        if flag_adv:
            adversary = adver_method
            if blackbox or cw:
                advdata = adversary(clndata, target)
            elif batch_size is not None:  # AA
                advdata = adversary.run_standard_evaluation(clndata, target, bs=batch_size)
            else:
                # advdata = adversary.perturb(clndata)
                advdata = adversary.perturb(clndata, target)

            if draw_adv:
                advdata_n = advdata / 2 + 0.5
                grid = make_grid(advdata_n)
                tb.add_image('adv images', grid)

            with torch.no_grad():
                output_adv = model(advdata)
                # output_adv = model(advdata, plot_paras=plot_paras)
            H_adv_i = - F.softmax(output_adv, 1) * F.log_softmax(output_adv, 1)
            H_adv_i = H_adv_i.detach().cpu().numpy()
            H_adv_batch = np.sum(np.sum(H_adv_i, 1))
            H_adv += H_adv_batch

            test_advloss += F.cross_entropy(
                output_adv, target, reduction='sum').item()
            advpred = output_adv.max(1, keepdim=True)[1]
            advcorrect += advpred.eq(target.view_as(advpred)).sum().item()
            robustness += advpred.eq(clnpred).sum().item()

    if draw_adv:
        plt.figure(1)
        clndata = clndata / 2 + 0.5
        img_clean = clndata[0].cpu().detach().numpy()
        img_clean = img_clean.transpose(1, 2, 0)
        plt.title('clean image')
        plt.imshow(img_clean, cmap='gray')

        plt.figure(2)
        advdata = advdata / 2 + 0.5
        img_adv = advdata[0].cpu().detach().numpy()
        img_adv = img_adv.transpose(1, 2, 0)
        plt.title('adv image')
        plt.imshow(img_adv, cmap='gray')

        plt.show()

    test_clnloss /= len(data_loader.dataset)

    H /= len(data_loader.dataset)
    H_adv /= len(data_loader.dataset)

    if not flag_adv:
        print_log('\nTest set: avg cln loss: {:.4f},'
                  ' cln acc: {}/{} ({:.0f}%)\n'.format(
            test_clnloss, clncorrect, len(data_loader.dataset),
            100. * clncorrect / len(data_loader.dataset)), fp)

        if svhn_tage:
            return 10000. * clncorrect / len(data_loader.dataset)
        else:
            return clncorrect

    if fp:
        print_log('\nTest set: avg cln loss: {:.4f},'
                      ' cln acc: {}/{} ({:.0f}%)\n'.format(
            test_clnloss, clncorrect, len(data_loader.dataset),
            100. * clncorrect / len(data_loader.dataset)), fp)

        print_log('Test set: avg adv loss: {:.4f},'
                      ' adv acc: {}/{} ({:.0f}%)\n'.format(
            test_advloss, advcorrect, len(data_loader.dataset),
            100. * advcorrect / len(data_loader.dataset)), fp)

        print_log('Test set: avg adv loss: {:.4f},'
                  ' adv robustness: {}/{} ({:.0f}%)\n'.format(
            test_advloss, robustness, len(data_loader.dataset),
            100. * robustness / len(data_loader.dataset)), fp)

        print_log(f'\nClean test entropy = {H}\n', fp)



    if svhn_tage:
        return 10000. * clncorrect / len(data_loader.dataset), 10000. * advcorrect / len(data_loader.dataset)
    else:
        return clncorrect, advcorrect/100
