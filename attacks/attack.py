
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm

def attack(adver_method, model, data_loader, device, fp=None, flag_adv=True, draw_adv=False, blackbox=False, cw=False, batch_size=None, svhn_tage=False, plot_paras=False):

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

        with torch.no_grad():
            if not flag_adv:
                output = model(clndata)
            else:
                output = model(clndata)

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
                advdata = adversary.perturb(clndata, target)


            with torch.no_grad():
                output_adv = model(advdata)
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
