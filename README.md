## Feature Statistics with Uncertainty Help Adversarial Robustness

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.12
- torch = 1.8.1
- torchvision = 0.9.1

## What is in this repository
 - Codes for our feature statistics with uncertainty module in standard training (FSU-ST) and the FSU with fine-tuning strategy in adversarial training (FSU-FiT) (Will be released when the manuscript is accepted)
   
 - Well-trained models optimized by FSU-ST are shown in the following links:

 <table>
  <caption></caption>
  <!-- 表格行标签 -->
  <tr>
   <th>Model</th>
   <th>Link</th>
  </tr>
  
  <tr>
   <td colspan="2" align='center'>MNIST</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://drive.google.com/file/d/16vf1gZeq9YDazOpCDFcwTdOnjlOvnI46/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1hpWrtGcx6IUG9En7JRuJEN3A57-REytD/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1PZ50SJvvdOSEAK_ifg31F_vNk-QXf3N3/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/13oOrsoLUPt-amSwALs4EbMryWOXuQyw0/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1BTHtbWHDiFePfM6llkBje_9cPy-MNkUI/view?usp=drive_link</td>
  </tr>
  
  <tr>
   <td colspan="2" align='center'>CIFAR10</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://drive.google.com/file/d/1T1XFR3ApKj70gT4uka4rB8dk1JflbYMl/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1kKBYzkpAOT9IZ6SU89U-dkLGtKKcOUex/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/16qlpAvmBtzPkGVyGwaVvun-bEEuEu-TQ/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/17PMKvg1VRXuhpl7ClPFkuLAjsER9kO90/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1HwfO-mrQmnytJ5GFLuT4g4wls2Jzwbdl/view?usp=drive_link</td>
  </tr>

  <tr>
   <td colspan="2" align='center'>SVHN</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://drive.google.com/file/d/17GGyjG9A7DoroSV-L_2SEYHRlKB0RyPU/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1uAKCMlIO0xgrbC5wqSFdzndutlzvY0-W/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1p1PQ2LbbMyZwY0BWJM_rDCmgWHT2GROl/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1UtwDJOAijl8QC-HZ6k7xq2GeJP0ZAso_/view?usp=drive_link</td>
  </tr>
  
  <tr>
   <td colspan="2" align='center'>CIFAR100</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://drive.google.com/file/d/1jbNQJpDBUCqRI54U-cyyw3RvBKh-9v9x/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1vBZFh4VoHQkV1hp1l-Nx_7PIZLCn51C_/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1vv0CaoK-k-MUknNZIJZues7bdsJRcV-S/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1unCkah2--tuYTI2ZozLuxjBjQlpA_vLE/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1mED6QzuSufDBXKURCl2Hr2l_Pqe-Ty6G/view?usp=drive_link</td>
  </tr>
 </table>

 - Well-trained <b>MAIL-TRADES+FSU-FiT</b> models for CIFAR10 are shown in the following links:

<table>
  <caption></caption>
  <!-- 表格行标签 -->
  <tr>
   <th>(&alpha;, &beta;)</th>
   <th>Link</th>
  </tr>
  <tr>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1viUYcKgdHPNQFCvMXq5h4MNxGucyNwNz/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1v1-s_HCnym0PNowSPcbo4vbXwwEYBmoX/view?usp=drive_link</td>
  </tr>

</table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL
