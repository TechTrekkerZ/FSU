## Mitigating Adversarial Shifts in Feature Statistics with Uncertainty-Aware Reconstruction

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.12
- torch = 1.8.1
- torchvision = 0.9.1

## What is in this repository
 - Codes for natural training with feature statistics with uncertainty module (FSU-NT) and the FSU with fine-tuning strategy in adversarial training (Will be released when the manuscript is accepted)
   
 - Well-trained models optimized with FSU are shown in the following links:

 <table style="width:100%; table-layout: fixed;">
  <colgroup>
    <col style="width: 20%;">   <!-- Model -->
    <col style="width: 320px;"> <!-- 固定第二列宽度 -->
    <col style="width: auto;">  <!-- Link -->
  </colgroup>
 
  <!-- 表格行标签 -->
  <tr>
   <th>Model</th>
   <th>(&alpha;, &beta;)</th>
   <th>Link</th>
  </tr>
  
  <tr>
   <td colspan="3" align='center'>MNIST</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(1, 1)</td>
   <td>https://drive.google.com/file/d/1A-wwqIqADjPqccssmqyYZ4zf3G74lmaU/view?usp=sharing</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1pUOnEjCy-FIPE353KER7Q_8gmRb9aQhz/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1yPHaJsEZjpV4MjLX-Lk-fm9wwTgpZY74/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.1, 1)</td>
   <td>https://drive.google.com/file/d/1AooKHsiBv18evndtYNyaHu1FBSPpIBNI/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1XzPa5VCjBF3VoDlLeEkwaHQ4tEA9BJyy/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.9, 1)</td>
   <td>https://drive.google.com/file/d/1HAvF8Q0hVF0srq9D8tlVwxi65CaC2xW8/view?usp=sharing</td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.7, 1)</td>
   <td>https://drive.google.com/file/d/1beVvNULuAPNg_eaWav10HszvPTMpuRPn/view?usp=sharing</td>
  </tr>

  
  <tr>
   <td colspan="3" align='center'>CIFAR10</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(1, 1)</td>
   <td>https://drive.google.com/file/d/1c34ApzCR--RSqgf5RVSxBcKVRNGxLQA5/view?usp=sharing</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.1, 1)</td>
   <td>https://drive.google.com/file/d/1BC3BtLsGjif7i9ZGv0nh5k9_8tXSKg4u/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/11cLooNQw-ZKiAMXjKQgvFdYB_ByeQVCC/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1dd0js95-YHnkj8U-M6JgQNs4SIuUQwHq/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.7, 1)</td>
   <td>https://drive.google.com/file/d/1i-NUW5zAOLMIBUwSjyPvR7eEjDk0Ps60/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.7, 1)</td>
   <td>https://drive.google.com/file/d/1OEedUYXbnHppMVdbUPtLEzJyAHTNhwKC/view?usp=sharing</td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1-ziNGGDzaheRRhLLmgPnDty0frb40S07/view?usp=sharing</td>
  </tr>

  <tr>
   <td colspan="3" align='center'>SVHN</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(1, 1)</td>
   <td>https://drive.google.com/file/d/1l2xNk95iV6nvKBy--3c8weMazse3ttGV/view?usp=sharing</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.9, 1)</td>
   <td>https://drive.google.com/file/d/1O1pp7yJNkDF5Ck8NDJmuuUm1T8Dd1pPK/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1dtFmcZ2GfMhbWb-Lfrt0e1M2YH3JgXM2/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1-vx_aPnmfWzt_1Vke0uwOHXv5403ymki/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/17dNovc7U3TILjIkjhgFgeDmP2sfXWmNj/view?usp=sharing</td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.1, 1)</td>
   <td>https://drive.google.com/file/d/1waT5k-tLQgnOMU9vZd_d2kFX9mFVManR/view?usp=sharing</td>
  </tr>
  
  <tr>
   <td colspan="3" align='center'>CIFAR100</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(4, 1)</td>
   <td>https://drive.google.com/file/d/1iDWUBhAxhvKNW61vHAzpHI3dQ_IP2Vns/view?usp=sharing</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td>https://drive.google.com/file/d/1lPjl7YoR2G28UXFsCyVeQ_fbkVEr7PV9/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.7, 1)</td>
   <td>https://drive.google.com/file/d/1MmJY1tp2nK6eYb_92Z2Zznzehhe2K7Zo/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1qkHmURYC1EoQtMgnWFCQR7xTX5dK-E2B/view?usp=sharing</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.9, 1)</td>
   <td>https://drive.google.com/file/d/1sLCHdlbexuixSTJl-VLp2cEqK33Q8LNb/view?usp=sharing</td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1hGrdPNllx94bAN2qw6s8em5EmZHNp6Bj/view?usp=sharing</td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.1, 1)</td>
   <td>https://drive.google.com/file/d/1si7ePX27PttqLRyTM9DlJqpwg7Uq8VCF/view?usp=sharing</td>
  </tr>
 </table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL

[5] DKL: https://github.com/jiequancui/DKL/tree/main/DKLv1

[6] RiFT: https://github.com/microsoft/robustlearn/tree/main/RiFT
