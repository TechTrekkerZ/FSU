## Feature Statistics with Uncertainty Help Adversarial Robustness

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.12
- torch = 1.8.1
- torchvision = 0.9.1

## What is in this repository
 - Codes for natural training with feature statistics with uncertainty module (FSU-NT) and the FSU with fine-tuning strategy in adversarial training (Will be released when the manuscript is accepted)
   
 - Well-trained models optimized with FSU are shown in the following links:

 <table>
  <caption></caption>
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
   <td>(/, /)</td>
   <td></td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.1, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.9, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.7, 1)</td>
   <td></td>
  </tr>

  
  <tr>
   <td colspan="3" align='center'>CIFAR10</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(/, /)</td>
   <td></td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.1, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.7, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.7, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.3, 1)</td>
   <td></td>
  </tr>

  <tr>
   <td colspan="3" align='center'>SVHN</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(/, /)</td>
   <td></td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.9, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.3, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.1, 1)</td>
   <td></td>
  </tr>
  
  <tr>
   <td colspan="3" align='center'>CIFAR100</td>
  </tr>
  <tr>
   <td>FSU-NT</td>
   <td>(/, /)</td>
   <td></td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>(0.3, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>(0.7, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>(0.9, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>AT+RiFT+FSU</td>
   <td>(0.5, 1)</td>
   <td></td>
  </tr>
  <tr>
   <td>DKL+FSU</td>
   <td>(0.1, 1)</td>
   <td></td>
  </tr>
 </table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL
