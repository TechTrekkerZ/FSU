## Mitigating Adversarial Shifts in Feature Statistics with Uncertainty-Aware Reconstruction

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.12
- torch = 1.8.1
- torchvision = 0.9.1

## What is in this repository
 - Codes for natural training with feature statistics with uncertainty module (FSU-NT) and the FSU with fine-tuning strategy in adversarial training (Will be released when the manuscript is accepted)
   
 - Well-trained models optimized with FSU are shown in the following links:

 <table>
  <!-- 表格行标签 -->
  <tr>
   <th>(&alpha;, &beta;)</th>
   <th>Model</th>
   <th>Link</th>
  </tr>
  
  <tr>
   <td colspan="3" align='center'>MNIST</td>
  </tr>
  <tr>
   <td>(1,  1)</td>
   <td>FSU-NT</td>
   <td>https://drive.google.com/file/d/1A-wwqIqADjPqccssmqyYZ4zf3G74lmaU/view?usp=sharing</td>
  </tr>
  <tr>
   <td>(0.3,  1)</td>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1pUOnEjCy-FIPE353KER7Q_8gmRb9aQhz/view?usp=sharing</td>
  </tr>

 </table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL

[5] DKL: https://github.com/jiequancui/DKL/tree/main/DKLv1

[6] RiFT: https://github.com/microsoft/robustlearn/tree/main/RiFT
