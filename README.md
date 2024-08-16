## Feature Statistics with Uncertainty Help Adversarial Robustness

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.9.12
- torch = 1.8.1
- torchvision = 0.9.1

## What is in this repository
 - Codes for our feature statistics with uncertainty module in standard training (FSU-ST) and the FSU with fine-tuning strategy in adversarial training (FSU-FiT) (Will be released when the manuscript is accepted)
   
 - Well-trained models optimized by adding the FSU module in standard training are shown in the following links:

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
   <td>https://XXXX</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1cpyApGH1pWe3p6v5NYlGDqZW9qAIP_i9/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1aCfk4ofMqLoqBFlrGvTlJDDFaUOh3akR/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1YvPX4XcIdxR02rPnYFk7YC5cK-FlYsgc/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1OjhUVZ4k_9tp-xf4kmEjPDvc26zpifQH/view?pli=1</td>
  </tr>
  <tr>
   <td colspan="2" align='center'>CIFAR10</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://XXXX</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1cpyApGH1pWe3p6v5NYlGDqZW9qAIP_i9/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1aCfk4ofMqLoqBFlrGvTlJDDFaUOh3akR/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1YvPX4XcIdxR02rPnYFk7YC5cK-FlYsgc/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1OjhUVZ4k_9tp-xf4kmEjPDvc26zpifQH/view?pli=1</td>
  </tr>
  <tr>
   <td colspan="2" align='center'>SVHN</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://XXXX</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1cpyApGH1pWe3p6v5NYlGDqZW9qAIP_i9/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1aCfk4ofMqLoqBFlrGvTlJDDFaUOh3akR/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1YvPX4XcIdxR02rPnYFk7YC5cK-FlYsgc/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1OjhUVZ4k_9tp-xf4kmEjPDvc26zpifQH/view?pli=1</td>
  </tr>
  <tr>
   <td colspan="2" align='center'>CIFAR100</td>
  </tr>
  <tr>
   <td>FSU-ST</td>
   <td>https://XXXX</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1cpyApGH1pWe3p6v5NYlGDqZW9qAIP_i9/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1aCfk4ofMqLoqBFlrGvTlJDDFaUOh3akR/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1YvPX4XcIdxR02rPnYFk7YC5cK-FlYsgc/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1OjhUVZ4k_9tp-xf4kmEjPDvc26zpifQH/view?pli=1</td>
  </tr>
 </table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL
