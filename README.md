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
   <td>https://XXXX</td>
  </tr>
  <tr>
   <td>TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1VFUYkle2_GlAxz6qdAgSdjiut3nXP5gG/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/13PdN0CHJDWtrVd78hwxTa8sI5pjHYX70/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/12bQMvIiMBj7qhLrPwLk_q5sVOng-EGEA/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1Mw4rPFO3ZzxsOJnMLk-yj8kdpzQcEHTv/view?usp=drive_link</td>
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
   <td>https://drive.google.com/file/d/1OjhUVZ4k_9tp-xf4kmEjPDvc26zpifQH/view?usp=drive_link</td>
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
   <td>https://drive.google.com/file/d/1DZMjIOpO07776-wtKYKb7T4-VjFCxVN8/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/14RXFKWXVhmx9Wlb3EUEkaUvrgC2-Jzf7/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/1t7lStxpRjpRe7BeCE7SwPyuv633KZm9o/view?usp=drive_link</td>
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
   <td>https://drive.google.com/file/d/1_o4FoxAX70WpxGKDWkEWky9inT2WJtTm/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>AT-AWP+FSU</td>
   <td>https://drive.google.com/file/d/1pT7JcuS9IyE8IkM8gQMyqoM7dtADHeUT/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MLCATWP+FSU</td>
   <td>https://drive.google.com/file/d/1njuT1heqTlYoQTiPkYP6DXLs6293cPG7/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>MAIL-TRADES+FSU</td>
   <td>https://drive.google.com/file/d/17Q0h9DlRksJ2Fmxba8iTg8RgKJCNiP7c/view?usp=drive_link</td>
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
   <td>https://drive.google.com/file/d/1Ni1G1xFtYCa5_C4QVhhiciXBZx0lmP2p/view?usp=drive_link</td>
  </tr>
  <tr>
   <td>(0.5, 1)</td>
   <td>https://drive.google.com/file/d/1ur47VTwVVLuG33uL-NwXHI7cpCdPPNLm/view?usp=drive_link</td>
  </tr>

</table>

## Reference Code
[1] TRADES: https://github.com/yaodongyu/TRADES

[2] AT-AWP: https://github.com/csdongxian/AWP

[3] MLCATWP: https://github.com/ChaojianYu/Understanding-Robust-Overfitting

[4] MAIL: https://github.com/QizhouWang/MAIL
