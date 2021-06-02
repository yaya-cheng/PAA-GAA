# Feature Space Targeted Attacks by Statistic Alignment (accpeted by IJCAI2021)

## This repo is the official **Pytorch code** implementation of our paper [Feature Space Targeted Attacks by Statistic Alignment](https://arxiv.org/pdf/2105.11645.pdf).

## Condensed Abstract
By adding human-imperceptible perturbations to images, DNNs can be easily fooled. As one of the mainstream methods, feature space targeted attacks perturb images by modulating their intermediate feature maps, for the discrepancy between the intermediate source and target features is minimized. However, the current choice of pixel-wise Euclidean Distance to measure the discrepancy is questionable because it unreasonably imposes a spatial-consistency constraint on the source and target features. Intuitively, an image can be categorized as “cat” no matter the cat is on the left or right of the image. To address this issue, we propose to measure this discrepancy using statistic alignment. Specifically, we design two novel approaches called Pair-wise Alignment Attack and Global-wise Alignment Attack, which attempt to measure similarities between feature maps by high-order statistics with translation invariance. Furthermore, we systematically analyze the layerwise transferability with varied difficulties to obtain highly reliable attacks. Extensive experiments verify the effectiveness of our proposed method, and it outperforms the state-of-the-art algorithms by a large margin.

## Effectiveness of our PAA and GAA
![image](https://github.com/yaya-cheng/PAA-GAA/blob/main/class.png)

tSuc and tTR performance w.r.t. relative layer depth for multiple transfer scenarios. The figure is split into four phases, corresponding to black-box attacks transferring from Den121, Inc-v3, VGG19, and Res50. All of our proposed methods outperform AA in most cases, which indicates the effectiveness of statistic alignment on various layers.


## Visualization of the adversarial examples
![image](https://github.com/yaya-cheng/PAA-GAA/blob/main/visualization%20of%20adversarial%20examples/all.png)

Visualization of adversarial examples with Den121 as the white-box. Original class: goldfish, targeted class: axolotl. Fromleft to right: Raw, MIFGSM, AA and PAAp.

- Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) (DenseNet can be found in [here](https://github.com/flyyufelix/DenseNet-Keras))
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf)

- Then put these models into ".models/"

- Run PAAp on Den121 under 2nd:

  ```python
  python fp_attack_den121.py --method 1 --kernel_type poly --kernel_for_furthe l_poly --byRank 1 --targetcls 2 
  ```

- Run GAA on Den121 under 2nd:

  ```python
  python fp_attack_den121.py --mmdMethod 2 --GAA 1 --byRank 1 --targetcls 2 
  ```

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{Cheng2021StatisticAlignment,
    title={Feature Space Targeted Attacks by Statistic Alignment},
    author={Gao, Lianli and Cheng, Yaya and Zhang, Qilong and Xu, Xing and Song, Jingkuan},
    Booktitle = {International Joint Conferences on Artificial Intelligence},
    year={2021}
}
```

