## View-Robust Backbone and Discriminative Reconstruction for Few-Shot Fine-Grained Image Classification

### abstract

We study few-shot fine-grained image classification, a task that faces two key challenges: (1) the scarcity of labeled samples amplifies the model’s sensitivity to viewpoint variations, resulting in feature inconsistency, and (2) reconstruction-based methods, while improving inter-class separability, inadvertently introduce intra-class variations, further complicating discrimination.To address these challenges, we propose the View-Robust Attention Selector (VRAS), a feature enhancement backbone designed to mitigate viewpoint-induced misclassifications. By integrating cross-scale feature interaction and adaptive selection mechanisms, VRAS effectively reduces spatial sensitivity arising from the limited viewpoint diversity in few-shot support sets. This approach not only preserves intra-class consistency but also enhances inter-class discriminability, ensuring robust feature representations.Furthermore, we introduce the Enhancement and Reconstruction (ER) module, designed to strengthen discriminative learning. ER achieves this by maximizing inter-class divergence while enhancing intra-class compactness through a regularized Ridge Regression optimization strategy. By dynamically suppressing low-saliency dimensions, ER maintains geometric coherence and effectively filters out semantic noise.Extensive experiments on three fine-grained datasets show that our method significantly outperforms state-of-the-art few-shot classification methods.

## View-Robust Backbone

* [VRAS-Conv-4](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-Conv4.py)
* [VRAS-ResNet-12](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-ResNet12.py)

## Experiment






## Code environment

You can create a conda environment with the correct dependencies using the following command lines:

```
conda env create -f environment.yml
conda activate VRAS
```

## 

## Train and test

For fine-grained few-shot classification, we provide the training and inference code

For CUB dataset,

`python experiments/CUB/VRAS-Conv-4/train.py`

`python experiments/CUB/VRAS-ResNet-12/train.py`

For Cars dataset,

`python experiments/cars/VRAS-Conv-4/train.py`

`python experiments/cars/VRAS-ResNet-12/train.py`

For Dogs dataset,

`python experiments/dogs/VRAS-Conv-4/train.py`

`python experiments/dogs/VRAS-ResNet-12/train.py`
