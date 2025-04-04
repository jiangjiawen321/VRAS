# View-Robust Backbone and Discriminative Reconstruction for Few-Shot Fine-Grained Image Classification

## 📄 Abstract

> We study few-shot fine-grained image classification, a task that faces two key challenges: (1) the scarcity of labeled samples amplifies the model’s sensitivity to viewpoint variations, resulting in feature inconsistency, and (2) reconstruction-based methods, while improving inter-class separability, inadvertently introduce intra-class variations, further complicating discrimination. To address these challenges, we propose the View-Robust Attention Selector (VRAS), a feature enhancement backbone designed to mitigate viewpoint-induced misclassifications. By integrating cross-scale feature interaction and adaptive selection mechanisms, VRAS effectively reduces spatial sensitivity arising from the limited viewpoint diversity in few-shot support sets. This approach not only preserves intra-class consistency but also enhances inter-class discriminability, ensuring robust feature representations. Furthermore, we introduce the Enhancement and Reconstruction (ER) module, designed to strengthen discriminative learning. ER achieves this by maximizing inter-class divergence while enhancing intra-class compactness through a regularized Ridge Regression optimization strategy. By dynamically suppressing low-saliency dimensions, ER maintains geometric coherence and effectively filters out semantic noise. Extensive experiments on three fine-grained datasets show that our method significantly outperforms state-of-the-art few-shot classification methods.

## 🖼️ Visualizations

**Overall Model Architecture (VRAS + ER):**

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/model.png?raw=true" alt="Model Architecture" width="70%">
  <br/>
  <em>Figure 1: Overall model architecture. (Please verify 'model.png' exists in the Figure directory. Adjust filename/extension if necessary, e.g., model.jpg)</em>
</p>

**ER Module & Feature Maps:**

<table>
  <tr>
    <td align="center" valign="top">
      <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/ER.png?raw=true" alt="ER Module" width="400">
      <br/><em>Figure 2: ER Module. (Verify 'ER.png')</em>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/feature_map.png?raw=true" alt="Feature Maps" width="400">
      <br/><em>Figure 3: Feature Map Visualization. (Verify 'feature_map.png')</em>
    </td>
  </tr>
</table>
*(Adjust filenames/extensions and paths in the `src` attributes above if the images don't display correctly)*

## 🧱 View-Robust Backbones

This repository provides implementations for two VRAS backbone variants:

* [`VRAS-Conv-4`](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-Conv4.py)
* [`VRAS-ResNet-12`](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-ResNet12.py)

## ⚙️ Setup Environment

Create and activate the Conda environment using the provided file:

```bash
conda env create -f environment.yml
conda activate VRAS
```

## 🚀 Training and Evaluation

Run the training and evaluation scripts for the different datasets and backbones:

**CUB Dataset:**

```bash
# Using VRAS-Conv-4
python experiments/CUB/VRAS-Conv-4/train.py

# Using VRAS-ResNet-12
python experiments/CUB/VRAS-ResNet-12/train.py
```

**Cars Dataset:**

```bash
# Using VRAS-Conv-4
python experiments/cars/VRAS-Conv-4/train.py

# Using VRAS-ResNet-12
python experiments/cars/VRAS-ResNet-12/train.py
```

**Dogs Dataset:**

```bash
# Using VRAS-Conv-4
python experiments/dogs/VRAS-Conv-4/train.py

# Using VRAS-ResNet-12
python experiments/dogs/VRAS-ResNet-12/train.py
```

## 📊 Experimental Results

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/table1.jpeg?raw=true" alt="Experimental Results Table 1" width="80%">
  <br/><em>Table 1: Main experimental results.</em>
</p>
<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/table2.jpeg?raw=true" alt="Experimental Results Table 2" width="80%">
  <br/><em>Table 2: Additional experimental results.</em>
</p>

