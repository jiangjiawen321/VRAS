<h1 align="center">🌄 View-Robust Backbone and Discriminative Reconstruction for Few-Shot Fine-Grained Image Classification</h1>

<p align="center">
  <a href="https://example.com/path/to/your/paper">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-IJCNN%202025-blue" />
  </a>
  <a href="https://github.com/jiangjiawen321/VRAS">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/jiangjiawen321/VRAS?style=social" />
  </a>
</p>

<p align="center"><strong>⭐ If you find our code useful, please consider starring this repository!</strong></p>

---

## 📄 Abstract

> We study few-shot fine-grained image classification, a task that faces two key challenges: (1) the scarcity of labeled samples amplifies the model’s sensitivity to viewpoint variations, resulting in feature inconsistency, and (2) reconstruction-based methods, while improving inter-class separability, inadvertently introduce intra-class variations, further complicating discrimination.
To address these challenges, we propose the View-Robust Attention Selector (VRAS), a feature enhancement backbone designed to mitigate viewpoint-induced misclassifications. By integrating cross-scale feature interaction and adaptive selection mechanisms, VRAS effectively reduces spatial sensitivity arising from the limited viewpoint diversity in few-shot support sets. This approach not only preserves intra-class consistency but also enhances inter-class discriminability, ensuring robust feature representations.
Furthermore, we introduce the Enhancement and Reconstruction (ER) module, designed to strengthen discriminative learning. ER achieves this by maximizing inter-class divergence while enhancing intra-class compactness through a regularized Ridge Regression optimization strategy. By dynamically suppressing low-saliency dimensions, ER maintains geometric coherence and effectively filters out semantic noise.
> Extensive experiments on three fine-grained datasets show that our method significantly outperforms state-of-the-art few-shot classification methods.

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/1.jpeg" alt="Model Architecture" width="100%">
  <br/><em>Figure 1: Model Architecture.</em>
</p>

---

## 🧱 View-Robust Backbones

🔧 This repository provides implementations for two VRAS backbone variants:

- 🧩 [VRAS-Conv-4](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-Conv4.py) – lightweight, simple convolutional baseline.
- 🧱 [VRAS-ResNet-12](https://github.com/jiangjiawen321/VRAS/blob/main/models/backbones/VRAS-ResNet12.py) – deeper architecture with stronger representation capacity.

---

## ⚙️ Setup Environment

📦 To get started, create and activate the Conda environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate VRAS
```

✅ Environment ready!

---

## 🚀 Training and Evaluation

🎯 Run the training and evaluation scripts on different datasets:

### 🕊 CUB Dataset

```bash
# VRAS-Conv-4
python experiments/CUB/VRAS-Conv-4/train.py

# VRAS-ResNet-12
python experiments/CUB/VRAS-ResNet-12/train.py
```

### 🚗 Cars Dataset

```bash
# VRAS-Conv-4
python experiments/cars/VRAS-Conv-4/train.py

# VRAS-ResNet-12
python experiments/cars/VRAS-ResNet-12/train.py
```

### 🐶 Dogs Dataset

```bash
# VRAS-Conv-4
python experiments/dogs/VRAS-Conv-4/train.py

# VRAS-ResNet-12
python experiments/dogs/VRAS-ResNet-12/train.py
```

---

## 📊 Experimental Results

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/table1.jpeg?raw=true" alt="Experimental Results Table 1" width="100%">
  <br/><em>Table 1: Comparison with state-of-the-art methods.</em>
</p>

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/table2.jpeg?raw=true" alt="Experimental Results Table 2" width="100%">
  <br/><em>Table 2: Ablation Study.</em>
</p>

<p align="center">
  <img src="https://github.com/jiangjiawen321/VRAS/blob/main/Figure/table3.jpeg?raw=true" alt="Viewpoint-Robust Ablation Study" width="70%">
  <br/><em>Table 3: Viewpoint-Robust Ablation Study.</em>
</p>

---

## 🙏 Acknowledgement

Special thanks to the open-source community, especially [**FRN**](https://github.com/Tsingularity/FRN), whose work inspired part of this project. 💡

---

## 📈 Star History

<p align="center">
  <a href="https://www.star-history.com/#jiangjiawen321/VRAS&Date">
    <img src="https://api.star-history.com/svg?repos=jiangjiawen321/VRAS&type=Date" alt="Star History Chart"/>
  </a>
</p>

