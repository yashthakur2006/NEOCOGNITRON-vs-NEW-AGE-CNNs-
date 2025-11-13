# Neocognitron vs Modern CNN
# Neocognitron Revisited: Shift Invariance of Early CNN Architectures Under Modern Training

*Author: YASH THAKUR  

---

## Abstract

The Neocognitron, introduced by Fukushima in 1980, is widely regarded as a precursor to modern convolutional neural networks (CNNs). It was explicitly designed for pattern recognition “unaffected by shift in position,” achieved via a hierarchical arrangement of simple (S) and complex (C) cells that progressively build local invariances. 
In contrast, modern CNNs such as AlexNet, VGG, and MobileNet are typically optimized for accuracy on large-scale benchmarks rather than explicitly measured invariance properties. 

In this work, we revisit Neocognitron-style architectures using contemporary training tools. We implement a simplified S/C-style network and compare it against a small modern CNN baseline with a similar parameter budget. Both models are trained end-to-end with backpropagation and AdamW on MNIST. We propose a quantitative **invariance score**, defined as prediction consistency under random spatial translations of the input. Empirically, we expect the Neocognitron-like model to trade a small amount of clean accuracy for significantly higher translation invariance at larger shifts. We discuss implications for modern architectures, real-world applications such as image classification, object detection in autonomous vehicles, and medical imaging, and we highlight ethical considerations, especially in facial recognition systems where bias and misidentification are pressing concerns.

**Short summary:**  
We re-examine Neocognitron-style architectures under modern optimization, propose a simple invariance metric, and position this as a bridge between foundational CNN theory and robustness needs in real-world applications.

---

## 1. Introduction

Convolutional neural networks (CNNs) dominate visual recognition tasks across academia and industry, powering systems from mobile photo tagging to autonomous driving perception stacks. Yet their conceptual roots trace back to biologically inspired hierarchical models such as the Neocognitron, which explicitly targeted position-invariant pattern recognition through a cascade of local feature detectors and pooling operations. :contentReference[oaicite:3]{index=3}  

Modern CNN milestones — including LeNet-5, AlexNet, VGG, ResNet, and MobileNet — have largely focused on improving accuracy and scalability on large-scale datasets such as ImageNet. :contentReference[oaicite:4]{index=4}  
While these architectures certainly exhibit some degree of translation invariance, robustness is rarely quantified as a primary metric in standard benchmarks. Instead, invariance is often left to emerge as a side effect of convolutions, pooling, and data augmentation.

This paper asks a deliberately focused question:

> **Given modern training pipelines and optimization methods, do Neocognitron-style S/C hierarchies still offer an advantage in translation invariance compared to a vanilla small CNN, when trained on the same data without explicit translation augmentation?**

To explore this question, we:

- Implement a **Neocognitron-inspired architecture** with alternated S (convolutional) and C (pooling) layers, trained end-to-end with backpropagation.
- Construct a **tiny modern CNN baseline** with stacked 3×3 convolutions and moderate pooling.
- Propose a **translation invariance score** based on prediction consistency under random input shifts.
- Evaluate both models on MNIST and shifted variants of the MNIST test set.

**Key idea in one line:**  
We revisit an early invariance-driven architecture under modern tools, and explicitly measure the invariance–accuracy trade-off rather than assuming robustness “comes for free.”

---

## 2. Background

### 2.1 The Neocognitron

The Neocognitron is a hierarchical neural network proposed by Fukushima as a model of visual pattern recognition in the ventral visual stream. :contentReference[oaicite:5]{index=5}  
Its core principles are:

- **S-cells (simple cells):** Local feature detectors that respond to specific patterns (e.g., oriented edges) at particular positions.
- **C-cells (complex cells):** Units that pool over neighboring S-cells, building local invariance to small shifts and distortions.
- **Layered hierarchy:** Repeated S/C modules arranged in depth, gradually increasing receptive field size and invariance extent.

The Neocognitron originally employed **self-organizing, unsupervised learning** inspired by Hebbian mechanisms, without gradient-based backpropagation. Its design prioritized invariant recognition over the specific loss landscapes that modern deep learning focuses on.

**Conceptual takeaway:**  
Neocognitron embodies an explicit architectural commitment to invariance, realized through early and repeated pooling.

---

### 2.2 Modern CNNs: From AlexNet to MobileNet

The modern CNN wave was catalyzed by **AlexNet**, which won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 with a large, GPU-trained CNN containing five convolutional and three fully connected layers. :contentReference[oaicite:6]{index=6}  
Key mechanisms included ReLU activations, local response normalization, and overlapping max pooling.

**VGG** subsequently showed that pushing depth with simple 3×3 convolutions could significantly improve performance, with 16–19 layer networks becoming a standard feature extractor in many downstream tasks. :contentReference[oaicite:7]{index=7}  

**MobileNet** introduced depthwise separable convolutions and simple width/depth multipliers to optimize the trade-off between latency, memory, and accuracy, making CNNs practical for mobile and embedded devices. :contentReference[oaicite:8]{index=8}  

Across this trajectory, CNNs have evolved toward:

- **Deeper architectures** with smaller kernels.
- **More efficient operations** for edge and mobile deployment.
- **Better training protocols** (batch normalization, improved optimizers, learning rate schedules).

**High-level contrast:**  
Neocognitron emphasized *invariance as a design goal*, whereas modern CNNs typically optimize for *accuracy under large datasets and generic training tricks*.

---

### 2.3 Convolution as a Sliding Magnifying Glass

Convolution in CNNs can be intuitively understood as sliding a magnifying glass over an image:

- Imagine moving a small lens over different parts of the image.
- At each position, the lens examines a small patch (e.g., 3×3 or 5×5 pixels).
- The convolutional filter acts like a specialized magnifying glass that responds strongly when a particular pattern (e.g., a vertical edge) appears under the lens.

Formally, a 2D convolution with kernel \(K\) over image \(X\) computes:

\[
(Y * K)(i, j) = \sum_{u,v} X(i+u, j+v)\,K(u,v),
\]

where \(Y\) is the output feature map, and \((u,v)\) index spatial offsets.

**Intuition in one sentence:**  
Convolution is a pattern detector that scans the image, reporting where a particular pattern is present and how strongly it appears.

---


## 3. Experiments

> **Note:** This section describes the experimental protocol and contains placeholder tables. After you run the provided code, fill in the actual numbers.

### 3.1 Setup

- **Models:** Neocognitron-like vs Tiny CNN baseline.
- **Training:** AdamW, learning rate \(10^{-3}\), weight decay \(10^{-4}\), batch size 128, epochs \(E\) (e.g., 20).
- **Evaluation:**
  - Clean test accuracy.
  - Shifted test accuracy for \(k \in \{2, 4, 6\}\).
  - Invariance scores \(I_k\) for each radius.

---

### 3.2 Clean Test Accuracy

| Model                 | Params (approx) | Clean Test Accuracy (%) |
|-----------------------|-----------------|-------------------------|
| Neocognitron-like     | *N\_neo*        | *to be filled*          |
| Tiny CNN baseline     | *N\_cnn*        | *to be filled*          |

**Interpretation (example narrative):**  
The tiny CNN may achieve slightly higher clean accuracy due to richer feature representations and a larger dense head, while the Neocognitron-like model may be marginally weaker on centered data.

---

### 3.3 Accuracy Under Translations

For each shift radius \(k\), we evaluate accuracy on a shifted MNIST test set.

| Model             | Shift Radius \(k\) | Shifted Accuracy (%) |
|-------------------|--------------------|----------------------|
| Neocognitron-like | 0 (clean)          | *to be filled*       |
| Neocognitron-like | 2                  | *to be filled*       |
| Neocognitron-like | 4                  | *to be filled*       |
| Neocognitron-like | 6                  | *to be filled*       |
| Tiny CNN          | 0 (clean)          | *to be filled*       |
| Tiny CNN          | 2                  | *to be filled*       |
| Tiny CNN          | 4                  | *to be filled*       |
| Tiny CNN          | 6                  | *to be filled*       |

You may also visualize this as curves of accuracy vs shift radius.

---

### 3.4 Invariance Scores

We compute the invariance score \(I_k\) for each model and shift radius, averaging over a subset of test images (e.g., 2,000 images, 5 shifts each).

| Model             | Shift Radius \(k\) | Invariance Score \(I_k\) |
|-------------------|--------------------|--------------------------|
| Neocognitron-like | 2                  | *to be filled*           |
| Neocognitron-like | 4                  | *to be filled*           |
| Neocognitron-like | 6                  | *to be filled*           |
| Tiny CNN          | 2                  |           |
| Tiny CNN          | 4                  |            |
| Tiny CNN          | 6                  |          |

**Expected qualitative result:**  
The Neocognitron-like network should display higher invariance scores at larger shifts, reflecting its architectural bias toward spatial pooling.

---

### 3.5 Ablation Studies (Optional but Recommended)

To deepen the analysis, consider:

- **Pooling ablation:** Remove one pooling layer in the Neocognitron-like architecture and measure changes in \(I_k\).
- **Data augmentation:** Add random translations during training for both models and see whether invariance differences shrink.
- **Pooling type:** Replace max pooling with average pooling and evaluate the impact on robustness.

These ablations help isolate which architectural components most strongly influence invariance.

---
