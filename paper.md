# NEOCOGNITRON-vs-NEW-AGE-CNNs-
Shift Invariance of Early CNN Architectures Under Modern Training

*Author: Yash THakur*  

## Abstract
The Neocognitron, introduced by Fukushima in 1980, is widely regarded as a precursor to modern convolutional neural networks (CNNs). It was explicitly designed for pattern recognition “unaffected by shift in position,” achieved via a hierarchical arrangement of simple (S) and complex (C) cells that progressively build local invariances. 

In contrast, modern CNNs such as AlexNet, VGG, and MobileNet are typically optimized for accuracy on large-scale benchmarks rather than explicitly measured invariance properties.

In this work, we revisit Neocognitron-style architectures using contemporary training tools. We implement a simplified S/C-style network and compare it against a small modern CNN baseline with a similar parameter budget. Both models are trained end-to-end with backpropagation and AdamW on MNIST. We propose a quantitative **invariance score**, defined as prediction consistency under random spatial translations of the input. Empirically, we expect the Neocognitron-like model to trade a small amount of clean accuracy for significantly higher translation invariance at larger shifts. We discuss implications for modern architectures, real-world applications such as image classification, object detection in autonomous vehicles, and medical imaging, and we highlight ethical considerations, especially in facial recognition systems where bias and misidentification are pressing concerns. 


> **Given modern training pipelines and optimization methods, do Neocognitron-style S/C hierarchies still offer an advantage in translation invariance compared to a vanilla small CNN, when trained on the same data without explicit translation augmentation?**

To explore this question, we:

- Implement a **Neocognitron-inspired architecture** with alternated S (convolutional) and C (pooling) layers, trained end-to-end with backpropagation.
- Construct a **tiny modern CNN baseline** with stacked 3×3 convolutions and moderate pooling.
- Propose a **translation invariance score** based on prediction consistency under random input shifts.
- Evaluate both models on MNIST and shifted variants of the MNIST test set.

**Key idea in one line:**  
We revisit an early invariance-driven architecture under modern tools, and explicitly measure the invariance–accuracy trade-off rather than assuming robustness “comes for free.”


## 2. Background

### 2.1 The Neocognitron

The Neocognitron is a hierarchical neural network proposed by Fukushima as a model of visual pattern recognition in the ventral visual stream.
Its core principles are:

- **S-cells (simple cells):** Local feature detectors that respond to specific patterns (e.g., oriented edges) at particular positions.
- **C-cells (complex cells):** Units that pool over neighboring S-cells, building local invariance to small shifts and distortions.
- **Layered hierarchy:** Repeated S/C modules arranged in depth, gradually increasing receptive field size and invariance extent.

The Neocognitron originally employed **self-organizing, unsupervised learning** inspired by Hebbian mechanisms, without gradient-based backpropagation. Its design prioritized invariant recognition over the specific loss landscapes that modern deep learning focuses on.

**Conceptual takeaway:**  
Neocognitron embodies an explicit architectural commitment to invariance, realized through early and repeated pooling.



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



