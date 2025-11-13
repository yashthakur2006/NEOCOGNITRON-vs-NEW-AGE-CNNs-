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




