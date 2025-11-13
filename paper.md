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


## 3. Methods

### 3.1 Problem Setup: Measuring Translation Invariance

We consider the standard MNIST digit recognition task, with 28×28 grayscale images of digits \(0\)–\(9\). We define three evaluation regimes:

1. **Clean test set:** Standard MNIST test images (centered digits).
2. **Shifted test sets:** Images randomly translated by up to \(\pm k\) pixels in each direction, with zero-padding, for \(k \in \{2, 4, 6\}\).
3. **Invariance score:** A metric that measures prediction consistency across multiple random shifts of the same base image.

**Goal in precise form:**  
Given two models with similar parameter counts but different architectures (Neocognitron-inspired vs vanilla CNN), quantify how their performance degrades under increasing translation and how stable their predictions remain under such transformations.

---

### 3.2 Architectures

#### 3.2.1 Neocognitron-inspired network

We implement a simplified S/C cascade that preserves the spirit of the Neocognitron while being fully trainable with backpropagation:

- **S1:** Conv2d(1 → 16, kernel 5, padding 2), ReLU  
- **C1:** MaxPool2d(2×2)
- **S2:** Conv2d(16 → 32, kernel 5, padding 2), ReLU  
- **C2:** MaxPool2d(2×2)
- **S3:** Conv2d(32 → 64, kernel 3, padding 1), ReLU  
- **Global average pooling** → 64-d feature vector  
- **Linear classifier:** 64 → 10 logits

This architecture uses **aggressive early pooling**, reducing spatial dimensions quickly and pushing the network to discard precise positional information in favor of pattern presence.

**Architectural summary:**

- Alternating conv–pool (S/C) blocks.
- Strong inductive bias toward local shift invariance.
- Compact classifier head via global average pooling.

#### 3.2.2 Tiny CNN baseline

The baseline is a small, modern-style CNN emphasizing depth of convolutions before pooling:

- Conv2d(1 → 32, kernel 3, padding 1), ReLU  
- Conv2d(32 → 32, kernel 3, padding 1), ReLU  
- MaxPool2d(2×2)
- Conv2d(32 → 64, kernel 3, padding 1), ReLU  
- MaxPool2d(2×2)
- Flatten → FC(64·7·7 → 128), ReLU → FC(128 → 10)

This design preserves more spatial detail in early layers and uses fully connected layers at the end, more in line with AlexNet/VGG-style classification heads. :contentReference[oaicite:9]{index=9}  

**Architectural summary:**

- More local detail before pooling.
- Standard conv–conv–pool pattern.
- Higher capacity in the dense classifier head.

---

### 3.3 Training Protocol

- **Dataset:** MNIST, standard train/test split.
- **Preprocessing:** Pixel values normalized to \([0, 1]\) via `ToTensor()`.
- **Optimizer:** AdamW with learning rate \(10^{-3}\) and weight decay \(10^{-4}\).
- **Batch size:** 128.
- **Epochs:** e.g., 10–40 (to be selected empirically; initial experiments can start at 10 and increase if underfitting).
- **Data augmentation:** None beyond basic normalization (no training-time translations), to isolate architectural contributions to invariance.

**Training objective:**  
Minimize cross-entropy loss between predicted logits and ground-truth digit labels.

---

### 3.4 Translation and Invariance Metrics

#### 3.4.1 Shifted test sets

For each shift radius \(k \in \{2, 4, 6\}\):

- For each test image \(x\), sample an integer horizontal shift \(\Delta x\) and vertical shift \(\Delta y\) independently from \([-k, k]\).
- Translate the image by \((\Delta x, \Delta y)\), filling any uncovered regions with zeros.
- Evaluate test accuracy on the resulting shifted dataset.

This yields **accuracy vs shift radius** curves for each architecture.

#### 3.4.2 Invariance score

For a given model \(f\) and shift radius \(k\), define the invariance score as follows:

- Let \(x\) be an original test image and \(\hat{y}(x) = \arg\max_c f_c(x)\) its predicted label.
- Generate \(K\) independent random shifts \(T_i(x)\) with \((\Delta x_i, \Delta y_i) \sim \text{Uniform}\{-k, \dots, k\}\).
- Define the per-sample invariance:

\[
I(x) = \frac{1}{K} \sum_{i=1}^K \mathbf{1}\big[\hat{y}(T_i(x)) = \hat{y}(x)\big].
\]

- The global invariance score for radius \(k\) is

\[
I_k = \mathbb{E}_{x \sim \mathcal{D}_{\text{test}}} [I(x)].
\]

In practice, we approximate \(I_k\) by averaging over a subset (e.g., 2,000–10,000 test images) and a small \(K\) (e.g., 5–10 shifts per image).

**Interpretation:**  
An invariance score near 1 indicates that the model’s prediction is stable under the specified range of translations; a low score indicates sensitivity to small positional changes.

---

## 4. Real-World Applications of CNNs

Although our experiments focus on MNIST, the concepts are relevant to major application domains where CNNs are deployed at scale.

### 4.1 Image Classification

In classical image classification, CNNs ingest an image and output a single label, such as “cat” or “traffic light.” Large-scale benchmarks like ImageNet have driven innovations in architectures such as AlexNet, VGG, ResNet, and their successors. :contentReference[oaicite:10]{index=10}  

Translation invariance is beneficial here: a cat should be recognized regardless of whether it appears slightly left or right in the frame. However, these models are typically trained with data augmentation (random crops, flips), making it difficult to disentangle architectural from augmentation-induced invariance.

**Key point:**  
Architectural invariance (e.g., pooling, large receptive fields) and augmentation-based invariance both contribute to robustness in classification.

---

### 4.2 Object Detection and Autonomous Vehicles

Object detection extends classification by localizing multiple objects within a scene using bounding boxes or segmentation masks. CNN-based detectors like Faster R-CNN, SSD, and YOLO variants rely heavily on convolutional backbones.  

In autonomous driving, detection systems must robustly locate pedestrians, vehicles, and traffic signs under varying weather, lighting, and camera positions. Invariance and equivariance to translation, scale, and moderate viewpoint changes are crucial for safety.

**Connection to our study:**  
If a detection backbone is overly sensitive to small translations, detection stability may degrade when objects move slightly between frames. Architectures with stronger inherent invariance can serve as more robust feature extractors for such systems.

---

### 4.3 Medical Imaging

CNNs are extensively used in medical imaging for tasks such as:

- Classifying lesions in dermatology images.
- Detecting tumors in MRI or CT scans.
- Segmenting anatomical structures in ultrasound or X-ray data.

Here, invariance has two aspects:

- **Positional invariance:** A lesion should be detected irrespective of exact pixel location in the projected image.
- **Clinical invariance:** Predictions should be robust to scanner differences, patient pose, and minor acquisition artefacts.

Model failures in medical imaging can have serious clinical consequences, making robustness and interpretability critical evaluation dimensions.

**Relevance:**  
Studying simple invariance metrics on toy datasets provides a sandbox for understanding how architectural choices scale to safety-critical domains.

---

### 4.4 Efficient CNNs for Edge Devices

MobileNet and related families introduce depthwise separable convolutions and tunable width/depth multipliers to reduce computation and model size, enabling on-device inference on smartphones, drones, and IoT devices. :contentReference[oaicite:11]{index=11}  

In edge scenarios:

- Power and latency constraints limit model complexity.
- On-device processing is preferred for privacy in applications like face unlocking or home surveillance.
- Data distributions can shift due to local environmental conditions.

Architectures that maintain invariance under such shifts, without relying solely on large-scale augmentation, are particularly attractive.

**Bridge to this paper:**  
The Neocognitron-inspired architecture is relatively lightweight and may inspire future efficient designs that explicitly bias for invariance.

---

## 5. Ethical Considerations

### 5.1 Bias in Facial Recognition

Deep CNNs power face verification and identification systems used by social networks, device unlocking, and law enforcement. Multiple studies have shown that such systems often exhibit **demographic bias**, with higher error rates for people of color, women, and other underrepresented groups. :contentReference[oaicite:12]{index=12}  

Investigations have documented wrongful arrests based on misidentification, with disproportionate impact on Black individuals. :contentReference[oaicite:13]{index=13}  
Civil rights organizations and human rights bodies have raised concerns that live facial recognition, when deployed without strict regulation and transparency, can violate privacy, freedom of assembly, and equality rights.

**Core issue:**  
Architectural sophistication alone does not ensure fairness; data, labels, deployment context, and governance structures are equally critical.

---

### 5.2 Invariance vs Fairness

While this paper focuses on translation invariance, **not all invariances are ethically benign**. For example:

- A model that is invariant to lighting changes might still encode sensitive attributes that correlate with race.
- Attempts to “blind” models to sensitive features by preprocessing alone can fail, because deep networks can reconstruct such features indirectly. :contentReference[oaicite:14]{index=14}  

Fairness requires:

- Careful dataset curation and representation.
- Explicit fairness constraints or regularizers.
- Continuous monitoring of model behavior post-deployment.

**Ethical reminder:**  
Robustness to perturbations is necessary but not sufficient; robust yet biased models can consistently produce harmful outcomes.

---

### 5.3 Responsible Deployment

Any research that touches architectures used in safety-critical or surveillance-related settings should:

- Report limitations clearly.
- Avoid overstating robustness claims.
- Encourage reproducibility and community scrutiny.

In this work, we restrict ourselves to MNIST as a pedagogical testbed and explicitly avoid drawing direct deployment recommendations for high-stakes applications.

---


## 6. Experiments

> **Note:** This section describes the experimental protocol and contains placeholder tables. After you run the provided code, fill in the actual numbers.

### 6.1 Setup

- **Models:** Neocognitron-like vs Tiny CNN baseline.
- **Training:** AdamW, learning rate \(10^{-3}\), weight decay \(10^{-4}\), batch size 128, epochs \(E\) (e.g., 20).
- **Evaluation:**
  - Clean test accuracy.
  - Shifted test accuracy for \(k \in \{2, 4, 6\}\).
  - Invariance scores \(I_k\) for each radius.

---

### 6.2 Clean Test Accuracy

| Model                 | Params (approx) | Clean Test Accuracy (%) |
|-----------------------|-----------------|-------------------------|
| Neocognitron-like     | *N\_neo*        | *to be filled*          |
| Tiny CNN baseline     | *N\_cnn*        | *to be filled*          |

**Interpretation (example narrative):**  
The tiny CNN may achieve slightly higher clean accuracy due to richer feature representations and a larger dense head, while the Neocognitron-like model may be marginally weaker on centered data.

---

### 6.3 Accuracy Under Translations

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

### 6.4 Invariance Scores

We compute the invariance score \(I_k\) for each model and shift radius, averaging over a subset of test images (e.g., 2,000 images, 5 shifts each).

| Model             | Shift Radius \(k\) | Invariance Score \(I_k\) |
|-------------------|--------------------|--------------------------|
| Neocognitron-like | 2                  | *to be filled*           |
| Neocognitron-like | 4                  | *to be filled*           |
| Neocognitron-like | 6                  | *to be filled*           |
| Tiny CNN          | 2                  | *to be filled*           |
| Tiny CNN          | 4                  | *to be filled*           |
| Tiny CNN          | 6                  | *to be filled*           |

**Expected qualitative result:**  
The Neocognitron-like network should display higher invariance scores at larger shifts, reflecting its architectural bias toward spatial pooling.

---

### 6.5 Ablation Studies (Optional but Recommended)

To deepen the analysis, consider:

- **Pooling ablation:** Remove one pooling layer in the Neocognitron-like architecture and measure changes in \(I_k\).
- **Data augmentation:** Add random translations during training for both models and see whether invariance differences shrink.
- **Pooling type:** Replace max pooling with average pooling and evaluate the impact on robustness.

These ablations help isolate which architectural components most strongly influence invariance.

---

## 7. Discussion

Our study revisits an early vision-inspired architecture and examines its behavior under modern training. Several themes emerge:

1. **Architectural inductive bias:**  
   The Neocognitron-like network’s early and repeated pooling naturally encourages invariance, which manifests as higher \(I_k\) scores at larger shift radii.

2. **Accuracy–invariance trade-off:**  
   The tiny CNN may outperform on clean, centered data but drops faster under large translations, highlighting a trade-off between fine-grained spatial fidelity and robustness.

3. **Relevance to modern CNN design:**  
   While contemporary architectures rarely adopt explicit S/C terminology, design choices such as pooling schedules, receptive-field growth, and downsampling strategies remain crucial knobs that implicitly shape invariance.

4. **Limitations of simplistic testbeds:**  
   MNIST is a small, clean, grayscale dataset; results may not directly generalize to complex natural images or multi-object scenes, but they provide a controlled sandbox for reasoning about invariance.

**Conceptual moral:**  
Old ideas like Neocognitron’s architectural invariance bias still offer insights when interrogated with modern tools and evaluation metrics.

---



