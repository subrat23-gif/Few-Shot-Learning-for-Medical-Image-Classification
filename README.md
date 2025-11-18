# Hybrid Few-Shot Learning for Chest X-Ray Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Abstract

Medical imaging datasets often suffer from severe class imbalance and limited samples for rare diseases. Conventional supervised models trained on large datasets fail to generalize to these “tail” classes.
This project introduces a Hybrid Few-Shot Learning (FSL) framework that combines:
- A DenseNet121 imbalance-aware specialist model, trained on large-scale chest X-ray datasets.
- A Prototypical Network with Cosine Similarity, using the specialist backbone to classify novel diseases such as COVID-19 and Tuberculosis from very few labeled   samples.
- This approach leverages domain-specific transfer learning instead of relying on generic ImageNet features.

Key Features
- Imbalance-Aware Training: Uses WeightedRandomSampler to train DenseNet121 on skewed medical datasets.
- Domain-Specific Transfer: Transfers radiology-specific features learned during supervised training to the FSL model.
- Cosine Metric Learning: More stable than Euclidean in high-dimensional embeddings.
- Unified Preprocessing Pipeline: Normalization and augmentation consistent with TorchXRayVision standards.

Methodology
Stage 1: Specialist Model (Supervised Learning)
Backbone: DenseNet121
Loss Strategy: Class-weighted sampling
Output: Robust feature extractor trained on large base classes
Performance: ~86% Weighted F1-Score

Stage 2: Weight Extraction (Backbone Transplant)
Remove classification head
Freeze / partially fine-tune backbone
Obtain domain-specific embedding network

Stage 3: Few-Shot Learning (Meta-Learning)
Model: Prototypical Network
Task: 2-Way, 10-Shot evaluation on novel diseases
Metric: Cosine Similarity

Training: Episodic meta-learning on base classes

Datasets Used
Source	Purpose
NIH ChestX-ray14	Base classes
RSNA Pneumonia Dataset	Base classes
COVID-19 Radiography Dataset	Novel class
Tuberculosis CXR Dataset	Novel class
Installation & Usage
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/Hybrid-FSL-ChestXRay.git
cd Hybrid-FSL-ChestXRay

2. Install Requirements
pip install -r requirements.txt

3. Preprocess the Datasets
python src/preprocess.py

4. Train the Specialist Model
python train_specialist.py

5. Run Few-Shot Learning
python train_fsl.py

| Model            | Metric    | Backbone Weights      | Accuracy (Novel Classes) |
| ---------------- | --------- | --------------------- | ------------------------ |
| Prototypical Net | Euclidean | ImageNet              | 61.43% ± 0.65%           |
| Hybrid ProtoNet  | Cosine    | Imbalanced-Pretrained | 91.6%                    |


