
## 📌 Abstract
Medical imaging datasets often suffer from severe class imbalance and scarcity of data for rare diseases (e.g., COVID-19, Tuberculosis). Standard Deep Learning models struggle with these "tail" classes. 

This project proposes a **Hybrid Few-Shot Learning (FSL)** framework. Instead of training Prototypical Networks from scratch or using generic ImageNet weights, we first train a DenseNet121 "specialist" on a large, imbalanced medical dataset. We then surgically extract the backbone and transplant it into a Prototypical Network to perform Few-Shot classification on novel classes using Cosine Similarity.

## 🚀 Key Features
1.  **Imbalance-Aware Baseline:** A `DenseNet121` trained with `WeightedRandomSampler` to handle severe class imbalance (Normal: 72k images vs Tuberculosis: 560 images).
2.  **Domain-Specific Transfer:** Transfers weights from the Chest X-ray specialist model to the FSL learner, rather than using generic ImageNet weights.
3.  **Metric Learning Upgrade:** Utilizes **Cosine Similarity** (instead of Euclidean distance) in the Prototypical Loss function for better high-dimensional feature clustering.
4.  **Robust Preprocessing:** Standardized pipeline for NIH, RSNA, and Kaggle datasets using `TorchXRayVision` normalization standards.

## 📊 Methodology & Architecture

### Stage 1: The Specialist (Supervised Learning)
We train a standard classifier on the "Base" classes (Normal, Pneumonia).
* **Architecture:** DenseNet121 (modified head).
* **Strategy:** Weighted Random Sampling to penalize majority classes.
* **Result:** ~86% F1-Score (Weighted).

### Stage 2: The Transplant (Weight Extraction)
We strip the classification head from the Stage 1 model and freeze the feature extractor (Backbone). This backbone has learned specific radiographic features (opacities, infiltrates) that ImageNet weights lack.

### Stage 3: The Few-Shot Learner (Meta-Learning)
We use the transplanted backbone in a **Prototypical Network**.
* **Task:** 2-Way, 10-Shot Classification (COVID-19 vs. Tuberculosis).
* **Metric:** Cosine Similarity.
* **Training:** Episodic training on base classes, tested on novel classes.

## 📂 Dataset Strategy
The project aggregates data from four major sources:
* **NIH ChestX-ray14** (Base classes)
* **RSNA Pneumonia Challenge** (Base classes)
* **COVID-19 Radiography Database** (Novel class)
* **Tuberculosis Chest X-ray Database** (Novel class)

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone [https://github.com/YOUR_USERNAME/Hybrid-FSL-ChestXRay.git](https://github.com/YOUR_USERNAME/Hybrid-FSL-ChestXRay.git)
cd Hybrid-FSL-ChestXRay

### 2. Install Dependencies
'''bash
pip install -r requirements.txt

### 3. Data Setup
'''bash
Clean and standardise images
python src/preprocess.py

### Results

| Model               | Metric     | Backbone Weights       | Accuracy (Novel Classes) |
|--------------------|------------|-------------------------|---------------------------|
| Prototypical Net   | Euclidean  | ImageNet                | 61.43% ± 0.65%            |
| Hybrid ProtoNet    | Cosine     | Imbalanced-Pretrained  | 91.6% (Target)             |



