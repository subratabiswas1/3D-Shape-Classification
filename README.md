# 🌀 3D Shape Classification using PointNet

This project implements **3D point cloud classification** on the **ModelNet10 dataset** using a simplified version of the **PointNet architecture** in PyTorch.  

---

## 🎯 Objective
The goal of this project is to classify **3D shapes** (represented as point clouds) into one of the **10 object categories** in the ModelNet10 dataset.  
Example categories include: `chair`, `sofa`, `bathtub`, `bed`, etc.  

---

## 🛠️ Method
We implement a **PointNet-based classifier** that directly consumes point clouds without voxelization or mesh conversion.  

### Key Components:
- **Data Augmentation**  
  - Random rotation around z-axis  
  - Random jittering  
  - Conversion to PyTorch tensors  

- **Model Architecture** (PointNet Classifier)
  - Shared MLPs via `Conv1d` layers  
  - Max pooling to extract global features  
  - Fully connected layers with dropout and batch norm  
  - Final classification with **log-softmax**  

- **Training & Evaluation**
  - Loss: **Negative Log-Likelihood (NLLLoss)**  
  - Optimizer: **Adam**  
  - Metrics: Loss & Accuracy  

---

## ⚙️ Setup Process

### 1. Clone this repository
```bash
git clone https://github.com/your-username/3d-shape-classification.git
cd 3d-shape-classification
