# Breast Cancer Wisconsin Diagnostic Analysis
## COME 403 Project Report
By Mert Topkaya and Azizcan Yalçın 

## Overview
This project implements and compares various machine learning algorithms for breast cancer diagnosis using the Wisconsin Breast Cancer Diagnostic dataset. The goal is to accurately classify breast masses as either benign (0) or malignant (1) using features extracted from digitized images of fine needle aspirates (FNA).

## Dataset
Source: [UCI Machine Learning Repository - Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Features
The dataset includes 30 features computed from cell nuclei images:
- Radius (mean, SE, worst)
- Texture (mean, SE, worst)
- Perimeter (mean, SE, worst)
- Area (mean, SE, worst)
- Smoothness (mean, SE, worst)
- Compactness (mean, SE, worst)
- Concavity (mean, SE, worst)
- Concave points (mean, SE, worst)
- Symmetry (mean, SE, worst)
- Fractal dimension (mean, SE, worst)

## Models Implemented
1. K-Nearest Neighbors (KNN)
   - K=3
   - K=7
   - K=11

2. Multi-Layer Perceptron (MLP)
   - Single hidden layer (32 neurons)
   - Two hidden layers (32, 32 neurons)
   - Three hidden layers (32, 32, 32 neurons)

3. Naive Bayes
   - Gaussian Naive Bayes implementation

## Model Performance Comparison
### Accuracy Rankings
1. Neural Network (3 hidden layers): 96.79%
2. KNN (K=11): 96.00%
3. Neural Network (1 hidden layer): 95.61%
4. Neural Network (2 hidden layers): 94.74%
5. KNN (K=3) & KNN (K=7): 94.00%
6. Naive Bayes: 92.00%

## Implementation Details
### Data Preprocessing
- Standard scaling applied to features
- Train-test splits:
  - KNN: 70-30 split
  - MLP: 80-20 split
  - Naive Bayes: 75-25 split

### MLP Configuration
- Optimizer: Adam
- Maximum iterations: 5000
- Activation: ReLU
- Batch size: 64
- Learning rate: 0.001
- Alpha: 0.01

## Technologies Used
- Python
- Libraries:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Results Visualization
The project includes visualizations for:
- Model rankings by accuracy
- Model rankings by precision
- Model rankings by recall
- Model rankings by F1-score

## Usage
1. Ensure all required libraries are installed:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

2. Load the dataset and run the analysis:
```python
import pandas as pd
df_bcancer = pd.read_csv("wdbc.data", header=None)
```

3. Follow the Jupyter notebook for complete analysis pipeline.

## Conclusions
- The Neural Network with three hidden layers achieved the best overall performance
- KNN with K=11 showed strong performance, suggesting that a larger neighborhood provides better classification for this dataset
- All models achieved accuracy above 90%, demonstrating the effectiveness of machine learning for breast cancer diagnosis
