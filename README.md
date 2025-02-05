# COME 403 Project Report

**Authors:**  
- Mert Topkaya 
- Azizcan Yalçın

**Project:** Comparison of Machine Learning Models on the Breast Cancer Wisconsin (Diagnostic) Dataset

**Dataset Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

---

## Overview

This project demonstrates the analysis and classification of the Breast Cancer Wisconsin (Diagnostic) dataset using multiple machine learning algorithms. The goal is to evaluate and compare the performance of different models on the task of predicting whether a tumor is malignant or benign.

The following models have been implemented and compared:
- **K-Nearest Neighbors (KNN):** Tested with K=3, K=7, and K=11.
- **Multi-Layer Perceptron (MLP):** Tested with 1, 2, and 3 hidden layers (each with 32 neurons).
- **Naive Bayes:** Implemented using the GaussianNB classifier.

For each model, key performance metrics such as Accuracy, Precision, Recall, and F1-Score are calculated. Additionally, the project includes visualizations that rank the models based on these metrics.

---

## Technologies & Libraries

- **Python 3.x**
- **NumPy** – Numerical computations.
- **Pandas** – Data manipulation and analysis.
- **Matplotlib & Seaborn** – Data visualization.
- **Scikit-learn** – Machine learning algorithms and evaluation metrics.

---

## Project Structure

```
├── README.md                  # This file
├── wdbc.data                  # Dataset file (download from UCI repository)
└── project_script.py          # Main Python script containing the code below
```

---

## Installation & Requirements

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required packages:**

   It is recommended to use a virtual environment. For example, using `pip`:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Download the Dataset:**

   Download `wdbc.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and place it in the project root directory.

---

## Code Overview

The project code is structured as follows:

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using Pandas.
   - Columns are labeled with meaningful names.
   - The diagnosis column is encoded from categorical values (M = Malignant, B = Benign) to numeric values (1 for malignant, 0 for benign).
   - The data is split into training and testing sets, ensuring class stratification.
   - Features are standardized using `StandardScaler`.

2. **K-Nearest Neighbors (KNN) Classifier:**
   - KNN classifiers are implemented with `n_neighbors=3`, `n_neighbors=7`, and `n_neighbors=11` using the Euclidean distance metric.
   - For each KNN model, confusion matrices and accuracy scores are calculated.

3. **Multi-Layer Perceptron (MLP) Classifier:**
   - Three MLP configurations are tested:
     - One hidden layer (32 neurons).
     - Two hidden layers (32 neurons each).
     - Three hidden layers (32 neurons each).
   - Classification reports (precision, recall, f1-score) are generated for each configuration.

4. **Naive Bayes Classifier:**
   - A Gaussian Naive Bayes classifier is implemented.
   - The model is trained on the standardized data and evaluated using confusion matrices and classification reports.

5. **Model Performance Comparison:**
   - A summary table of performance metrics for each model is created.
   - Models are ranked by Accuracy, Precision, Recall, and F1-Score.
   - Seaborn is used to visualize the rankings through bar plots.

---

## Running the Code

To run the project, simply execute the main script:

```bash
python project_script.py
```

This will perform the following:
- Load and preprocess the dataset.
- Train and evaluate the KNN, MLP, and Naive Bayes classifiers.
- Print out performance metrics.
- Generate visualizations comparing model performances.

---

## Results & Analysis

The code outputs:
- Accuracy scores for KNN models (with different K values).
- Detailed classification reports for each MLP configuration.
- Performance metrics (Accuracy, Precision, Recall, F1-Score) for all models.
- Visual bar plots ranking models based on each metric.

These outputs help in understanding which model performs best on the given dataset and under which configurations.

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, please contact:
- **Mert Topkaya**  
- **Azizcan Yalçın**

Or open an issue in this repository.
