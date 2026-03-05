# 04-single-neuron
# Fraud Detection – Single Neuron Model

## Overview

This project implements a **single-neuron neural network** to detect fraudulent bank transactions. The notebook includes preprocessing steps, handling class imbalance using **Random Undersampling**, model training, evaluation, and an additional experiment using **PCA (Principal Component Analysis)**.

The original dataset comes from Kaggle, but the version used in this project was **preprocessed beforehand** to make it suitable for the experiments. For this reason, the **processed dataset is already included in this repository** and must be uploaded manually when running the notebook.

Original dataset source:
https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection

---

# How to Run the Notebook in Google Colab

## 1. Download the Repository

Download or clone this repository to your computer.
The repository already includes the **preprocessed dataset used in the experiments**.

---

## 2. Open the Notebook in Google Colab

1. Go to **https://colab.research.google.com**
2. Click **File → Upload Notebook**
3. Upload the `.ipynb` notebook from this repository.

---

## 3. Upload the Dataset to Colab

Since the dataset is included in the repository, upload it manually to the Colab environment.

Run the following cell:

```python
from google.colab import files
uploaded = files.upload()
```

Then select the **processed dataset file included in this repository**.

---

## 4. Load the Dataset

After uploading the dataset, load it in the notebook:

```python
import pandas as pd

dataset = pd.read_csv("dataset.csv")
```

Replace `"dataset.csv"` with the correct filename.

---

## 5. Run the Notebook

Once the dataset is uploaded:

1. Click **Runtime → Run all**
2. The notebook will automatically:

   * load the dataset
   * perform preprocessing
   * handle class imbalance using **Random Undersampling**
   * scale the features
   * train the single-neuron model
   * evaluate the model
   * run the **PCA dimensionality reduction experiment**

---

# Required Libraries

The notebook uses the following Python libraries:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
```

One additional dependency must be installed in Colab:

```python
!pip install scikit-optimize
```

---

# Notes

* The dataset included in this repository is **a preprocessed version of the original Kaggle dataset**.
* The preprocessing was done beforehand to make the dataset suitable for experimentation.
* Fraud detection datasets are typically **highly imbalanced**, which is why **Random Undersampling** was applied during training.
* PCA was also tested to evaluate the impact of **dimensionality reduction on model performance**.

---

# Authors

* Camila – [[GitHub repository link](https://github.com/camilagzzaa/04-single-neuron.git)]
* Team Member – [GitHub repository link]
* Team Member – [GitHub repository link]

