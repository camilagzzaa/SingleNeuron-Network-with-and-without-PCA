# 04 Single Neuron Model - Fraud Detection

## Overview

This project implements a **single-neuron neural network** to detect fraudulent bank transactions. The notebook includes preprocessing, handling class imbalance using **Random Undersampling**, model training, evaluation, and an additional experiment using **PCA (Principal Component Analysis)**.

Original dataset source:
https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection

To upload the dataset to the repository, the processed dataset had to be **compressed** due to file size limitations. Therefore, the dataset is included in the repository as:

**`dataset_compressed.zip`**

---

# Repository Contents

The repository contains the following files:

* **Single-Neuron Network with and without PCA.ipynb** – Main notebook containing the full experiment.
* **dataset_compressed.zip** – Compressed version of the processed dataset used in the notebook.
* **README.md** – Instructions on how to run the project.

---

# How to Run the Notebook in Google Colab

## 1. Download the Repository

Download or clone this repository to your computer.

---

## 2. Extract the Dataset

Locate the file:

```
dataset_compressed.zip
```

Unzip the file on your computer. This will produce the **processed dataset file used in the notebook**.

---

## 3. Open the Notebook in Google Colab

1. Go to **https://colab.research.google.com**
2. Click **File → Upload Notebook**
3. Upload the notebook:

```
Single-Neuron Network with and without PCA.ipynb
```

---

## 4. Upload the Dataset to Colab

Once the notebook is open, upload the extracted dataset file by running:

```python
from google.colab import files
uploaded = files.upload()
```

Select the **dataset file that you extracted from `dataset_compressed.zip`**.

---

## 5. Load the Dataset

After uploading, load the dataset in the notebook:

```python
import pandas as pd

dataset = pd.read_csv("dataset_name.csv")
```

Replace `"dataset_name.csv"` with the name of the extracted dataset file.

---

## 6. Run the Notebook

Once the dataset is uploaded:

1. Click **Runtime → Run all**
2. The notebook will automatically:

   * load the dataset
   * preprocess the data
   * balance the training set using **Random Undersampling**
   * scale the features
   * train the **single-neuron neural network**
   * evaluate model performance
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

* The dataset in the repository is **not the original Kaggle dataset**, but a **preprocessed version used for the experiments**.
* The dataset was **compressed (`dataset_compressed.zip`) to allow it to be uploaded to the repository**.
* Fraud detection datasets are **highly imbalanced**, which is why **Random Undersampling** was applied during training.
* PCA was tested to evaluate the effect of **dimensionality reduction on model performance**.

---

---

# Authors

* Mariana Samperio – [[GitHub repository link](https://github.com/mariana-samperio-cuevas/SingleNeuron-Network-with-and-without-PCA.git)]
* Matteo Peroni – [GitHub repository link]
* Camila Gonzalez – [[GitHub repository link](https://github.com/camilagzzaa/SingleNeuron-Network-with-and-without-PCA.git)]


