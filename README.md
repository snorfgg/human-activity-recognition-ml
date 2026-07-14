# Human Activity Recognition using Machine Learning

Academic project developed for the **Feature Engineering for Computational Learning** course at the **University of Coimbra**.

## Overview

This project investigates **Human Activity Recognition (HAR)** using wearable sensor data collected from body-mounted sensors.

The project implements a complete machine learning pipeline, covering data preprocessing, feature engineering, feature selection, classification and performance evaluation. Several approaches were compared, including statistical features, dimensionality reduction techniques and deep feature embeddings.

---

## Features

- Statistical and frequency-domain feature extraction
- Data preprocessing and normalization
- Principal Component Analysis (PCA)
- ReliefF feature selection
- SMOTE data augmentation
- HARNet5 deep feature embeddings
- Custom k-Nearest Neighbours (k-NN) implementation
- Within-subject and between-subject evaluation
- Automatic model evaluation and performance comparison

---

## Technologies

- Python
- NumPy
- SciPy
- scikit-learn
- PyTorch
- Matplotlib
- Joblib

---

## Dataset

This project uses the **FORTH-TRACE Human Activity Recognition Dataset**.

The dataset is **not included** in this repository. Please download it from the official source and follow its licence terms.

---

## Project Structure

```
.
├── src/
│   ├── mainActivity.py
│   └── PartB.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Getting Started

Clone the repository

```bash
git clone https://github.com/<username>/human-activity-recognition-ml.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python src/mainActivity.py
```

or

```bash
python src/PartB.py
```

---

## Results

The implemented pipeline evaluates different Human Activity Recognition strategies using:

- Feature engineering
- PCA
- ReliefF
- HARNet5 embeddings
- Custom k-NN classifier

Performance is assessed through within-subject and between-subject evaluation protocols.

---

## Authors

- Diogo Freitas
- Rafael Bernardo

---

## Acknowledgements

This project was developed as part of the **Feature Engineering for Computational Learning** course at the **University of Coimbra**.

The project uses the **FORTH-TRACE Human Activity Recognition Dataset**, developed by researchers from FORTH and the University of Crete.
