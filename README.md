# 💳 AI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end **Machine Learning system** that detects fraudulent credit card transactions using transaction data and neural networks.

This project demonstrates a **real-world fraud detection pipeline**, including data preprocessing, exploratory data analysis, model training, and evaluation.

---

# 🚀 Project Highlights

✔ Real-world **financial fraud detection problem**
✔ Handling **highly imbalanced dataset**
✔ Neural Network model built with **TensorFlow**
✔ Modular **Machine Learning pipeline**
✔ Fraud **risk prediction system**

---

# 🎯 Project Objective

Financial institutions process millions of transactions daily. Detecting fraudulent activity quickly is essential to prevent financial loss.

This project builds an **AI-powered fraud detection system** capable of:

* analyzing transaction data
* detecting suspicious patterns
* predicting fraudulent transactions
* generating fraud risk scores

---

# 📊 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

Source: Kaggle

Dataset characteristics:

* **284,807 transactions**
* **492 fraudulent transactions**
* Highly **imbalanced dataset**

This imbalance makes fraud detection a **challenging machine learning problem**.

Dataset link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 📊 Exploratory Data Analysis

The dataset was analyzed to understand transaction patterns and fraud distribution.

## Fraud vs Normal Transactions

## Transaction Amount Distribution

---

# 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow

---

# 🏗 Project Architecture

```
fraud-ai-system/

data/
   creditcard.csv

src/
   data_loader.py
   preprocessing.py
   visualization.py
   model.py
   trainer.py
   evaluator.py

utils/
   metrics.py

main.py
requirements.txt
README.md
```

The project follows a **modular machine learning pipeline design** to separate responsibilities and improve maintainability.

---

# 🤖 Machine Learning Model

A **Neural Network** implemented using TensorFlow.

Model architecture:

Input Layer
↓
Dense (32) – ReLU
↓
Dense (16) – ReLU
↓
Dense (8) – ReLU
↓
Output Layer (Sigmoid)

Loss Function:

Binary Cross Entropy

Optimizer:

Adam

---

# 📉 Model Training

The model was trained using:

* **80/20 train-test split**
* **10 epochs**
* **batch size = 2048**

Training performance example:

---

# 📊 Model Evaluation

Due to class imbalance, **accuracy alone is not a reliable metric**.

The system evaluates performance using:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* Classification Report

These metrics are essential in fraud detection systems.

---

# ⚠ Fraud Risk Prediction

The trained model can evaluate new transactions and generate a **fraud risk score**.

Example output:

```
Transaction Amount: $320
Transaction Time: 18:42

Fraud Risk Score: 0.92

⚠ Fraud Detected
```

This simulates how real fraud detection systems work in financial institutions.

---

# 🚀 How to Run the Project

Clone the repository:

```
git clone <repo-link>
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the project:

```
python main.py
```

---

# 📈 Future Improvements

Potential enhancements for the system:

* ROC Curve visualization
* Precision-Recall curve
* SMOTE for class imbalance handling
* Feature importance analysis
* Fraud detection dashboard
* Real-time transaction monitoring

---

# 🏆 Resume Description

**AI Fraud Detection System**

* Developed a neural network model to detect fraudulent credit card transactions
* Performed exploratory data analysis on highly imbalanced financial data
* Implemented preprocessing and feature scaling
* Evaluated model performance using precision and recall metrics
* Built a modular machine learning pipeline using Python and TensorFlow

---

# 👩‍💻 Author

Developed as part of the **#DataScienceBootcamp Capstone Project**.

