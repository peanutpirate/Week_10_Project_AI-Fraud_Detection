# 💳 AI Fraud Detection System

An end-to-end **machine learning system** that detects fraudulent credit card transactions using neural networks and transaction data analysis.

This project demonstrates a **real-world financial fraud detection pipeline**, including data preprocessing, exploratory data analysis, model training, and evaluation.

---

# 🎯 Project Objective

Financial institutions process millions of transactions every day. Detecting fraudulent activity quickly is critical to prevent financial loss.

This project builds an **AI-powered fraud detection system** that:

* analyzes transaction data
* detects suspicious patterns
* predicts whether a transaction is fraudulent

---

# 📊 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

Source: Kaggle

Dataset size:

* **284,807 transactions**
* **492 fraudulent transactions**
* Highly **imbalanced dataset**

This makes fraud detection a **challenging real-world machine learning problem**.

Dataset link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* TensorFlow
* Scikit-learn

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

The project is designed using a **modular machine learning pipeline** structure.

---

# 🔍 Exploratory Data Analysis

Key analyses performed:

* Fraud vs Normal transaction distribution
* Transaction amount distribution
* Dataset imbalance analysis
* Transaction patterns

Example visualizations include:

* Fraud distribution chart
* Transaction amount histogram

---

# 🤖 Machine Learning Model

A **Neural Network** was implemented using TensorFlow.

Model architecture:

```
Input Layer
↓
Dense (32) - ReLU
↓
Dense (16) - ReLU
↓
Dense (8) - ReLU
↓
Output (Sigmoid)
```

Loss Function:

```
Binary Cross Entropy
```

Optimizer:

```
Adam
```

---

# 📉 Model Training

The model was trained using:

* **80/20 train-test split**
* **10 epochs**
* **batch size = 2048**

The training process records:

* training loss
* validation loss

These metrics help evaluate model performance.

---

# 📊 Model Evaluation

The system evaluates the model using:

* Confusion Matrix
* Precision
* Recall
* Classification Report

⚠ In fraud detection problems, **accuracy alone is not reliable** due to class imbalance.

More important metrics:

* **Precision**
* **Recall**
* **F1 Score**

---

# ⚠ Fraud Risk Prediction

The trained model can evaluate new transactions and produce a **fraud risk score**.

Example output:

```
Transaction Amount: $320

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

Run the system:

```
python main.py
```

---

# 📈 Future Improvements

Possible improvements for the system:

* ROC Curve analysis
* Precision-Recall curve
* SMOTE for class imbalance handling
* Feature importance analysis
* Fraud risk scoring dashboard

---

# 🏆 Resume Project Description

**AI Fraud Detection System**

* Developed a neural network model to detect fraudulent credit card transactions
* Performed exploratory data analysis on highly imbalanced financial data
* Implemented preprocessing and feature scaling
* Evaluated model performance using precision and recall metrics
* Built a modular machine learning pipeline using Python and TensorFlow

---

# 👩‍💻 Author

Data Science project developed as part of the Data Science Bootcamp w Gokce Cevik
