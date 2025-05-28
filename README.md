# TML25_A1_19 [Membership Inference Attack]

This repository investigates **membership inference attacks (MIAs)** by training a classifier to predict whether a data point was part of a model’s training dataset — using only the model's output logits. The project focuses on feature engineering from model outputs (specifically ResNet18) and evaluating the effectiveness of a logistic regression-based attack model.

---

##  Project Overview

### 1. Exploratory Data Analysis (EDA)
We began by performing EDA on both **public and private datasets**, focusing on:
- Label distribution
- Membership status (member vs. non-member) distribution  
This helped maintain a realistic data split (approximately 80:20) across different groups.

### 2. Attack Pipeline

#### Assumption:
We already know the target model and architecture (**ResNet18**), so we skip training a shadow model. Instead, we directly engineer features from the model's output logits and train an attack model to infer membership.

#### Feature Engineering:
Extracted from the model logits:
- `logit_std`: Standard deviation of the logits
- `prediction_confidence`: Confidence score of the top predicted class
- `prediction_entropy`: Entropy of the predicted probability distribution
- `correct_prediction`: Binary flag (1 if correct, 0 otherwise)
- `top1_minus_top2`: Difference between top-1 and top-2 logits
- `margin_confidence`: Logit margin between true and highest incorrect class
- `true_class_rank`: Rank of the true class logit in the prediction vector

#### Training the Attack Model:
- Split public dataset to mimic original 80-20 distribution across membership and class labels.
- Trained a **logistic regression model** with **GridSearchCV** to tune:
  - Number of iterations
  - Regularization constant (C)
  - Solver
- Evaluated on private (test) data.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| AUC | **0.6413** |
| TPR @ FPR = 0.05 | **0.0770** |
| Test Accuracy | **0.6075** |

### Classification Report
          precision    recall  f1-score   support

       0       0.67      0.42      0.52      1999
       1       0.58      0.79      0.67      2001


---

## Observations

### 1. Severe Recall Drop for Class 0 (Non-Members)
- **Recall of 42%** indicates the model misses a large portion of non-members.
- Overconfidence in predicting members suggests insufficient feature separation between classes.

### 2. Underfitting or Poor Feature Discrimination
- Most features are derived from logits — primarily reflecting **model confidence**.
- These features may **generalize poorly**, especially under distribution shifts or for unseen samples.

---

## Limitations
- Feature engineering was primarily limited to **confidence-based metrics** derived from logits (e.g., entropy, confidence, margins), which may not capture nuanced patterns distinguishing members from non-members.
- The attack model was restricted to **logistic regression**, which lacks the capacity to model complex decision boundaries.
- Did not explore **non-linear relationships** or interactions between features that could enhance predictive performance.
- Computational and time constraints prevented testing of **more sophisticated feature extraction pipelines** or ensemble models.

---

## Future Work
- Explore **advanced feature engineering**, including:
  - Gradient-based features
  - Temporal/logit evolution (if applicable)
  - Class-wise statistical embeddings
- Incorporate **non-linear models** such as:
  - **Random Forests**
  - **XGBoost**
  - **Multi-layer Perceptrons (MLPs)**
  - **Maximum Likelihood Estimators (MLE)**
- Evaluate feature importance and model interpretability to identify which logits-derived features contribute most to attack success.
