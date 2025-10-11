# SV2LSTM-CCFD-Model
SV2LSTM is a CCFD model that combines an optimized feature selection process with a hybrid model of soft voting and Autoencoder-LSTM. The model addresses key challenges in fraud detection, particularly the imbalance in datasets and the risks of overfitting, by using techniques such as resampling, outlier removal, and hyperparameter tuning. 

# Datasets

To prepare the code, first download the following Kaggle datasets, and store them into the 'data' folder:
- [Credit Card Fraud Detection dataset 2013](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
- [Credit Card Fraud Detection dataset 2023](https://www.kaggle.com/dsv/6492730)
- [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [IEEE-CIS](https://kaggle.com/competitions/ieee-fraud-detection)
- [Small Card Data](https://www.kaggle.com/datasets/shubhamjoshi2130of/abstract-data-set-for-credit-card-fraud-detection)

# Technologies Used

- **Programming Language:** Python
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Techniques:**
- **Soft voting based on DT, ET, RF, XGBoost, Adaboost:**
- **LSTM Neural Network:** Captures temporal dependencies in transaction data
- **ROS, AdaSyn, and SMOTE:** Addresses class imbalance by oversampling the minority class

# Project Workflow

## Data Preparation

Load and preprocess the credit card fraud dataset. Then, apply cleaning, normalization, and encoding. Detect highly correlated features and remove redundant ones. After that, the model handles class imbalance with resampling (e.g., oversampling,or hybrid sampling).

## Feature Reduction & Selection
Then, applies feature importance aggregation from base models.

## Base Learners (Level 1)
Train diverse base models to capture complementary decision patterns: XGBoost, ET, RF, DT, and Adaboost. Each model outputs probability predictions on both train/test sets.

## Meta-Feature Construction

Stack the outputs (probabilities) from all base models to form new datasets. These meta-features represent the learned consensus of the base models.
## Soft Voting Optimization

Use a VotingClassifier combined with  GridSearch to optimize weights for each base learner. The optimized ensemble provides balanced performance and smooth probability calibration.

## Meta-Learner (Level 2: LSTM)

### Meta-feature compression with AE


### Final LSTM classification

Final classification with LSTM which learns temporal/relational dependencies among base model predictions:

## Model Training & Callbacks

Train with validation monitoring and adaptive learning with ReduceLROnPlateau which makes the model learns robust ensemble dynamics.
## Evaluation & Generalization

Evaluate on both training and testing through different metrics including: accuracy, precision, recall, F1-score, AUC, MCC, Kappa value, G-mean, Sensitivity, and compute generalization gap.

## Final Output

A robust Stacked Voting + LSTM (SV2LSTM) model that combines classical ensemble diversity with deep sequential learning, that adapts dynamically via LSTM meta-learning and handles imbalance, and overfitting efficiently.

# How to Run the Project

Clone the repository:

```
git clone https://github.com/kelhachimi/SV2LSTM-CCFD-Model.git
```

Navigate to the project directory:

```
cd SV2LSTM-CCFD-Model
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Jupyter Notebook:

```
jupyter notebook CCFD_Model.ipynb
```
