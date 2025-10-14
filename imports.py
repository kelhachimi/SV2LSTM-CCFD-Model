# Standard Libraries

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

from collections import Counter

# Data Manipulation & Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from scipy import stats
from scipy.stats import shapiro

# TensorFlow / Keras

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Recall

# Scikit-learn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, IsolationForest, RandomForestClassifier, 
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             mean_squared_error, roc_auc_score, roc_curve,
                             classification_report, cohen_kappa_score, confusion_matrix,
                             fbeta_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# Imbalanced-learn

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# Gradient Boosting

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Optional / Misc

import xgboost as xgb
