# make a regression prediction with an RFE pipeline
from numpy import mean, std
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

import globals as glb
import data_loader as dl

import logging
logger = logging.getLogger()

print("New Experiment is running...")
glb.print_global_parameters()

# -----------------------
# load dataset
# -----------------------


glb.EXPERIMENT_FEATURES=False
glb.DATASET_SAMPLE_SIZE=20000
#X, y, target_column = dl.get_higgs_boson_dataset()
X, y, target_column = dl.get_forest_dataset()
print(f"Data loaded, sample count: {len(X)}")

# -----------------------
# figure out feature names
# -----------------------
if isinstance(X, pd.DataFrame):
    feature_names = X.columns.tolist()
else:
    # fallback if X is a numpy array
    feature_names = [f"f{i}" for i in range(X.shape[1])]

print(f"Original feature count: {len(feature_names)}")
print(f"Original features: {feature_names}")

# -----------------------
# RFE + model pipeline
# -----------------------
n_select = min(28, len(feature_names))
#rfe = RFE(estimator=RandomForestClassifier(random_state=42),n_features_to_select=n_select)
rfe = RFE(estimator=RandomForestClassifier(random_state=42),n_features_to_select=n_select)


pipe = Pipeline(steps=[
    ("rfe", rfe),
])

# fit the pipeline
pipe.fit(X, y)

# get selected features
support_mask = pipe.named_steps["rfe"].get_support()
selected_features = [name for name, keep in zip(feature_names, support_mask) if keep]

print(f"Selected feature count after RFE: {len(selected_features)}")
print(f"Selected features: {selected_features}")

# (optional) reduced X if you need it later
if isinstance(X, pd.DataFrame):
    X_reduced = X[selected_features].copy()
else:
    X_reduced = X[:, support_mask]

# quick printouts as well (besides logging)
print("Original feature count:", len(feature_names))
print("Original features:", feature_names)
print("Selected feature count:", len(selected_features))
print("Selected features:", selected_features)