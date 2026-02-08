# Importing the SGPDES technique
from sgpdes.sgpdes.spgdes import SGPDES

# Perceptron PoolGenerator
from util.poolgenerator import PoolGenerator


import warnings
import requests
import zipfile
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import data_loader as dataloader
import globals as glb

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Function to download and unzip the dataset
def download_and_unzip(url, local_filename, extract_to):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as file:
            file.write(response.content)
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise Exception(f"Failed to download file from {url}")
    print(f"Downloaded and extracted: {local_filename}")

# Function to load dataset
def load_data(name):
    data_path = f"{name}/{name}.dat"
    data = pd.read_csv(data_path, comment='@', header=None)
    X = data.iloc[:, :-1].astype(float)
    y = pd.Categorical(data.iloc[:, -1].astype(str).str.strip(), categories=["positive", "negative"], ordered=True).codes
    return X, pd.Series(y)

# Function to evaluate each method
def evaluate_method(ctl, X_train, X_test, y_train, y_test, method_name, fold, ir):
    ctl.fit(X_train, y_train)
    y_pred = ctl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    print(f"Dataset Fold {fold}: Accuracy {method_name} = {accuracy:.3f}, F1 Score {method_name} = {f1:.3f}, IR = {ir:.3f}")
    return accuracy, f1

# Function to process each dataset
def process_dataset(dataset, methods):
    name = dataset["name"]
    #download_and_unzip(dataset["url"], f"{name}.zip", name)
    #X, y = load_data(name)

    XX, yy, target_column = dataloader.get_higgs_boson_dataset()
    X = XX.to_numpy()
    y = yy.to_numpy()

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    results = {method: {"accuracies": [], "f1_scores": []} for method in methods}
    imbalance_ratios, reduction_rates_sgpdes = [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):

        print("Fold:1")

        ##X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        ##y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        #ir = y_train.value_counts().max() / y_train.value_counts().min()

        import pandas as pd
        ir = pd.Series(y_train).value_counts().max() / pd.Series(y_train).value_counts().min()

        imbalance_ratios.append(ir)

        pool_generator = PoolGenerator(n_classifier=10)
        pool_classifiers = pool_generator.PoolGeneration(X_train, X_test, y_train, y_test)

        for method_name, ctl in methods.items():
            accuracy, f1 = evaluate_method(ctl(pool_classifiers), X_train, X_test, y_train, y_test, method_name, fold, ir)
            results[method_name]["accuracies"].append(accuracy)
            results[method_name]["f1_scores"].append(f1)

            if "SGPDES" in method_name:
                _, reduction_rate = ctl(pool_classifiers).fit(X_train, y_train)
                reduction_rates_sgpdes.append(reduction_rate)

    summary = []
    for method, metrics in results.items():
        summary.append({
            "Dataset": name,
            "Method": method,
            "Mean Accuracy": np.mean(metrics["accuracies"]),
            "Std Accuracy": np.std(metrics["accuracies"]),
            "Mean F1 Score": np.mean(metrics["f1_scores"]),
            "Std F1 Score": np.std(metrics["f1_scores"]),
            "Reduction Rate DSEL": np.mean(reduction_rates_sgpdes) if "SGPDES" in method else None
        })

    return summary, np.mean(imbalance_ratios)

# Dictionary of methods for dynamic instantiation
methods = {
    "SGPDES KNN": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDKNN", CONSENSUSTH=101, resultprint=False),
    "SGPDES RF": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDRF", CONSENSUSTH=101, resultprint=False),
    ##"SGPDES SVM": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDSVM", CONSENSUSTH=101, resultprint=False),
    "SGPDES XGB": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDXGB", CONSENSUSTH=101, resultprint=False)
}

# List of datasets to process
datasets = [
    {"name": "glass1", "url": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass1.zip"}
]

# Process all datasets and save results
all_results = []
for dataset in datasets:
    summary, mean_ir = process_dataset(dataset, methods)
    all_results.extend(summary)
    print(f"Dataset: {dataset['name']}, Mean IR of 5 folds = {mean_ir:.3f}")

# Save results to CSV
results_df = pd.DataFrame(all_results)
#results_df.to_csv("results.csv", index=False)

print(results_df)