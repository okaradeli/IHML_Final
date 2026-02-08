from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import TorchMLPClassifier

from deslib.des.meta_des import METADES
from deslib.des.knora_e import KNORAE
from deslib.static.stacked import StackedClassifier
#from deslib.des.base import BaseDES
from deslib.static.stacked import StackedClassifier
from deslib.des.meta_des import BaseDES

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
import numpy as np
import data_loader as dataloader
import globals as glb

#MAGIC DATASET
#XX,y,target_column = dataloader.get_magic_dataset()
#X=XX.to_numpy()

#HIGGS BOSON DATASET
XX,yy,target_column = dataloader.get_higgs_boson_dataset()
X=XX.to_numpy()
y=yy.to_numpy()

# BADDEBT (LOAN DEFAULT) Dataset
#XX,yy,target_column = dataloader.get_baddebit_dataset()
#X=XX.to_numpy()
#y=yy.to_numpy()

print("Dataset:"+str(glb.DATASET+" Train+Validation Dataset Size:"+str(len(X))))

# 5-Fold CV
#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

#for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Base classifier pool (fresh each fold)
    pool = [
        DecisionTreeClassifier(),
        DecisionTreeClassifier(),
        LGBMClassifier(verbose=-1, random_state=102),
        TorchMLPClassifier.build_mlp_classifier(input_size=32, hidden_sizes=[64, 32], output_size=2, epochs=20),
        RandomForestClassifier(),
        XGBClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
    ]
    for clf in pool:
        clf.fit(X_train, y_train)

    # Fit META-DES
    #des_algo=StackedClassifier(pool_classifiers=pool,passthrough=True)
    des_algo = METADES(pool_classifiers=pool)
    # Initialize the DES model
    #des_algo = KNORAE(pool)

    #des_algo = BaseDES(pool_classifiers=pool)
    des_algo.fit(X_train, y_train)

    # Predict
    y_pred = des_algo.predict(X_test)


    #SCORER SETTING

    #scorer = precision_score(y_test, y_pred)
    #scorer = f1_score(y_test, y_pred)
    #scorer = roc_auc_score(y_test, y_pred)

    #Accuracy
    scorer = accuracy_score(y_test, y_pred)
    fold_scores_acc = []
    fold_scores_acc.append(scorer)
    #print(f"Fold {fold} Accuracy Score: {scorer:.4f}")

    #Precision
    scorer = precision_score(y_test, y_pred)
    fold_scores_prec = []
    fold_scores_prec.append(scorer)
    #print(f"Fold {fold} Precision Score: {scorer:.4f}")

    #Recall
    scorer = recall_score(y_test, y_pred)
    fold_scores_recall = []
    fold_scores_recall.append(scorer)
    #print(f"Fold {fold} Recall Score: {scorer:.4f}")

    #F1
    scorer = f1_score(y_test, y_pred)
    fold_scores_f1 = []
    fold_scores_f1.append(scorer)
    #print(f"Fold {fold} F1 Score: {scorer:.4f}")

    #ROC_AUC
    scorer = roc_auc_score(y_test, y_pred)
    fold_scores_roc = []
    fold_scores_roc.append(scorer)
    #print(f"Fold {fold} ROC AUC Score: {scorer:.4f}")

# Final result
print(f"\nMean CV Accuracy Score: {np.mean(fold_scores_acc):.4f}")
print(f"\nMean CV Precision Score: {np.mean(fold_scores_prec):.4f}")
print(f"\nMean CV Recall Score: {np.mean(fold_scores_recall):.4f}")
print(f"\nMean CV F1 Score: {np.mean(fold_scores_f1):.4f}")
print(f"\nMean CV ROC AUC Score: {np.mean(fold_scores_roc):.4f}")