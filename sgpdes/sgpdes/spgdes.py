import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Perceptron
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier

# Working well
class SGPDES(BaseEstimator, ClassifierMixin):
    def __init__(self, WMA=25, ESD=0.001, EL=0.1, KI=7, max_iter=100000, pool_classifiers=None, DESNumbNN=5, Selector_Mode="MODELBASEDRF", CONSENSUSTH=95, resultprint=False):
        self.WMA = WMA
        self.ESD = ESD
        self.EL = EL
        self.KI = KI
        self.max_iter = max_iter
        self.DESNumbNN = DESNumbNN
        self.Selector_Mode = Selector_Mode
        self.CONSENSUSTH = CONSENSUSTH
        self.pool_classifiers_ = pool_classifiers
        self.pool_classifiers = pool_classifiers
        self.resultprint = resultprint

    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y)

        # Creating the TR set
        TR = np.column_stack((X, y))

        # Run HSGP to generate prototypes
        R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes = self._hsgp_tracking(TR, self.WMA, self.ESD, self.EL, self.KI, self.max_iter)

        # Store prototype data for later use
        self.prototype_data_ = pd.DataFrame(np.array(R))

        # Generate the base classifier pool
        X = pd.DataFrame(X)
        y = pd.Series(y)

        prototype_data = np.array(R)
        prototype_data = pd.DataFrame(prototype_data)

        num_TR = len(TR)
        num_R = len(R)

        # Calculate the reduction rate
        reduction_rate = (num_TR - num_R) / num_TR

        self.clt_sgpdes_ = self._train_sgpdes(
            prototype_data=self.prototype_data_,
            X_train_meta=X,
            Y_train_meta=y,
            pool_classifiers=self.pool_classifiers_,
            DESNumbNN=self.DESNumbNN,
            Selector_Mode=self.Selector_Mode,
            resultprint=self.resultprint
        )

        return self, reduction_rate

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['prototype_data_', 'pool_classifiers_', 'clt_sgpdes_'])

        # Validate inputs
        X = check_array(X)
        X = pd.DataFrame(X)

        predictions = []

        for i in range(len(X)):
            final_prediction = self._predict_sgpdes(
                prototype_data=self.prototype_data_,
                X_test=X.iloc[i:i+1,:],
                pool_classifiers=self.pool_classifiers_,
                DESNumbNN=self.DESNumbNN,
                Selector_Mode=self.Selector_Mode,
                clt_sgpdes=self.clt_sgpdes_,
                CONSENSUSTH=self.CONSENSUSTH,
                resultprint=self.resultprint
            )
            predictions.append(final_prediction)

        predictions = [x[0] if isinstance(x, np.ndarray) else x for x in predictions]

        return predictions

    def score(self, X, y):
        check_is_fitted(self, ['prototype_data_', 'pool_classifiers_', 'clt_sgpdes_'])

        # Validate inputs
        X = check_array(X)
        X = pd.DataFrame(X)

        predictions = []

        for i in range(len(X)):
            final_prediction = self._predict_sgpdes(
            prototype_data=self.prototype_data_,
            X_test=X.iloc[i:i+1,:],
            pool_classifiers=self.pool_classifiers_,
            DESNumbNN=self.DESNumbNN,
            Selector_Mode=self.Selector_Mode,
            clt_sgpdes=self.clt_sgpdes_,
            CONSENSUSTH=self.CONSENSUSTH,
            resultprint=self.resultprint
          )
            predictions.append(final_prediction)

        predictions = [x[0] if isinstance(x, np.ndarray) else x for x in predictions]

        predictions = np.array(predictions)
        y_test = np.array(y)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    def _train_sgpdes(self, prototype_data, X_train_meta, Y_train_meta, pool_classifiers, DESNumbNN, Selector_Mode, resultprint):
        """
        Function to train a classifier using the SGP-DES approach based on data complexity and competence metrics.

        :param prototype_data: Prototype data used to calculate complexity metrics.
        :param X_train_meta: Training data.
        :param Y_train_meta: Training labels.
        :param pool_classifiers: Classifier pool to be used.
        :param DESNumbNN: Number of nearest neighbors for complexity metrics calculation.
        :param Selector_Mode: Selector mode to be used.
        :return: A trained model.
        """
        dsel_data_complexity_metrics = self._dsel_generation_complexity_metrics(prototype_data, X_train_meta, DESNumbNN, pool_classifiers, resultprint)
        dsel_data_complexity_metrics_competence = self._dsel_generation_complexity_metrics_competence(X_train_meta, Y_train_meta, dsel_data_complexity_metrics, self.pool_classifiers_)

        dsel_data_complexity_metrics_reset = dsel_data_complexity_metrics.reset_index(drop=True)
        dsel_data_complexity_metrics_competence_reset = dsel_data_complexity_metrics_competence.reset_index(drop=True)

        dsel_data_complexity_combined = pd.concat([dsel_data_complexity_metrics_reset, dsel_data_complexity_metrics_competence_reset], axis=1)
        dsel_data_complexity_combined = dsel_data_complexity_combined.drop(columns='classifier_id')

        if Selector_Mode == "MODELBASEDXGB":
            dtrain = xgb.DMatrix(dsel_data_complexity_combined.iloc[:, :-1], label=dsel_data_complexity_combined.iloc[:, -1])

            params = {
                'max_depth': 3,
                'eta': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }

            num_rounds = 100
            clt_sgpdes = xgb.train(params, dtrain, num_rounds)

        if Selector_Mode == "MODELBASEDSVM":
            svm_pipeline = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='auto', probability=True))
            clt_sgpdes = svm_pipeline.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

        if Selector_Mode == "MODELBASEDKNN":
           knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
           clt_sgpdes = knn_pipeline.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

        if Selector_Mode == "MODELBASEDRF":
           rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
           clt_sgpdes = rf_model.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

        return clt_sgpdes

    def _predict_sgpdes(self, prototype_data, X_test, pool_classifiers, DESNumbNN, Selector_Mode, clt_sgpdes, CONSENSUSTH, resultprint=False):
        all_predictions = np.array([classifier.predict(X_test) for classifier in pool_classifiers]).T
        predictions = []
        consensus_predictions, count = mode(all_predictions, axis=1, keepdims=True)

        consensus_percentage = count / len(pool_classifiers) * 100
        overall_consensus_percentage = np.mean(consensus_percentage)

        if overall_consensus_percentage >= CONSENSUSTH:
           pred_scalar = mode(all_predictions, axis=None,keepdims=True)[0][0]
           return pred_scalar

        dsel_data_complexity_metrics_test = self._dsel_generation_complexity_metrics(prototype_data, X_test, DESNumbNN, pool_classifiers, resultprint)

        if Selector_Mode == "MODELBASEDXGB":
            dtest = xgb.DMatrix(dsel_data_complexity_metrics_test)
            preds = clt_sgpdes.predict(dtest)
            predictions = np.where(preds > 0.5, 1, 0)

        if Selector_Mode == "MODELBASEDSVM":
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)

        if Selector_Mode == "MODELBASEDKNN":
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)

        if Selector_Mode == "MODELBASEDRF":
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)

        selected_predictions = []

        for idx, prediction1 in enumerate(predictions):
            if prediction1 == 0:
                pred = pool_classifiers.estimators_[idx].estimator.predict(X_test)
                selected_predictions.extend(pred)

        if not selected_predictions:
            pred_scalar = mode(all_predictions, axis=None,keepdims=True)[0][0]
            return pred_scalar

        all_predictions = np.array(selected_predictions)
        global_majority_vote =  mode(all_predictions, axis=None,keepdims=True)[0][0]

        return global_majority_vote

    def _dsel_generation_complexity_metrics(self, prototype_data, X_complexy, DESNumbNN, pool_classifiers, resultprint):
        import time
        X_prototype = prototype_data.iloc[:, :-1]
        y_prototype = prototype_data.iloc[:, -1]
        complexity_metrics_list = []
        complexity_metrics_list_df = pd.DataFrame()
        complexity_metrics_df = pd.DataFrame()
        nbrs = NearestNeighbors(n_neighbors=DESNumbNN, algorithm='auto').fit(X_prototype)

        times_neighbors = []
        times_prototype_processing = []
        times_distance_calculation = []
        times_metric_calculation = []

        for x in range(len(X_complexy)):
            start_neighbors = time.time()
            _, indices = nbrs.kneighbors([X_complexy.iloc[x, :]])
            end_neighbors = time.time()
            times_neighbors.append(end_neighbors - start_neighbors)

            start_prototype_processing = time.time()
            nearest_prototypes = X_prototype.iloc[indices[0], :]
            centroid = nearest_prototypes.mean().values.reshape(1, -1)
            centroid_full = centroid
            end_prototype_processing = time.time()
            times_prototype_processing.append(end_prototype_processing - start_prototype_processing)

            start_distance_calculation = time.time()
            distance_X_Centroid = distance.cdist([X_complexy.iloc[x, :]], centroid, 'euclidean')
            end_distance_calculation = time.time()
            times_distance_calculation.append(end_distance_calculation - start_distance_calculation)

            start_metric_calculation = time.time()
            metrics_int = self._scores_BaseClassifiersPYTHON5(
                X_localRegion=nearest_prototypes,
                y_localRegion=y_prototype.iloc[indices[0]],
                X_prev=X_complexy.iloc[x, :],
                n_classifier=len(pool_classifiers),
                pool_classifiers=pool_classifiers,
            )
            end_metric_calculation = time.time()
            times_metric_calculation.append(end_metric_calculation - start_metric_calculation)

            metrics_int = np.array(metrics_int)
            centroid = np.repeat(distance_X_Centroid[:, [0]], len(metrics_int))

            df_distance_X_Centroid = pd.DataFrame(centroid, columns=['distance'])
            df_metrics_int = pd.DataFrame(metrics_int, columns=[f'metric_{i}' for i in range(metrics_int.shape[1])])
            df_concatenado = pd.concat([df_metrics_int, df_distance_X_Centroid], axis=1)
            num_rows = df_concatenado.shape[0]
            centroid_repeated = np.tile(centroid_full, (num_rows, 1))
            centroid_df = pd.DataFrame(centroid_repeated)

            df_concatenado = pd.concat([df_concatenado,centroid_df], axis=1)

            complexity_metrics_df = pd.concat([complexity_metrics_df, df_concatenado], axis=0)
            complexity_metrics_df = pd.DataFrame(complexity_metrics_df, columns=[f'metric_{i}' for i in range(metrics_int.shape[1])])

        average_time_neighbors = sum(times_neighbors) / len(times_neighbors)
        average_time_prototype_processing = sum(times_prototype_processing) / len(times_prototype_processing)
        average_time_distance_calculation = sum(times_distance_calculation) / len(times_distance_calculation)
        average_time_metric_calculation = sum(times_metric_calculation) / len(times_metric_calculation)

        if resultprint == True:
            print(f"Average time for neighbor search: {average_time_neighbors} seconds")
            print(f"Average time for prototype processing: {average_time_prototype_processing} seconds")
            print(f"Average time for distance calculation: {average_time_distance_calculation} seconds")
            print(f"Average time for metric calculation: {average_time_metric_calculation} seconds")

        return complexity_metrics_df

    def _dsel_generation_complexity_metrics_competence(self, X_train, Y_train,complexity_metrics_df, pool_classifiers):
        dsel_data_predict_list = []
        classifiers_id_list = []

        for x in range(len(X_train)):
            for m in range(len(pool_classifiers)):
                dsel_data_predict_list.append(1 - (pool_classifiers.estimators_[m].estimator.predict(X_train.iloc[x, :].values.reshape(1, -1)) == Y_train.iloc[x]).astype(int))
                classifiers_id_list.append(m)

        dsel_data_predict_df = pd.DataFrame(dsel_data_predict_list, columns=['competence'])
        classifiers_id_df = pd.DataFrame(classifiers_id_list, columns=['classifier_id'])
        dsel_data_complexity_predict = pd.concat([classifiers_id_df, dsel_data_predict_df], axis=1)

        return dsel_data_complexity_predict

    def _scores_BaseClassifiersPYTHON5_optimized(self, X_localRegion, y_localRegion, X_prev, n_classifier, pool_classifiers):
        X_localRegion = np.asarray(X_localRegion)
        y_localRegion = np.asarray(y_localRegion)
        X_prev_np = X_prev.values.reshape(1, -1)
        X_prev_np = np.atleast_2d(X_prev.values)

        results = np.zeros((n_classifier, 6))

        decisions = [clf.estimator.decision_function(X_prev_np) for clf in pool_classifiers.estimators_]
        scores = [clf.estimator.score(X_localRegion, y_localRegion) for clf in pool_classifiers.estimators_]
        class_predictions = [clf.estimator.predict(X_prev_np) for clf in pool_classifiers.estimators_]
        prob_supports = [clf.predict_proba(X_localRegion) for clf in pool_classifiers.estimators_]
        max_probs = [prob.max(axis=1) for prob in prob_supports]
        overall_supports = [max_prob.mean() for max_prob in max_probs]

        for idx, clf in enumerate(pool_classifiers.estimators_):
            mask = y_localRegion == class_predictions[idx]
            relevant_data = X_localRegion[mask]

            if relevant_data.size > 0:
                relevant_labels = y_localRegion[mask]
                class_score = clf.estimator.score(relevant_data, relevant_labels)
                class_decision = clf.estimator.decision_function(relevant_data)
                class_prob_support = clf.predict_proba(relevant_data).max(axis=1).mean()
                class_decision_mean = class_decision.mean()
            else:
                class_score = 0
                class_decision_mean = 0
                class_prob_support = 0

            results[idx] = [
                overall_supports[idx],
                class_prob_support,
                scores[idx],
                class_score,
                class_decision_mean,
                decisions[idx].mean()
            ]

        return results

    def _scores_BaseClassifiersPYTHON5(self, X_localRegion, y_localRegion, X_prev, n_classifier, pool_classifiers):
        X_localRegion = np.asarray(X_localRegion)
        y_localRegion = np.asarray(y_localRegion)
        X_prev_np = X_prev.values.reshape(1, -1)

        results = np.zeros((n_classifier, 6))

        decisions = np.array([clf.estimator.decision_function(X_prev_np) for clf in pool_classifiers.estimators_[:n_classifier]])
        scores = np.array([clf.estimator.score(X_localRegion, y_localRegion) for clf in pool_classifiers.estimators_[:n_classifier]])
        class_predictions = np.array([clf.estimator.predict(X_prev_np) for clf in pool_classifiers.estimators_[:n_classifier]])

        prob_supports = np.array([clf.predict_proba(X_localRegion) for clf in pool_classifiers.estimators_[:n_classifier]])
        max_probs = prob_supports.max(axis=2)
        overall_supports = max_probs.mean(axis=1)

        for idx in range(n_classifier):
            clf = pool_classifiers[idx]
            mask = y_localRegion == class_predictions[idx]
            relevant_data = X_localRegion[mask]

            if relevant_data.size > 0:
                class_score = clf.estimator.score(relevant_data, y_localRegion[mask])
                class_decision = clf.estimator.decision_function(relevant_data)
                class_prob_support = clf.predict_proba(relevant_data).max(axis=1).mean()
                class_decision_mean = class_decision.mean()
            else:
                class_score = 0
                class_decision_mean = 0
                class_prob_support = 0

            results[idx] = [
                overall_supports[idx],
                class_prob_support,
                scores[idx],
                class_score,
                class_decision_mean,
                decisions[idx].mean()
            ]

        return results

    def _predict_BaseClassifiersPYTHON4(self,X_prev,n_classifier,pool_classifiers):
        score_all_prev = np.zeros((0), float)
        score_all_prev_np = np.zeros((0), float)

        for x in range(int(n_classifier)):
          score_all_prev = pool_classifiers.estimators_[x].estimator.predict(X_prev)
          score_all_prev_np = np.append(score_all_prev_np, np.array(score_all_prev), axis=None)

        score_all_prev = score_all_prev_np
        return score_all_prev

    def _hsgp_tracking(self,TR, WMA, ESD, EL, KI, max_iter):
        S_Geral = [TR.copy()]  # Initializing S_Geral with the TR set
        E = []  # Entropies
        R = []  # Result
        iter_num = 1
        sma_values = []  # SMA values
        sd_values = []  # Standard Deviation values
        num_prototypes = 0  # Prototype count
        prototypes = []  # List to store prototypes
        sma_values_rep = []  # Replicated SMA values
        average_entropies = []  # Average entropies of subsets
        accuracy_TR = 0
        accuracy_R = 0

        while iter_num <= max_iter:
            S_Geral_with_classes = [s for s in S_Geral if len(np.unique(s[:, -1])) > 1]
            if not S_Geral_with_classes:
                break

            S_L = max(S_Geral_with_classes, key=lambda subset: subset.shape[0])
            S_Geral = [s for s in S_Geral if not np.array_equal(s, S_L)]

            centroid = self._calculate_centroid(S_L)
            threshold = np.median(euclidean_distances(S_L[:, :-1], centroid.reshape(1, -1)))
            S_1, S_2 = self._split(S_L, centroid, threshold)

            if S_1.size > 0:
                S_Geral.append(S_1)
            if S_2.size > 0:
                S_Geral.append(S_2)

            entropies = [self._calculate_entropy(subset, np.max(TR[:, -1]) + 1) for subset in S_Geral]
            average_entropy = np.mean(entropies)
            average_entropies.append(average_entropy)

            e_i = average_entropy
            E.append(e_i)

            if WMA <= iter_num:
                sma_value = self._sma(E, WMA)
                sma_values.append(sma_value)
                sd_value = self._standard_deviation(E, WMA, sma_value)
                sd_values.append(sd_value)
                sma_values_rep.append(sma_value)
                proto_gen = self._is_proto_generating(WMA, sma_value, E, ESD)
                if proto_gen:
                    break
            iter_num += 1

        for subset in S_Geral:
            if subset.size > 0:
                centroid = self._calculate_centroid(subset)
                entropy_subset = self._calculate_entropy(subset, np.max(TR[:, -1]) + 1)
                if entropy_subset < EL:
                    I = self._is_instance_selecting([subset], centroid, TR, EL, KI)
                    if I.size > 0:
                        num_prototypes += 1
                        R.extend(I)
                        prototypes.append(centroid)
            else:
                print("Empty subset found, skipping calculations.")

        R_array = np.array(R)

        accuracy_TR, accuracy_R = self._calculate_accuracy(TR, R_array)
        accuracy_TR *= 100
        accuracy_R *= 100
        reduction_rate = len(R) / len(TR) * 100

        return R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes

    # Necessary functions
    def _calculate_centroid(self, subset):
        return np.mean(subset[:, :-1], axis=0)

    def _split(self, subset, centroid, threshold):
        distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
        S_1 = subset[distances[:, 0] <= threshold]
        S_2 = subset[distances[:, 0] > threshold]
        return S_1, S_2

    def _select_largest_subset(self, S_1, S_2):
        diameter_1 = np.max(euclidean_distances(S_1[:, :-1], S_1[:, :-1]))
        diameter_2 = np.max(euclidean_distances(S_2[:, :-1], S_2[:, :-1]))
        return S_1 if diameter_1 > diameter_2 else S_2

    def _is_instance_selecting(self,subsets, centroid, training_set, entropy_level, k):
        selected_instances = []
        for subset in subsets:
            if self._calculate_entropy(subset, np.max(training_set[:, -1]) + 1) <= entropy_level:
                distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
                nearest_indices = np.argsort(distances, axis=0)[:k].flatten()
                selected_instances.extend(subset[nearest_indices])
        return np.array(selected_instances)

    def _calculate_entropy(self, subset, num_classes):
        if len(subset) == 0:
            return 0
        num_classes = int(num_classes)
        class_counts = np.bincount(subset[:, -1].astype(int), minlength=num_classes)
        probabilities = class_counts / np.sum(class_counts)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        normalized_entropy = entropy / np.log2(num_classes)
        return normalized_entropy if not np.isneginf(normalized_entropy) else 0

    def _sma(self, entropy_values, window_size):
        return np.mean(entropy_values[-window_size:])

    def _standard_deviation(self,entropy_values, window_size, sma_value):
        return np.sqrt(np.sum((entropy_values[-window_size:] - sma_value) ** 2) / (window_size - 1))

    def _is_proto_generating(self,window_size, sma_value, entropy_values, esd):
        sd_value = self._standard_deviation(entropy_values, window_size, sma_value)
        return sd_value < esd

    def _calculate_accuracy(self,TR, R):
        X_TR = TR[:, :-1]
        y_TR = TR[:, -1]
        X_train, X_val, y_train, y_val = train_test_split(X_TR, y_TR, test_size=0.1, stratify=y_TR, random_state=42)

        knn_TR = KNeighborsClassifier(n_neighbors=1)
        knn_TR.fit(X_train, y_train)
        accuracy_TR = knn_TR.score(X_val, y_val)

        accuracy_R = 0
        if len(R) > 0:
            X_R = R[:, :-1]
            y_R = R[:, -1]

            knn_R = KNeighborsClassifier(n_neighbors=1)
            knn_R.fit(X_R, y_R)
            accuracy_R = knn_R.score(X_val, y_val)

        return accuracy_TR, accuracy_R
