
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def HSGP(TR, WMA, ESD, EL, KI, max_iter):
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

        centroid = calculate_centroid(S_L)
        threshold = np.median(euclidean_distances(S_L[:, :-1], centroid.reshape(1, -1)))
        S_1, S_2 = split(S_L, centroid, threshold)

        if S_1.size > 0:
            S_Geral.append(S_1)
        if S_2.size > 0:
            S_Geral.append(S_2)

        entropies = [calculate_entropy(subset, np.max(TR[:, -1]) + 1) for subset in S_Geral]
        average_entropy = np.mean(entropies)
        average_entropies.append(average_entropy)

        e_i = average_entropy
        E.append(e_i)

        if WMA <= iter_num:
            sma_value = sma(E, WMA)
            sma_values.append(sma_value)
            sd_value = standard_deviation(E, WMA, sma_value)
            sd_values.append(sd_value)
            sma_values_rep.append(sma_value)
            proto_gen = is_proto_generating(WMA, sma_value, E, ESD)
            if proto_gen:
                break
        iter_num += 1

    for subset in S_Geral:
        if subset.size > 0:
            centroid = calculate_centroid(subset)
            entropy_subset = calculate_entropy(subset, np.max(TR[:, -1]) + 1)
            if entropy_subset < EL:
                I = is_instance_selecting([subset], centroid, TR, EL, KI)
                if I.size > 0:
                    num_prototypes += 1
                    R.extend(I)
                    prototypes.append(centroid)
        else:
            print("Empty subset found, skipping calculations.")

    R_array = np.array(R)

    accuracy_TR, accuracy_R = calculate_accuracy(TR, R_array)
    accuracy_TR *= 100
    accuracy_R *= 100
    reduction_rate = len(R) / len(TR) * 100

    return R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes

def calculate_centroid(subset):
    return np.mean(subset[:, :-1], axis=0)

def split(subset, centroid, threshold):
    distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
    S_1 = subset[distances[:, 0] <= threshold]
    S_2 = subset[distances[:, 0] > threshold]
    return S_1, S_2

def is_instance_selecting(subsets, centroid, training_set, entropy_level, k):
    selected_instances = []
    for subset in subsets:
        if calculate_entropy(subset, np.max(training_set[:, -1]) + 1) <= entropy_level:
            distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
            nearest_indices = np.argsort(distances, axis=0)[:k].flatten()
            selected_instances.extend(subset[nearest_indices])
    return np.array(selected_instances)

def calculate_entropy(subset, num_classes):
    if len(subset) == 0:
        return 0
    num_classes = int(num_classes)
    class_counts = np.bincount(subset[:, -1].astype(int), minlength=num_classes)
    probabilities = class_counts / np.sum(class_counts)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    normalized_entropy = entropy / np.log2(num_classes)
    return normalized_entropy if not np.isneginf(normalized_entropy) else 0

def sma(entropy_values, window_size):
    return np.mean(entropy_values[-window_size:])

def standard_deviation(entropy_values, window_size, sma_value):
    return np.sqrt(np.sum((entropy_values[-window_size:] - sma_value) ** 2) / (window_size - 1))

def is_proto_generating(window_size, sma_value, entropy_values, esd):
    sd_value = standard_deviation(entropy_values, window_size, sma_value)
    return sd_value < esd

def calculate_accuracy(TR, R):
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

