class PoolGenerator:
    def __init__(self, n_classifier):
        from sklearn.linear_model import Perceptron
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import BaggingClassifier

        self.n_classifier = int(n_classifier)
        self.model = CalibratedClassifierCV(Perceptron(max_iter=100))
        self.pool_classifiers = BaggingClassifier(self.model, n_estimators=self.n_classifier)

    def PoolGeneration(self, X_train, X_test, y_train, y_test):
        from sklearn.model_selection import train_test_split
        import numpy as np

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)
        
        # Train a pool of classifiers
        self.pool_classifiers.fit(X_train, y_train)
        
        for x in range(self.n_classifier):
            self.pool_classifiers.estimators_[x].estimator.fit(X_train, y_train)
        
        return self.pool_classifiers
