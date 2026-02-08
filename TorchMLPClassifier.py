import torch
from torch import nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', epochs=50, lr=0.01, batch_size=32):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if self.activation not in activation_map:
            raise ValueError(f"Unsupported activation: {self.activation}")

        layers = []
        in_features = self.input_size

        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(activation_map[self.activation])
            in_features = hidden

        layers.append(nn.Linear(in_features, self.output_size))
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.classes_ = np.unique(y)  # <-- sklearn standardı

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values  # 💡 numpy array'e çevir
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X)
            return torch.argmax(logits, dim=1).numpy()

    def predict_proba(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def score(self, X, y):
        import pandas as pd
        from sklearn.metrics import accuracy_score

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        return accuracy_score(y, self.predict(X))

def build_mlp_classifier(input_size, hidden_sizes, output_size, **kwargs):
    return TorchMLPClassifier(input_size, hidden_sizes, output_size, **kwargs)