import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.05, n_iterations=10000, lambda_reg=0.001):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250) # Tránh tràn số
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        # Gradient Descent
        for i in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Loss + L2 Regularization
            epsilon = 1e-15
            data_loss = - (1/n_samples) * np.sum(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
            reg_loss = (self.lambda_reg / (2*n_samples)) * np.sum(self.weights**2)
            
            if i % 100 == 0:
                self.loss_history.append(data_loss + reg_loss)
            
            # Gradient
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + (self.lambda_reg/n_samples) * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class SMOTE:
    def __init__(self, k_neighbors=5, random_state=42):
        self.k = k_neighbors
        self.rng = np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2: return X, y
            
        minority_class = unique[np.argmin(counts)]
        X_minority = X[y == minority_class]
        
        n_minority = len(X_minority)
        n_synthetic = np.max(counts) - n_minority
        
        if n_synthetic <= 0: return X, y
            
        # Tính khoảng cách KNN Vectorized (nhanh hơn loop)
        # (a-b)^2 = a^2 + b^2 - 2ab
        X2 = np.sum(X_minority**2, axis=1).reshape(-1, 1)
        D2 = X2 + X2.T - 2 * np.dot(X_minority, X_minority.T)
        D2 = np.maximum(D2, 0)
        
        # Tìm k hàng xóm gần nhất (bỏ qua chính nó ở index 0)
        neighbors_indices = np.argsort(D2, axis=1)[:, 1:self.k+1]
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = self.rng.integers(0, n_minority)
            sample = X_minority[idx]
            
            # Chọn ngẫu nhiên 1 hàng xóm
            nn_idx = self.rng.choice(neighbors_indices[idx])
            neighbor = X_minority[nn_idx]
            
            # Nội suy
            gap = self.rng.random()
            new_sample = sample + gap * (neighbor - sample)
            synthetic_samples.append(new_sample)
            
        X_res = np.vstack((X, np.array(synthetic_samples)))
        y_res = np.hstack((y, np.full(n_synthetic, minority_class)))
        
        # Shuffle dữ liệu
        perm = self.rng.permutation(len(X_res))
        return X_res[perm], y_res[perm]

def cross_val_score(model, X, y, n_folds=5, metric='accuracy'):
    n_samples = len(y)
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    scores = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))
        
        clone = LogisticRegression(model.lr, model.n_iter, model.lambda_reg)
        clone.fit(X[train_idx], y[train_idx])
        y_pred = clone.predict(X[val_idx])
        
        if metric == 'accuracy':
            scores.append(np.mean(y_pred == y[val_idx]))
            
    return np.array(scores)