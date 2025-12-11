import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []   
    
    def fit(self, X, y):
            n_samples, n_features = X.shape  
            w = np.ones(n_samples) / n_samples 

            for _ in range(self.n_estimators):
                model = DecisionTreeClassifier(max_depth=1)  
                model.fit(X, y, sample_weight=w)  
                predictions = model.predict(X)  

                err = np.sum(w * (predictions != y)) / np.sum(w)

                alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

                self.models.append(model) 
                self.alphas.append(alpha)  

                w *= np.exp(-alpha * y * predictions)  
                w /= np.sum(w)

    def predict(self, X):
            strong_preds = np.zeros(X.shape[0])  

            for model, alpha in zip(self.models, self.alphas):
                predictions = model.predict(X)  
                strong_preds += alpha * predictions  

            return np.sign(strong_preds).astype(int)