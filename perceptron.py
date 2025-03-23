import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.00001, n_iterations=10000, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)

        n_features = X.shape[1]
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            errors = 0

            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            
            if errors == 0:
                break
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
