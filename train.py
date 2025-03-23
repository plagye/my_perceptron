import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from utils import load_data, plot_decision_regions, accuracy

def train_perceptron(X, y, learning_rate=0.01, n_iterations=1000, random_state=1):
    model = Perceptron(learning_rate=learning_rate, n_iterations=n_iterations, random_state=random_state)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy_score = accuracy(y, y_pred)

    print(f'Training completed in {model.n_iterations} iterations.')
    print(f'Final weights: {model.weights}')
    print(f'Final bias: {model.bias}')
    print(f'Accuracy: {accuracy_score:.4f}')

    return model, accuracy_score

def train_with_visualization(X, y, learning_rate=0.01, n_iterations=1000, random_state=1):
    if X.shape[1] != 2:
        raise ValueError('This function is only implemented for 2D data.')
    
    model, accuracy_score = train_perceptron(X, y, learning_rate=learning_rate, n_iterations=n_iterations, random_state=random_state)

    plt.figure(figsize=(10, 6))
    plot_decision_regions(X, y, model, title=f'Perceptron Decision Boundary (Accuracy: {accuracy_score:.4f})')
    plt.tight_layout()
    plt.show()

    return model

def split_data(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]

    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
