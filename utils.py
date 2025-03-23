import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_data(file_path, target_column=None, features=None):
    data = pd.read_csv(file_path)

    if target_column is None:
        target_column = data.columns[-1]

    y = data[target_column].values

    if features is None:
        features = [col for col in data.columns if col != target_column]

    X = data[features].values

    return X, y

def plot_decision_regions(X, y, classifier, resolution=0.02, title='Decision Boundary'):
    if X.shape[1] != 2:
        raise ValueError('This function is only implemented for 2D data.')
    
    markers = ('o', 's')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    return plt

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

