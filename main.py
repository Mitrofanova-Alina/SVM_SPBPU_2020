import numpy as np
from sklearn.datasets import make_blobs
from SVM import SVM

def linear_train():
    model = SVM(C=0.5, kernel='linear', gamma='auto', tol=1e-3)
    n_dim = 2
    length = 30
    x, y = make_blobs(n_samples=length, centers=2, random_state=6)
    y[y == 0] = -1
    print("-----------------------")
    print("Training set: ")
    print("x = \n", x)
    print("y = ", y)
    print("-----------------------")
    model.fit(x, y)
    model.find_support_vectors(x, y)
    model.linear_draw(x, y)

def non_linear_train():
    model = SVM(C=0.5, kernel='rbf', gamma='auto', tol=1e-3)
    n_dim = 2
    length = 30
    x, y = make_blobs(n_samples=length, centers=2, random_state=6)
    y[y == 0] = -1
    print("-----------------------")
    print("Training set: ")
    print("x = \n", x)
    print("y = ", y)
    print("-----------------------")
    model.fit(x, y)
    model.find_support_vectors(x, y)
    model.non_linear_draw(x, y)

if __name__ == "__main__":
    print("Hello world!")

    # linear_train()
    non_linear_train()

    print("Bye, world!")

