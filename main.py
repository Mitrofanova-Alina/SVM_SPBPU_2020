import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, KFold

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
    model.draw_decision(x, y)


def circle_model():
    x = np.array([[0, 0],
                  [2, 0], [-2, 0], [0, 2], [0, -2],
                  [1, np.sqrt(3)], [1, -np.sqrt(3)], [-1, np.sqrt(3)], [-1, -np.sqrt(3)],
                  [1, 0], [-1, 0], [0, 1], [0, -1],
                  [0.5, np.sqrt(0.75)], [0.5, -np.sqrt(0.75)], [-0.5, np.sqrt(0.75)], [-0.5, -np.sqrt(0.75)],
                  [1, 1], [-1, 1], [-1, -1], [1, -1],
                  [4, 0], [3, 1], [2, 2], [1, 3],
                  [0, 4], [-3, 1], [-2, 2], [-1, 3],
                  [-4, 0], [-3, -1], [-2, -2], [-1, -3],
                  [0, -4], [3, -1], [2, -2], [1, -3],
                  [2, 3], [-2, 3], [-2, -3], [2, -3],
                  [3, 2], [-3, 2], [-3, -2], [3, -2],
                  [1, 4], [-1, 4], [-1, -4], [1, -4],
                  [4, 1], [-4, 1], [-4, -1], [4, -1]])
    y = np.array([1,
                  1, 1, 1, 1,
                  1, 1, 1, 1,
                  1, 1, 1, 1,
                  1, 1, 1, 1,
                  1, 1, 1, 1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1
                  ])
    return x, y


def non_linear_train():
    model = SVM(C=10, kernel='rbf', gamma=0.1, tol=1e-3)
    length = 20
    x, y = circle_model()
    print("-----------------------")
    print("Training set: ")
    print("x = \n", x)
    print("y = ", y)
    print("-----------------------")
    model.fit(x, y)
    model.find_support_vectors(x, y)
    model.draw_decision(x, y)


def number_of_mark(y, mark):
    indices = [i for i in range(len(y)) if y[i] == mark]
    return len(indices)


def create_array_real_data():
    fname = 'german.data-numeric'
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    x = []
    y = []
    for item in content:
        curr_str = np.array([int(value) for value in item if value != ''])
        y.append(curr_str[-1])
        x.append(np.delete(curr_str, -1))

    x = np.array(x)
    y = np.array(y)
    y[y == 2] = -1
    print("Length of dataset: ", len(x))
    print("Number of attributes: ", len(x[0]))
    print("Number of good clients: ", number_of_mark(y, 1))
    print("Number of bad clients: ", number_of_mark(y, -1))
    return x, y


def cross_validation(x_train, y_train, C, gamma):
    model = SVM(C=C, kernel='rbf', gamma=gamma, tol=1e-2)
    cross = lambda arr, sz: [arr[i:i + sz] for i in range(0, len(arr), sz)]
    x_cross_val = np.array(cross(x_train, 160))
    y_cross_val = np.array(cross(y_train, 160))
    indices = np.array(range(5))
    score = 0
    for i in range(5):
        curr_indices = np.delete(indices, i)
        x_curr_valid = x_cross_val[i]
        y_curr_valid = y_cross_val[i]
        x_curr_train = np.vstack(x_cross_val[curr_indices])
        y_curr_train = y_cross_val[curr_indices].ravel()
        model.fit(x_curr_train, y_curr_train)
        model.number_support_vectors()
        y_curr_valid_predict = model.predict(x_curr_valid, x_curr_train, y_curr_train)
        curr_score = model.score_error(y_curr_valid_predict, y_curr_valid)
        print("i = ", i, ". Score error = ", curr_score, ", i = ", i,)
        score += curr_score
    print("Average score: " ,score / 5)
    return score / 5


def real_data_train():
    x, y = create_array_real_data()
    shuffle_index = np.random.permutation(len(y))
    x = x[shuffle_index]
    y = y[shuffle_index]
    # 1000 elements: 800 for training, 200 for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # C = [1, 10]
    # gamma = [0.01, 0.1, 0.5, 1.0]
    # average_error = np.zeros((len(C), len(gamma)))
    # for i in range(len(C)):
    #     for j in range(len(gamma)):
    #         print("Cross-validation for parameters C = ", C[i], ", gamma = ", gamma[j])
    #         average_error[i][j] = cross_validation(x_train, y_train, C=C[i], gamma=gamma[j])
    # find C = 1, gamma = 0.01
    print("Create model C = ", 1000, ", gamma = ", 1)
    model = SVM(C=1, kernel='rbf', gamma=0.01, tol=1e-2)
    print("Fit model with train sequence")
    model.fit(x_train, y_train)
    model.number_support_vectors()
    print("Predict model on test sequence")
    y_test_predict = model.predict(x_test, x_train, y_train)
    score = model.score_error(y_test_predict, y_test)
    print("Score error = ", score)


if __name__ == "__main__":
    print("Hello world!")

    # linear_train()
    # non_linear_train()
    real_data_train()

    print("Bye, world!")
