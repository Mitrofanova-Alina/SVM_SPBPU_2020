import numpy as np
from matplotlib import pyplot as plt


class SVM:
    def __init__(self, C=1, kernel='rbf', gamma='auto', tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.alpha = None
        self.b = None
        self.__kernel_func = None

    def __get_gamma(self, x):
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'auto':
            return 1.0 / x.shape[1]
        elif self.gamma == 'scale':
            x_var = x.var()
            return 1.0 / (x.shape[1] * x_var) if x_var > 1e-7 else 1.0
        else:
            raise ValueError(f"'{self.gamma}' is incorrect value for gamma")

    def __get_kernel_function(self, x):
        if self.kernel == 'linear':
            return self.linear_kernel
        elif self.kernel == 'rbf':
            self.gamma = self.__get_gamma(x)
            return self.rbf_kernel
        else:
            raise ValueError(f"'{self.kernel}' is incorrect value for kernel")

    def value_decision_function(self, point, x, y):
        summa = np.sum(self.alpha[i] * y[i] * self.__kernel_func(x[i], point) for i in range(len(y)))
        return summa + self.b

    def linear_draw(self, x, y):
        fig, ax = plt.subplots(1, 1)
        fig.set_dpi(200)

        h = 0.05
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        x_p = np.arange(x_min, x_max, h)

        w = np.sum(self.alpha[i] * y[i] * x[i] for i in range(len(y)))
        k = -  w[0] / w[1]
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='summer')
        ax.text(x_max - 2, y_max - 0.5, "Class 2, y = -1")
        ax.text(x_min + 0.1, y_min + 0.5, "Class 1, y = 1")
        for i in range(len(y)):
            if self.alpha[i] > 0:
                if y[i] == 1:
                    ax.plot(x_p, k * x_p + (1 - self.b) / w[1], 'k--')
                    ax.scatter(x[i, 0], x[i, 1], c="crimson")
                    ax.text(x_min + 0.1, k * (x_min + 0.1) + (1 - self.b) / w[1] - 1, '(w, x) + b = 1', rotation=-22)
                else:
                    ax.plot(x_p, k * x_p + (- 1 - self.b) / w[1], 'k--')
                    ax.scatter(x[i, 0], x[i, 1], c="orange")
                    ax.text(x_min + 0.1, k * (x_min + 0.1) + (- 1 - self.b) / w[1] - 1, '(w, x) + b = -1', rotation=-22)

        ax.plot(x_p, k * x_p - self.b / w[1], 'k-')
        ax.text(x_min + 0.1, k * (x_min + 0.1) - self.b / w[1] - 1, '(w, x) + b = 0', rotation=-22)

        title = "Training set with decision regions for " + str(self.kernel) + " kernel, \n C = " + str(
            self.C) + ", tol = " + str(self.tol)
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        plt.show()

    def non_linear_draw(self, x, y):
        fig, ax = plt.subplots(1, 1)
        fig.set_dpi(200)

        h = 0.05
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        print("Calculate points")
        points = [[x_item, y_item] for y_item in np.arange(y_min, y_max, h) for x_item in np.arange(x_min, x_max, h)]
        points = np.array(points)
        print("Calculate y")
        y_points = [self.value_decision_function(item, x, y) for item in points]
        points = np.array(points)
        y_points = np.array(y_points)
        y_points[y_points > 0.0] = 1
        y_points[y_points < 0.0] = -1

        ax.scatter(points[:, 0], points[:, 1], c=y_points, cmap='Pastel2')
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='summer')
        for i in range(len(y)):
            if self.alpha[i] > 0:
                if y[i] == -1:
                    ax.scatter(x[i, 0], x[i, 1], c="crimson")
                else:
                    ax.scatter(x[i, 0], x[i, 1], c="orange")

        title = "Training set with decision regions for " + str(self.kernel) + " kernel, \n C = " + str(
            self.C) + ", gamma = " + str(self.gamma) + ", tol = " + str(self.tol)
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        plt.show()

    def find_support_vectors(self, x, y):
        w = np.sum(self.alpha[i] * y[i] * x[i] for i in range(len(y)))
        for i in range(len(y)):
            f_x = w.dot(x[i]) + self.b
            if self.alpha[i] > 0:
                print("Support vector ", x[i], ", value = ", f_x)
            else:
                print("Vector outside the band ", x[i], ", value = ", f_x)

    def __SOR(self, M, e):
        length = len(e)
        omega = 1
        alpha_prev = np.zeros(length)
        alpha_curr = np.zeros(length)
        num_iter = 0
        while True:
            num_iter += 1
            for i in range(length):
                sum1 = np.sum(M[i][j] * alpha_curr[j] for j in range(0, i))
                sum2 = np.sum(M[i][j] * alpha_prev[j] for j in range(i + 1, length))
                alpha_curr[i] = omega / M[i][i] * (e[i] - sum1 - sum2) + (1 - omega) * alpha_prev[i]
                if alpha_curr[i] <= 0:
                    alpha_curr[i] = 0
                elif 0 < alpha_curr[i] < self.C:
                    continue
                else:
                    alpha_curr[i] = self.C
            norma = self.norm(alpha_prev, alpha_curr)
            if norma <= self.tol:
                break
            alpha_prev = np.array(alpha_curr)
        print("-----------------------")
        print("Iterations: ", num_iter)
        print("Norm: ", norma)
        return alpha_curr

    def __get_Q(self, x, y):
        length = len(y)
        Q = [[y[i] * y[j] * self.__kernel_func(x[i], x[j]) for j in range(length)] for i in range(length)]
        return np.array(Q)

    def __multy(self, y):
        length = len(y)
        matrix = [[y[i] * y[j] for j in range(length)] for i in range(length)]
        return np.array(matrix)

    def __get_M(self, Q, y):
        return np.array(Q + self.__multy(y))

    def fit(self, x, y):
        self.__kernel_func = self.__get_kernel_function(x)
        Q = self.__get_Q(x, y)
        e = np.ones(len(y))
        M = self.__get_M(Q, y)
        # print("Q = \n", Q)
        # print("e = ", e)
        # print("M = \n", M)
        self.alpha = self.__SOR(M, e)
        self.b = np.sum(self.alpha * y)
        print("-----------------------")
        print("Answer SOR:")
        print("Alpha = ", self.alpha)
        print("b = ", self.b)
        print("-----------------------")

    def predict(self, points, x, y):
        y_predict = np.zeros(len(points))
        i = 0
        for item in points:
            print("Predict ", item)
            value = self.value_decision_function(item, x, y)
            y_predict[i] = np.sign(value)
        y_predict[y_predict == 0] = 1
        return y_predict

    def linear_kernel(self, x1, x2):
        return x1.dot(x2)

    def rbf_kernel(self, x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        s1, _ = x1.shape
        s2, _ = x2.shape
        norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
        norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
        return np.exp(- self.gamma * (norm1 + norm2 - 2 * x1.dot(x2.T)))[0][0]

    def norm(self, x1, x2):
        return np.linalg.norm(np.atleast_2d(x2 - x1))
