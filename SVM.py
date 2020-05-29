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

    def draw_decision(self, x, y):
        fig, ax = plt.subplots(1, 1)
        fig.set_dpi(200)

        point_colors = ['mediumseagreen', 'tomato']

        h = 0.05
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        p_Y, p_X = np.meshgrid(np.arange(y_min, y_max, h), np.arange(x_min, x_max, h))
        xy = np.vstack([p_X.ravel(), p_Y.ravel()]).T
        P = np.array([self.value_decision_function(item, x, y) for item in xy])
        P = P.reshape(p_X.shape)
        ax.contour(p_X, p_Y, P, colors=['blue', 'black', 'magenta'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        for i in range(len(y)):
            curr_color = point_colors[0]
            if y[i] == 1:
                curr_color = point_colors[1]
            if self.alpha[i] > 0.0:
                if self.alpha[i] == self.C:
                    ax.scatter(x[i, 0], x[i, 1], marker='s', c=curr_color)
                else:
                    ax.scatter(x[i, 0], x[i, 1], marker='X', c=curr_color)
            else:
                ax.scatter(x[i, 0], x[i, 1], marker='o', c=curr_color)

        if self.kernel == 'linear':
            ax.text(x_max - 2, y_max - 0.5, "Class 2, y = -1")
            ax.text(x_min + 0.1, y_min + 0.5, "Class 1, y = 1")

            w = np.sum(self.alpha[i] * y[i] * x[i] for i in range(len(y)))
            k = - w[0] / w[1]

            ax.text(x_min + 0.1, k * (x_min + 0.1) + (1 - self.b) / w[1] - 1, '(w, x) + b = 1', rotation=-22)
            ax.text(x_min + 0.1, k * (x_min + 0.1) - self.b / w[1] - 1, '(w, x) + b = 0', rotation=-22)
            ax.text(x_min + 0.1, k * (x_min + 0.1) + (- 1 - self.b) / w[1] - 1, '(w, x) + b = -1',rotation=-22)
        else:
            ax.text(x_min, y_max-0.5, 'Violet line corresponds f(x) = -1')
            ax.text(x_min, y_min+0.5, 'Blue line corresponds f(x) = 1')
            ax.axis('equal')

        title = "Training set with decision regions for " + str(self.kernel) + " kernel, \n C = " \
                + str(self.C) + ", gamma = " + str(self.gamma) + ", tol = " + str(self.tol)
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
        omega = 0.1
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
        # print("-----------------------")
        # print("Answer SOR:")
        # print("Alpha = ", self.alpha)
        # print("b = ", self.b)
        # print("-----------------------")

    def predict(self, points, x, y):
        y_predict = np.array([np.sign(self.value_decision_function(item, x, y)) for item in points])
        y_predict[y_predict == 0] = 1
        return y_predict

    def number_support_vectors(self):
        link_supp_vec = 0
        supp_vec = 0
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0.0:
                if self.alpha[i] == self.C:
                    link_supp_vec += 1
                else:
                    supp_vec += 1

        print("Number of support vectors: ", supp_vec)
        print("Number of link support vectors: ", link_supp_vec)

    def score_error(self, y_predict, y_real):
        score = 0
        for i in range(len(y_predict)):
            if y_predict[i] != y_real[i]:
                score += 1
        return score / len(y_predict)

    def linear_kernel(self, x1, x2):
        return x1.dot(x2)

    def rbf_kernel(self, x1, x2):
        return np.exp(- self.gamma * (self.norm(x1, x2)) ** 2)

    def norm(self, x1, x2):
        return np.linalg.norm(np.atleast_2d(x2 - x1))
