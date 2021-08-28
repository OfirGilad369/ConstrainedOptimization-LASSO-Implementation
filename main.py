import math
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def ex2d():
    x = np.array([[0.0],
                  [0.0]])
    result_x, objective_history, constraint1_history, constraint2_history, constraint3_history = Steepest_Descent(x)
    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    fig3, axs3 = plt.subplots()
    fig4, axs4 = plt.subplots()
    axs1.plot(objective_history, label="objective")
    axs2.plot(constraint1_history, label="constraint1")
    axs3.plot(constraint2_history, label="constraint2")
    axs4.plot(constraint3_history, label="constraint3")
    axs1.set_ylabel('Values')
    axs2.set_ylabel('Values')
    axs3.set_ylabel('Values')
    axs4.set_ylabel('Values')
    axs1.set_xlabel('Iterations')
    axs2.set_xlabel('Iterations')
    axs3.set_xlabel('Iterations')
    axs4.set_xlabel('Iterations')
    axs1.set_title('Function values')
    axs2.set_title('Constraint1 violation')
    axs3.set_title('Constraint2 violation')
    axs4.set_title('Constraint3 violation')
    axs1.legend()
    axs2.legend()
    axs3.legend()
    axs4.legend()
    plt.show()
    print("x = ", result_x)


def objective(x, mu):
    x1 = x[0][0]
    x2 = x[1][0]
    function = math.pow(x1+x2, 2) - 10*(x1+x2)
    penalty1 = mu * math.pow(3*x1 + x2 - 6, 2)
    penalty2 = mu * math.pow(max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5), 2)
    penalty3 = mu * math.pow(max(0.0, -x1), 2)
    objective_x = function + penalty1 + penalty2 + penalty3
    return objective_x


def gradient(x, mu):
    x1 = x[0][0]
    x2 = x[1][0]
    sub_derivative_x1 = 2*x1 + 2*x2 - 10
    penalty1_x1 = mu * (18*x1 + 6*x2 - 36)
    penalty2_x1 = mu * max(0.0, 4*math.pow(x1, 3) + 4*x1*math.pow(x2, 2) - 20*x1)
    penalty3_x1 = mu * max(0.0, -2*x1)
    derivative_x1 = sub_derivative_x1 + penalty1_x1 + penalty2_x1 + penalty3_x1

    sub_derivative_x2 = 2 * x1 + 2 * x2 - 10
    penalty1_x2 = mu * (2*x2 + 6*x1 - 12)
    penalty2_x2 = mu * max(0.0, 4*math.pow(x2, 3) + 4*x2*math.pow(x1, 2) - 20*x2)
    penalty3_x2 = 0
    derivative_x2 = sub_derivative_x2 + penalty1_x2 + penalty2_x2 + penalty3_x2

    gradient_x = np.array([[derivative_x1],
                           [derivative_x2]])
    return gradient_x


def Steepest_Descent(x, alpha=1.0, iterations=50):
    mu_vector = [0.01, 0.1, 1.0, 10.0, 100.0]
    objective_history = []
    constraint1_history = []
    constraint2_history = []
    constraint3_history = []
    current_x = x
    for mu in mu_vector:
        print("mu = " + str(mu))
        for i in range(iterations):
            gradient_x = gradient(current_x, mu)
            if np.linalg.norm(gradient_x) < 1e-3:
                break
            d = -gradient_x
            alpha = Armijo_Linesearch(mu, current_x, d, gradient_x, alpha=alpha)
            current_x += alpha * d
            objective_x = objective(current_x, mu)
            objective_history.append(objective_x)
            x1 = x[0][0]
            x2 = x[1][0]
            constraint1_history.append(np.linalg.norm(3 * x1 + x2 - 6))
            constraint2_history.append(np.linalg.norm((max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5))))
            constraint3_history.append(np.linalg.norm((max(0.0, -x1))))

    return current_x, objective_history, constraint1_history, constraint2_history, constraint3_history


def Armijo_Linesearch(mu, x, d, gradient_x, alpha=1.0, beta=0.5, c=1e-5):
    objective_x = objective(x, mu)
    for i in range(10):
        objective_x_1 = objective(x + (alpha * d), mu)
        if objective_x_1 <= objective_x + (alpha * c * np.dot(d.transpose(), gradient_x)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def ex3cd():
    n = 5
    H = init_h(n)
    g = np.array([18, 6, -12, -6, 18])
    a = np.array([0] * n)
    b = np.array([5] * n)
    x0 = np.array([1] * n, dtype=np.float_)
    result_x = Coordinate_Descent(H, g, x0, a, b)
    print("x = ", result_x)


def init_h(n):
    H = np.ones(n)
    H *= -1
    return H + np.diag([6] * n)


def Coordinate_Descent(H, g, x0, a, b, iterations=100):
    """
    :param H: matrix
    :param g: vector
    :param x0: initial guess
    :param a: vector, lower bound constraint
    :param b: vector, upper bound constraint
    :return: final x
    """
    H_diag = np.diag(np.diag(H))
    H_rest = H - H_diag
    x = x0
    curr_norm = np.linalg.norm(x)
    for _ in range(iterations):
        for i in range(len(x)):
            x[i] = calc_xi(H_rest, H[i][i], g, x, a, b, i)
        last_norm = curr_norm
        curr_norm = np.linalg.norm(x)
        if abs(last_norm - curr_norm) < 0.001:
            break
    return x


def calc_xi(H_rest, Hii, g, x, a, b, i):
    """
    :param Hii: the coordinate (i,i) of the H matrix
    :param H: matrix
    :param g: vector
    :param x: current answer
    :param a: vector, lower bound constraint
    :param b: vector, upper bound constraint
    :return: final x
    """
    xi = - (H_rest[i] @ x - g[i])
    xi /= Hii
    # return xi
    return apply_constraints(xi, a, b, i)


def apply_constraints(xi, a, b, i):
    xi = min(xi, b[i])
    xi = max(xi, a[i])
    return xi


def ex4bc():
    # Generating Data
    A, b = generate_A_b()
    lambda_w, C, w = generate_lambda_C_w_()
    w_result, objective_history = Steepest_Descent_wrt(A, C, w, b, lambda_w)
    x = C @ w_result

    # Creating lambda graph
    fig1, axs1 = plt.subplots()
    axs1.plot(objective_history, label="objective value")
    axs1.set_yscale('log')
    axs1.set_ylabel('Values')
    axs1.set_xlabel('Iterations')
    axs1.set_title('Function values for lambda=50')
    axs1.legend()
    plt.show()
    print("x = ", x)

    # Creating non-zeros entries graph
    lambda_x_list = range(0, 105, 5)
    x_non_zeros_counter = []
    for lambda_w_value in lambda_x_list:
        w_result, objective_history = Steepest_Descent_wrt(A, C, w, b, lambda_w_value)
        x = C @ w_result
        x_non_zeros_counter.append(non_zeros_entries_percentage(x))
    fig2, axs2 = plt.subplots()
    axs2.plot(lambda_x_list, x_non_zeros_counter, label="non-zeros entries percentage")
    axs2.set_yscale('log')
    axs2.set_ylabel('Percentages')
    axs2.set_xlabel('Lambda values')
    axs2.set_title('Non-zeros entries percentage per lambda')
    axs2.legend()
    plt.show()


def generate_A_b():
    A = np.random.normal(0, 1, (100, 200))
    x = np.zeros(200)
    x[:20] = np.random.rand(20)
    np.random.shuffle(x)
    x = sparse.coo_matrix([x]).transpose()
    noise = np.random.normal(0, 0.1, (100, 1))
    b = A @ x + noise
    return A, b


def generate_lambda_C_w_():
    lambda_w = 50
    C = np.concatenate((np.eye(200), -np.eye(200))).transpose()
    w = np.random.rand(400, 1)
    return lambda_w, C, w


def non_zeros_entries_percentage(x):
    percentage = 0.0
    for i in range(x.shape[0]):
        if x[i][0] > 0:
            percentage += 1
    return percentage/2.0


def objective_wrt(A, C, w, b, lambda_w):
    ACw_b = A @ C @ w - b
    ones_vector = np.ones(w.shape)
    objective_w_ans = ACw_b.transpose() @ ACw_b + lambda_w * (ones_vector.transpose() @ w)
    return objective_w_ans


def gradient_wrt(A, C, w, b, lambda_w):
    ACw_b = A @ C @ w - b
    gradient_w_ans = 2 * C.transpose() @ A.transpose() @ ACw_b + lambda_w
    return gradient_w_ans


def Steepest_Descent_wrt(A, C, w, b, lambda_w, alpha=1.0, iterations=100):
    objective_history = []
    current_w = w
    for i in range(iterations):
        gradient_w = gradient_wrt(A, C, current_w, b, lambda_w)
        if np.linalg.norm(gradient_w) < 1e-3:
            break
        d = -gradient_w
        alpha = Armijo_Linesearch_wrt(A, C, current_w, b, lambda_w, d, gradient_w, alpha=alpha)
        current_w += alpha * d
        current_w = np.clip(current_w, 0, None)
        current_objective_w = objective_wrt(A, C, current_w, b, lambda_w)
        objective_history.append(np.linalg.norm(current_objective_w))
    return current_w, objective_history


def Armijo_Linesearch_wrt(A, C, w, b, lambda_w, d, gradient_w, alpha=1.0, beta=0.5, c=1e-5):
    objective_w = objective_wrt(A, C, w, b, lambda_w)
    for i in range(10):
        w_alpha_d = w + (alpha * d)
        w_alpha_d = np.clip(w_alpha_d, 0, None)
        objective_w_1 = objective_wrt(A, C, w_alpha_d, b, lambda_w)
        if objective_w_1 <= objective_w + (alpha * c * np.dot(d.transpose(), gradient_w)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


if __name__ == '__main__':
    ex2d()
    #ex3cd()
    #ex4bc()

