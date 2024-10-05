import numpy as np
import matplotlib.pyplot as plt

# Function to construct the l1_admm problem
def construct_l1_admm(A, B, opts):
    m, n = A.shape
    X = np.zeros((n, B.shape[1]))
    Z = np.zeros_like(X)
    U = np.zeros_like(X)
    return X, Z, U


# Function for the ADMM learning process
def admm_learning(A, B, X, Z, U, opts):
    rho = opts['rho']
    epsilon = 1e-6
    losses = []

    for k in range(opts['max_iter']):

        # constant learning rate
        alpha = 1

        # X-update (least squares problem)
        X = np.linalg.solve(A.T @ A + rho * np.eye(X.shape[0]), A.T @ B + rho * (Z - U))

        # Z-update (soft thresholding)
        Z_old = Z
        Z = np.maximum(0, X + U - 1 / rho) - np.maximum(0, -X - U - 1 / rho)

        # U-update (dual variable update)
        U = U + alpha * (X - Z)

        # Compute loss
        loss = np.linalg.norm(A @ Z - B, 'fro')
        losses.append(loss)

        # Check convergence based on change in loss
        if k > 0 and abs(losses[-1] - losses[-2]) < epsilon and loss < 5:
            break

    return Z, loss, k, losses


# Function to evaluate the result
def evaluate_result(Z, A, B):
    loss = np.linalg.norm(A @ Z - B, 'fro')
    return loss


# Function to draw the learning curve
def plot_learning_curve(losses):
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.show()

def evaluate():
    # Generate toy data
    d = 10
    na = 200
    nb = 100

    A = np.random.randn(d, na)
    X_true = np.random.randn(na, nb)
    B = np.dot(A, X_true)
    b = B[:, 0]

    opts = {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.1,
        'mu': 1e-4,
        'max_mu': 1e10,
        'DEBUG': 0
    }

    # Construct the l1_admm problem
    X, Z, U = construct_l1_admm(A, B, opts)

    # ADMM learning process
    Z, loss, iter_, losses = admm_learning(A, B, X, Z, U, opts)

    # Evaluate the result
    final_loss = evaluate_result(Z, A, B)
    print(iter_, final_loss)

    return -iter_

evaluate()
