import numpy as np
import matplotlib.pyplot as plt

def prox_elasticnet(b, lambda1, lambda2):
    """
    The proximal operator of the elastic net

    min_x lambda1 * ||x||_1 + 0.5 * lambda2 * ||x||_2^2 + 0.5 * ||x - b||_2^2
    """
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def prox_l1(b, lambda_val):
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss):
    term1 = comp_loss(E, loss)
    term2 = lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
    term3 = np.sum(Y1 * (A @ Z + E - B)) + np.sum(Y2 * (X - Z))
    term4 = (mu / 2) * (np.linalg.norm(A @ Z + E - B, 'fro') ** 2 + np.linalg.norm(X - Z, 'fro') ** 2)
    return term1 + term2 + term3 + term4

def elasticnetR_adaptive(A, B, lambda1, lambda2, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss)

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # Update X and E
        X = prox_elasticnet(Z - Y2 / mu, lambda1 / mu, lambda2 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Update Z
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        # Adaptive mu update based on augmented Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2

    obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 1,
    'loss': 'l1'
}

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# Regularized elastic net
lambda1 = 10
lambda2 = 10
X, E, obj, err, iter = elasticnetR_adaptive(A, B, lambda1, lambda2, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")

# # Plot the first column of X
# plt.stem(X[:, 0])
# plt.title('Regularized Elastic Net Result')
# plt.show()