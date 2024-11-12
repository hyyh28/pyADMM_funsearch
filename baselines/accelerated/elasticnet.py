import numpy as np
import matplotlib.pyplot as plt

def prox_elasticnet(b, lambda1, lambda2):
    # The proximal operator of the elastic net
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_):
    # Computes the augmented Lagrangian for elastic net
    term1 = 0.5 * np.linalg.norm(A @ Z - B, 'fro')**2
    term2 = np.linalg.norm(X, 1) + lambda_ * np.linalg.norm(X, 'fro')**2
    term3 = np.sum(Y1 * (A @ Z - B)) + np.sum(Y2 * (X - Z))
    term4 = (mu / 2) * (np.linalg.norm(A @ Z - B, 'fro')**2 + np.linalg.norm(X - Z, 'fro')**2)
    return term1 + term2 + term3 + term4

def elasticnet_adaptive(A, B, lambda_, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    
    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_)

    for iter in range(1, max_iter + 1):
        Xk, Zk = X.copy(), Z.copy()

        # update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_ / mu)
        # update Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)
        
        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        # Calculate objective and error for debug
        obj = np.linalg.norm(X, 1) + lambda_ * np.linalg.norm(X, 'fro')**2
        err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        # Adaptive mu update based on Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2

    obj = np.linalg.norm(X, 1) + lambda_ * np.linalg.norm(X, 'fro')**2
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
    
    return X, obj, err, iter

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

# Options for the elastic net minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
}

# Regularization parameter for elastic net
lambda_ = 0.01

# Perform elastic net minimization
X2, obj, err, iter = elasticnet_adaptive(A, B, lambda_, opts)
print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
