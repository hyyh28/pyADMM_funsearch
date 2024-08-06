import numpy as np
import matplotlib.pyplot as plt

def prox_elasticnet(b, lambda1, lambda2):
    """
    The proximal operator of the elastic net

    min_x lambda1*||x||_1 + 0.5*lambda2*||x||_2^2 + 0.5*||x-b||_2^2

    Parameters:
        b : numpy.ndarray
            Input vector or matrix
        lambda1 : float
            L1 regularization parameter
        lambda2 : float
            L2 regularization parameter

    Returns:
        x : numpy.ndarray
            Output vector or matrix after applying the proximal operator
    """
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def elasticnet_admm(A, B, lambda_val, opts):
    """
    Solve the elastic net minimization problem by ADMM

    min_X ||X||_1 + lambda*||X||_F^2, s.t. AX=B

    Parameters:
        A : numpy.ndarray
            d*na matrix
        B : numpy.ndarray
            d*nb matrix
        lambda_val : float
            Regularization parameter
        opts : dict
            Dictionary containing optimization options:
            tol       : termination tolerance
            max_iter  : maximum number of iterations
            mu        : stepsize for dual variable updating in ADMM
            max_mu    : maximum stepsize
            rho       : rho>=1, ratio used to increase mu
            DEBUG     : 0 or 1 for printing debug info

    Returns:
        X : numpy.ndarray
            na*nb matrix
        obj : float
            Objective function value
        err : float
            Residual ||AX-B||_F
        iter : int
            Number of iterations
    """
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = X.copy()
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # Update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_val / mu)

        # Update Z
        Z = invAtAI @ (AtB + X + (A.T @ (Y1 / mu) - Y2 / mu))

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X, 1) + lambda_val * np.linalg.norm(X, 'fro') ** 2
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X, 1) + lambda_val * np.linalg.norm(X, 'fro') ** 2
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, obj, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0
}

# 生成玩具数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# Elastic Net 正则化
lambda_val = 0.01
X, obj, err, iter = elasticnet_admm(A, B, lambda_val, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
plt.stem(X[:, 0])
plt.title('Elastic Net Regularization Result')
plt.show()
