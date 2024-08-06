import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambda_val):
    """
    The proximal operator of the l1 norm

    min_x lambda*||x||_1 + 0.5*||x - b||_2^2

    Parameters:
        b : numpy.ndarray
            Input vector or matrix
        lambda_val : float
            Regularization parameter

    Returns:
        x : numpy.ndarray
            Output vector or matrix after applying the proximal operator
    """
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def l1_admm(A, B, opts):
    """
    Solve the l1-minimization problem by ADMM

    min_X ||X||_1, s.t. AX=B

    Parameters:
        A : numpy.ndarray
            d*na matrix
        B : numpy.ndarray
            d*nb matrix
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
            objective function value
        err : float
            residual ||AX-B||_F
        iter : int
            number of iterations
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
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        
        # update X
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        
        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)
        
        # update residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X, 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        
        if chg < tol:
            break
        
        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)
    
    obj = np.linalg.norm(X, 1)
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

# 生成toy数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true
b = B[:, 0]

# L1 正则化
X, obj, err, iter = l1_admm(A, B, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
plt.stem(X[:, 0])
plt.title('L1 Regularization Result')
plt.show()
