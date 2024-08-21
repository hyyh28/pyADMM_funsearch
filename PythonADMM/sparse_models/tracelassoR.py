import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv

def tracelassoR(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))

    for iter in range(max_iter):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        # first super block {Z,e}
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / mu, lambda_ / mu)
        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        # second super block {x}
        x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + diagAtB(A, Y2 / mu + Z))

        dY1 = A @ x + e - b
        dY2 = Z - A @ np.diag(x)
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(e, loss) + lambda_ * nuclearnorm
            err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(e, loss) + lambda_ * nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
    
    return x, e, obj, err, iter

def prox_nuclear(B, lambda_):
    U, S, Vt = svd(B, full_matrices=False)
    svp = np.sum(S > lambda_)
    if svp >= 1:
        S = S[:svp] - lambda_
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def diagAtB(A, B):
    return np.array([np.dot(A[:, i], B[:, i]) for i in range(A.shape[1])])

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError('Loss function not supported')

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

# Options for the regularized trace Lasso minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1'
}

# Regularization parameter
lambda_ = 0.1

# Perform regularized trace Lasso minimization
x, e, obj, err, iter = tracelassoR(A, b, lambda_, opts)

# Plot the solution vector x
plt.stem(x)
plt.title('Regularized Trace Lasso Result')
plt.show()
