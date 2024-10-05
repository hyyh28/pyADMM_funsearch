import numpy as np
import matplotlib.pyplot as plt

def ksupportR(A, B, lambda_, k, opts):
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

    for iter in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()

        # First super block {X,E}
        temp = Z - Y2 / mu
        temp = prox_ksupport(temp.ravel(), k, lambda_ / mu)
        X = temp.reshape(na, nb)

        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        dY1 = A @ Z + E - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 0 or (iter + 1) % 10 == 0):
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter + 1}, mu={mu}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, err, iter + 1

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def prox_ksupport(v, k, lambda_):
    L = 1 / lambda_
    d = len(v)
    if k >= d:
        return L * v / (1 + L)
    elif k <= 1:
        k = 1

    z = np.abs(v)
    sorted_indices = np.argsort(-z)
    z = z[sorted_indices]
    ar = np.cumsum(z)
    z = z * L

    p = np.zeros_like(z)

    for r in range(k - 1, -1, -1):
        T = ar[k - r - 1] - r * z[k - r - 1]
        if (k - r) * z[k - r - 1] >= T:
            p[:k - r] = z[:k - r] / (L + 1)
            p[k - r:] = np.maximum(z[k - r:], 0)
            break

    p = p[sorted_indices.argsort()]
    return v - p

# Example usage
np.random.seed(0)
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0
}

lambda_ = 0.1
k = 10

# Perform k-support norm minimization
X, E, err, iter = ksupportR(A, B, lambda_, k, opts)

# Plot the solution vector X[:,0]
plt.stem(X[:, 0])
plt.show()
