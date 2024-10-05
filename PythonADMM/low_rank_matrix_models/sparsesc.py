import numpy as np
from scipy.linalg import eigh, svd

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def project_fantope(Q, k):
    U, D, _ = svd(Q)
    Dr = cappedsimplexprojection(D, k)
    return np.dot(U, np.dot(np.diag(Dr), U.T))

def cappedsimplexprojection(y0, k):
    n = len(y0)
    x = np.zeros(n)

    if k < 0 or k > n:
        raise ValueError('the sum constraint is infeasible!')

    if k == 0:
        return x

    if k == n:
        return np.ones(n)

    y = np.sort(y0)
    s = np.cumsum(y)
    y = np.append(y, np.inf)

    for b in range(n):
        gamma = (k + b - n - s[b]) / (b + 1)
        if (y[0] + gamma > 0) and (y[b] + gamma < 1) and (y[b + 1] + gamma >= 1):
            x[:b + 1] = y[:b + 1] + gamma
            x[b + 1:] = 1
            return x

    for a in range(n):
        for b in range(a + 1, n):
            gamma = (k + b - n + s[a] - s[b]) / (b - a)
            if (y[a] + gamma <= 0) and (y[a + 1] + gamma > 0) and (y[b] + gamma < 1) and (y[b + 1] + gamma >= 1):
                x[a + 1:b + 1] = y[a + 1:b + 1] + gamma
                x[b + 1:] = 1
                return x
    return x

def prox_nuclear(B, lambda_):
    U, S, Vt = svd(B, full_matrices=False)
    S = np.diag(S)
    svp = np.sum(S > lambda_)
    if svp >= 1:
        S = S[:svp] - lambda_
        X = np.dot(U[:, :svp], np.dot(S, Vt[:svp, :]))
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

def comp_loss(E, loss):
    if loss == 'l1':
        return np.sum(np.abs(E))
    elif loss == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError(f"Unknown loss function: {loss}")

def sparsesc(L, lambda_, k, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    n = L.shape[0]
    P = np.zeros((n, n))
    Q = np.zeros_like(P)
    Y = np.zeros_like(P)

    for iter_ in range(max_iter):
        Pk = P.copy()
        Qk = Q.copy()

        # Update P
        P = prox_l1(Q - (Y + L) / mu, lambda_ / mu)

        # Update Q
        temp = (P + Y / mu)
        temp = (temp + temp.T) / 2
        Q = project_fantope(temp, k)

        dY = P - Q
        chgP = np.max(np.abs(Pk - P))
        chgQ = np.max(np.abs(Qk - Q))
        chg = max(chgP, chgQ, np.max(np.abs(dY)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = np.trace(np.dot(P.T, L)) + lambda_ * np.sum(np.abs(Q))
            err = np.linalg.norm(dY, 'fro')
            print(f"iter {iter_+1}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = np.trace(np.dot(P.T, L)) + lambda_ * np.sum(np.abs(Q))
    err = np.linalg.norm(dY, 'fro')
    return P, obj, err, iter_ + 1

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = np.dot(A, X)
b = B[:, 0]

opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 0
}

# Sparse spectral clustering
lambda_ = 0.001
n = 100
X = np.random.randn(n, n)
W = np.abs(np.dot(X.T, X))
I = np.eye(n)
D = np.diag(np.sum(W, axis=1))
L = I - np.dot(np.dot(np.linalg.inv(np.sqrt(D)), W), np.linalg.inv(np.sqrt(D)))
k = 5

P, obj, err, iter_ = sparsesc(L, lambda_, k, opts)
print("Objective value:", obj)
print("Error:", err)
print("Iterations:", iter_)
