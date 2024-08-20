import numpy as np
from numpy.linalg import svd

def prox_nuclear(B, lambda_):
    U, S, Vt = svd(B, full_matrices=False)
    svp = np.sum(S > lambda_)
    if svp >= 1:
        S = S[:svp] - lambda_
        X = np.dot(U[:, :svp], np.dot(np.diag(S), Vt[:svp, :]))
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def unfold(X, shape, mode):
    return np.moveaxis(X, mode, 0).reshape((shape[mode], -1), order='F')

def fold(X, shape, mode):
    full_shape = list(shape)
    full_shape[mode] = -1
    return np.moveaxis(X.reshape(full_shape, order='F'), 0, mode)

def nmodeproduct(T, U, mode):
    return fold(np.dot(U, unfold(T, T.shape, mode)), T.shape, mode)

def lrtcR_snn(M, omega, alpha, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    dim = M.shape
    k = len(dim)
    omegac = np.setdiff1d(np.arange(np.prod(dim)), omega)

    X = np.zeros(dim)
    Y = [X.copy() for _ in range(k)]
    Z = [X.copy() for _ in range(k)]
    E = np.zeros(dim)
    Y2 = np.zeros(dim)

    for iter_ in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = [Z[i].copy() for i in range(k)]

        # Update Z and E
        sumtemp = np.zeros(dim)
        for i in range(k):
            Z[i] = fold(prox_nuclear(unfold(X + Y[i] / mu, dim, i), alpha[i] / mu)[0], dim, i)
            sumtemp += Z[i] - Y[i] / mu
        
        if loss == 'l1':
            E = prox_l1(-X + M - Y2 / mu, 1 / mu)
        elif loss == 'l2':
            E = (-X + M - Y2 / mu) * (mu / (1 + mu))
        else:
            raise ValueError('Unsupported loss function')

        # Update X
        X.ravel()[omega] = (sumtemp.ravel()[omega] - Y2.ravel()[omega] / mu - E.ravel()[omega] + M.ravel()[omega]) / (k + 1)
        X.ravel()[omegac] = sumtemp.ravel()[omegac] / k

        chg = max(np.max(np.abs(Xk - X)), np.max(np.abs(Ek - E)))
        err = 0
        for i in range(k):
            dY = X - Z[i]
            err += np.linalg.norm(dY)**2
            Y[i] += mu * dY
            chg = max(chg, np.max(np.abs(dY)), np.max(np.abs(Zk[i] - Z[i])))

        dY = E - M
        dY.ravel()[omega] += X.ravel()[omega]
        chg = max(chg, np.max(np.abs(dY)))
        Y2 += mu * dY
        err = np.sqrt(err + np.linalg.norm(dY)**2)

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            print(f'iter {iter_+1}, mu={mu}, err={chg}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return X, err, iter_ + 1


opts = {
    'mu': 1e-6,
    'rho': 1.1,
    'max_iter': 500,
    'DEBUG': 1,
    'loss': 'l1'  # or 'l2' for different loss functions
}

n1 = 50
n2 = n1
n3 = n1
r = 5

X = np.random.rand(r, r, r)
U1 = np.random.rand(n1, r)
U2 = np.random.rand(n2, r)
U3 = np.random.rand(n3, r)

X = nmodeproduct(X, U1, 0)
X = nmodeproduct(X, U2, 1)
X = nmodeproduct(X, U3, 2)

p = 0.5
omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]

M = np.zeros((n1, n2, n3))
M.ravel()[omega] = X.ravel()[omega]

lambda_ = [1, 1, 1]
Xhat, err, iter_ = lrtcR_snn(M, omega, lambda_, opts)

print("Error:", err)
print("Iterations:", iter_)
