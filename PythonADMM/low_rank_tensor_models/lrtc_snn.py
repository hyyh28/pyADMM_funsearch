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

def unfold(X, shape, mode):
    return np.moveaxis(X, mode, 0).reshape((shape[mode], -1), order='F')

def fold(X, shape, mode):
    full_shape = list(shape)
    full_shape[mode] = -1
    return np.moveaxis(X.reshape(full_shape, order='F'), 0, mode)

def nmodeproduct(T, U, mode):
    return fold(np.dot(U, unfold(T, T.shape, mode)), T.shape, mode)

def lrtc_snn(M, omega, alpha, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = M.shape
    k = len(dim)
    omegac = np.setdiff1d(np.arange(np.prod(dim)), omega)

    X = np.zeros(dim)
    X.ravel()[omega] = M.ravel()[omega]
    Y = [X.copy() for _ in range(k)]
    Z = [X.copy() for _ in range(k)]

    for iter_ in range(max_iter):
        Xk = X.copy()
        Zk = [Z[i].copy() for i in range(k)]

        # Update Z
        sumtemp = np.zeros(len(omegac))
        for i in range(k):
            Z[i] = fold(prox_nuclear(unfold(X + Y[i] / mu, dim, i), alpha[i] / mu)[0], dim, i)
            sumtemp += Z[i].ravel()[omegac] - Y[i].ravel()[omegac] / mu

        # Update X
        X.ravel()[omegac] = sumtemp / k

        chg = np.max(np.abs(Xk - X))
        err = 0
        for i in range(k):
            dY = X - Z[i]
            err += np.linalg.norm(dY)**2
            Y[i] += mu * dY
            chg = max(chg, np.max(np.abs(dY)), np.max(np.abs(Zk[i] - Z[i])))

        err = np.sqrt(err)

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            print(f'iter {iter_+1}, mu={mu}, err={err}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return X, err, iter_ + 1



opts = {
    'mu': 1e-6,
    'rho': 1.1,
    'max_iter': 500,
    'DEBUG': 1
}

n1 = 50
n2 = n1
n3 = n1
r = 5

X = np.random.randn(r, r, r)
U1 = np.random.randn(n1, r)
U2 = np.random.randn(n2, r)
U3 = np.random.randn(n3, r)

X = nmodeproduct(X, U1, 0)
X = nmodeproduct(X, U2, 1)
X = nmodeproduct(X, U3, 2)

p = 0.5
omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]

M = np.zeros((n1, n2, n3))
M.ravel()[omega] = X.ravel()[omega]

lambda_ = [1, 1, 1]
Xhat, err, iter_ = lrtc_snn(M, omega, lambda_, opts)

print("Error:", err)
print("Iterations:", iter_)
RSE = np.linalg.norm(X.ravel() - Xhat.ravel()) / np.linalg.norm(X.ravel())
print("Relative Squared Error (RSE):", RSE)
