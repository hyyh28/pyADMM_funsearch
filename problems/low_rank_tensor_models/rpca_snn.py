import numpy as np
from numpy.linalg import svd

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

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

def trpca_snn(X, alpha, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = X.shape
    k = len(dim)

    E = np.zeros(dim)
    Y = [np.zeros(dim) for _ in range(k)]
    L = [np.zeros(dim) for _ in range(k)]

    for iter_ in range(max_iter):
        Lk = L.copy()
        Ek = E.copy()

        # Update L
        sumtemp = np.zeros(dim)
        for i in range(k):
            L[i] = fold(prox_nuclear(unfold(X - E - Y[i] / mu, dim, i), alpha[i] / mu)[0], dim, i)
            sumtemp += L[i] + Y[i] / mu

        # Update E
        E = prox_l1(X - sumtemp / k, 1 / (mu * k))

        chg = np.max(np.abs(Ek - E))
        err = 0
        for i in range(k):
            dY = L[i] + E - X
            err += np.linalg.norm(dY)**2
            Y[i] = Y[i] + mu * dY
            chg = max(chg, np.max(np.abs(dY)), np.max(np.abs(Lk[i] - L[i])))

        err = np.sqrt(err)

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            print(f'iter {iter_+1}, mu={mu}, err={err}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return L[0], E, err, iter_ + 1


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

L = np.random.randn(r, r, r)
U1 = np.random.randn(n1, r)
U2 = np.random.randn(n2, r)
U3 = np.random.randn(n3, r)

L = nmodeproduct(L, U1, 0)
L = nmodeproduct(L, U2, 1)
L = nmodeproduct(L, U3, 2)

p = 0.05
m = int(p * n1 * n2 * n3)
temp = np.random.rand(n1 * n2 * n3)
I = np.argsort(temp)[:m]

Omega = np.zeros((n1, n2, n3))
Omega.ravel()[I] = 1

E = np.sign(np.random.rand(n1, n2, n3) - 0.5)
S = Omega * E

Xn = L + S

lambda_ = np.sqrt([max(n1, n2 * n3), max(n2, n1 * n3), max(n3, n1 * n2)])
lambda_ = [1, 1, 1]

Lhat, Shat, err, iter_ = trpca_snn(Xn, lambda_, opts)

print("Error:", err)
print("Iterations:", iter_)
