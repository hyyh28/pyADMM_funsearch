import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import svd

def prox_tnn(Y, rho):
    n1, n2, n3 = Y.shape
    X = np.zeros((n1, n2, n3), dtype=complex)
    Y = fft(Y, axis=2)
    tnn = 0
    trank = 0

    # First frontal slice
    U, S, Vt = svd(Y[:, :, 0], full_matrices=False)
    r = np.sum(S > rho)
    if r >= 1:
        S = S[:r] - rho
        X[:, :, 0] = np.dot(U[:, :r], np.dot(np.diag(S), Vt[:r, :]))
        tnn += np.sum(S)
        trank = max(trank, r)

    # For i = 2 to halfn3
    halfn3 = round(n3 / 2)
    for i in range(1, halfn3):
        U, S, Vt = svd(Y[:, :, i], full_matrices=False)
        r = np.sum(S > rho)
        if r >= 1:
            S = S[:r] - rho
            X[:, :, i] = np.dot(U[:, :r], np.dot(np.diag(S), Vt[:r, :]))
            tnn += 2 * np.sum(S)
            trank = max(trank, r)
        X[:, :, n3 - i] = np.conj(X[:, :, i])

    # If n3 is even
    if n3 % 2 == 0:
        i = halfn3
        U, S, Vt = svd(Y[:, :, i], full_matrices=False)
        r = np.sum(S > rho)
        if r >= 1:
            S = S[:r] - rho
            X[:, :, i] = np.dot(U[:, :r], np.dot(np.diag(S), Vt[:r, :]))
            tnn += np.sum(S)
            trank = max(trank, r)

    tnn = tnn / n3
    X = ifft(X, axis=2).real
    return X, tnn, trank

def tprod(A, B):
    n1, r, n3 = A.shape
    r, n2, n3 = B.shape
    C = np.zeros((n1, n2, n3))
    for i in range(n3):
        C[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
    return C

def lrtr_Gaussian_tnn(A, b, Xsize, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-6)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    n1, n2, n3 = Xsize['n1'], Xsize['n2'], Xsize['n3']
    X = np.zeros((n1, n2, n3))
    Z = np.zeros((n1, n2, n3))
    m = len(b)
    Y1 = np.zeros(m)
    Y2 = np.zeros((n1, n2, n3))

    I = np.eye(n1 * n2 * n3)
    invA = np.linalg.inv(A.T @ A + I) @ I

    for iter_ in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()

        # Update X
        X, Xtnn, _ = prox_tnn(Z - Y2 / mu, 1 / mu)

        # Update Z
        vecZ = invA @ (A.T @ (-Y1 / mu + b) + Y2.ravel() / mu + X.ravel())
        Z = vecZ.reshape(n1, n2, n3)

        dY1 = A @ vecZ - b
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = Xtnn
            err = np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2.ravel()) ** 2
            print(f'iter {iter_+1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = Xtnn
    err = np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2.ravel()) ** 2
    return X, obj, err, iter_ + 1


opts = {
    'mu': 1e-6,
    'rho': 1.1,
    'max_iter': 500,
    'DEBUG': 1
}

n1 = 30
n2 = n1
n3 = 5
r = int(0.2 * n1)  # tubal rank

L1 = np.random.randn(n1, r, n3) / n1
L2 = np.random.randn(r, n2, n3) / n2
X = tprod(L1, L2)  # low rank part

m = 3 * r * (n1 + n2 - r) * n3 + 1  # number of measurements
n = n1 * n2 * n3
A = np.random.randn(m, n) / np.sqrt(m)

b = A @ X.ravel()
Xsize = {'n1': n1, 'n2': n2, 'n3': n3}

Xhat, obj, err, iter_ = lrtr_Gaussian_tnn(A, b, Xsize, opts)

RSE = np.linalg.norm(Xhat.ravel() - X.ravel()) / np.linalg.norm(X.ravel())
trank = np.sum(np.linalg.svd(Xhat[:, :, 0], full_matrices=False)[1] > 1e-8)

print("Relative Squared Error (RSE):", RSE)
print("Tubal rank:", trank)
