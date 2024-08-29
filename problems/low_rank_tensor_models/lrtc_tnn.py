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
    A_hat = fft(A, axis=2)
    B_hat = fft(B, axis=2)
    C_hat = np.zeros((A.shape[0], B.shape[1], A.shape[2]), dtype=complex)
    for i in range(A.shape[2]):
        C_hat[:, :, i] = np.dot(A_hat[:, :, i], B_hat[:, :, i])
    return ifft(C_hat, axis=2).real

def lrtc_tnn(M, omega, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = M.shape
    X = np.zeros(dim)
    X.ravel()[omega] = M.ravel()[omega]
    E = np.zeros(dim)
    Y = np.zeros(dim)

    for iter_ in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()

        # Update X
        X, tnnX, _ = prox_tnn(-E + M + Y / mu, 1 / mu)

        # Update E
        E = M - X + Y / mu
        E.ravel()[omega] = 0

        dY = M - X - E
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = max(chgX, chgE, np.max(np.abs(dY)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = tnnX
            err = np.linalg.norm(dY)
            print(f'iter {iter_+1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        mu = min(rho * mu, max_mu)

    obj = tnnX
    err = np.linalg.norm(dY)
    return X, obj, err, iter_ + 1

# 主程序部分
opts = {
    'mu': 1e-6,
    'rho': 1.1,
    'max_iter': 500,
    'DEBUG': 1
}

n1 = 50
n2 = n1
n3 = n1
r = int(0.1 * n1)  # tubal rank

L1 = np.random.randn(n1, r, n3) / n1
L2 = np.random.randn(r, n2, n3) / n2
X = tprod(L1, L2)  # low rank part

p = 0.5
omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]

M = np.zeros((n1, n2, n3))
M.ravel()[omega] = X.ravel()[omega]

Xhat, obj, err, iter_ = lrtc_tnn(M, omega, opts)

print("Error:", err)
print("Iterations:", iter_)
RSE = np.linalg.norm(X.ravel() - Xhat.ravel()) / np.linalg.norm(X.ravel())
trank = np.sum(np.linalg.svd(Xhat[:, :, 0], full_matrices=False)[1] > 1e-8)

print("Relative Squared Error (RSE):", RSE)
print("Tubal rank:", trank)
