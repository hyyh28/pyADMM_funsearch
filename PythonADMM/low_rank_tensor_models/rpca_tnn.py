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

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def tprod(A, B):
    A_hat = fft(A, axis=2)
    B_hat = fft(B, axis=2)
    C_hat = np.zeros((A.shape[0], B.shape[1], A.shape[2]), dtype=complex)
    for i in range(A.shape[2]):
        C_hat[:, :, i] = np.dot(A_hat[:, :, i], B_hat[:, :, i])
    return ifft(C_hat, axis=2).real

def trpca_tnn(X, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = X.shape
    L = np.zeros(dim)
    S = np.zeros(dim)
    Y = np.zeros(dim)

    for iter_ in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()

        # Update L
        L, tnnL, _ = prox_tnn(-S + X - Y / mu, 1 / mu)

        # Update S
        S = prox_l1(-L + X - Y / mu, lambda_ / mu)

        dY = L + S - X
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chg = max(chgL, chgS, np.max(np.abs(dY)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = tnnL + lambda_ * np.sum(np.abs(S))
            err = np.linalg.norm(dY)
            print(f'iter {iter_+1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        mu = min(rho * mu, max_mu)

    obj = tnnL + lambda_ * np.sum(np.abs(S))
    err = np.linalg.norm(dY)
    return L, S, obj, err, iter_ + 1


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
L = tprod(L1, L2)  # low rank part

p = 0.1
m = int(p * n1 * n2 * n3)
temp = np.random.rand(n1 * n2 * n3)
I = np.argsort(temp)[:m]

Omega = np.zeros((n1, n2, n3))
Omega.ravel()[I] = 1
E = np.sign(np.random.rand(n1, n2, n3) - 0.5)
S = Omega * E  # sparse part, S = P_Omega(E)

Xn = L + S
lambda_ = 1 / np.sqrt(n3 * max(n1, n2))

Lhat, Shat, _, _, _ = trpca_tnn(Xn, lambda_, opts)

RES_L = np.linalg.norm(L.ravel() - Lhat.ravel()) / np.linalg.norm(L.ravel())
RES_S = np.linalg.norm(S.ravel() - Shat.ravel()) / np.linalg.norm(S.ravel())
trank = np.sum(np.linalg.svd(Lhat[:, :, 0], full_matrices=False)[1] > 1e-8)

print("RES_L:", RES_L)
print("RES_S:", RES_S)
print("Tubal rank:", trank)
