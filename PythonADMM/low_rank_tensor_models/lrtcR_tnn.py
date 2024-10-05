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

def comp_loss(E, loss):
    if loss == 'l1':
        return np.sum(np.abs(E))
    elif loss == 'l2':
        return 0.5 * np.sum(E ** 2)
    else:
        raise ValueError("Unsupported loss function")

def tprod(A, B):
    A_hat = fft(A, axis=2)
    B_hat = fft(B, axis=2)
    C_hat = np.zeros((A.shape[0], B.shape[1], A.shape[2]), dtype=complex)
    for i in range(A.shape[2]):
        C_hat[:, :, i] = np.dot(A_hat[:, :, i], B_hat[:, :, i])
    return ifft(C_hat, axis=2).real

def lrtcR_tnn(M, omega, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    dim = M.shape
    X = np.zeros(dim)
    Z = np.zeros(dim)
    E = np.zeros(dim)
    Y1 = np.zeros(dim)
    Y2 = np.zeros(dim)
    omegac = np.setdiff1d(np.arange(np.prod(dim)), omega)

    for iter_ in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        Ek = E.copy()

        # First super block {X, E}
        X, tnnX, _ = prox_tnn(Z - Y2 / mu, 1 / mu)
        temp = M - Y1 / mu
        temp.ravel()[omega] -= Z.ravel()[omega]

        if loss == 'l1':
            E = prox_l1(temp, lambda_ / mu)
        elif loss == 'l2':
            E = temp * (mu / (lambda_ + mu))
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z.ravel()[omega] = (-E.ravel()[omega] + M.ravel()[omega] - (Y1.ravel()[omega] - Y2.ravel()[omega]) / mu + X.ravel()[omega]) / 2
        Z.ravel()[omegac] = X.ravel()[omegac] + Y2.ravel()[omegac] / mu

        dY1 = E - M
        dY1.ravel()[omega] += Z.ravel()[omega]
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = tnnX + lambda_ * comp_loss(E, loss)
            err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
            print(f'iter {iter_+1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = tnnX + lambda_ * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return X, E, obj, err, iter_ + 1


opts = {
    'mu': 1e-6,
    'rho': 1.1,
    'max_iter': 500,
    'DEBUG': 1,
    'loss': 'l1'  # 可选 'l1' 或 'l2'
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

lambda_ = 0.5
Xhat, Ehat, obj, err, iter_ = lrtcR_tnn(M, omega, lambda_, opts)

print("Error:", err)
print("Iterations:", iter_)
RSE = np.linalg.norm(X.ravel() - Xhat.ravel()) / np.linalg.norm(X.ravel())
trank = np.sum(np.linalg.svd(Xhat[:, :, 0], full_matrices=False)[1] > 1e-8)

print("Relative Squared Error (RSE):", RSE)
print("Tubal rank:", trank)
