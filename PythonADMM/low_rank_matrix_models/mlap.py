import numpy as np
from scipy.linalg import svd


def prox_nuclear(B, lambd):
    U, S, Vt = svd(B, full_matrices=False)
    S = np.diag(S)
    svp = np.sum(S > lambd)
    if svp >= 1:
        S = S[:svp] - lambd
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

def prox_l1(b, lambd):
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def prox_l21(B, lambd):
    X = np.zeros_like(B)
    for i in range(X.shape[1]):
        nxi = np.linalg.norm(B[:, i])
        if nxi > lambd:
            X[:, i] = (1 - lambd / nxi) * B[:, i]
    return X

def prox_tensor_l21(B, lambd):
    X = np.zeros_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            v = B[i, j, :]
            nxi = np.linalg.norm(v)
            if nxi > lambd:
                X[i, j, :] = (1 - lambd / nxi) * B[i, j, :]
    return X

def comp_loss(E, loss):
    if loss == 'l1':
        return np.sum(np.abs(E))
    elif loss == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2

def mlap(X, lambd, alpha, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l21')

    d, n, K = X.shape
    Z = np.zeros((n, n, K))
    E = np.zeros((d, n, K))
    J = np.copy(Z)
    S = np.copy(Z)
    Y = np.copy(E)
    W = np.copy(Z)
    V = np.copy(Z)
    dY = np.copy(Y)

    XtX = np.zeros((n, n, K))
    invXtXI = np.zeros((n, n, K))
    I = np.eye(n)

    for i in range(K):
        XtX[:, :, i] = X[:, :, i].T @ X[:, :, i]
        invXtXI[:, :, i] = np.linalg.inv(XtX[:, :, i] + I)

    nuclearnormJ = np.zeros(K)

    for iter in range(max_iter):
        Zk = np.copy(Z)
        Ek = np.copy(E)
        Jk = np.copy(J)
        Sk = np.copy(S)

        # Super block {J, S}
        for i in range(K):
            J[:, :, i], nuclearnormJ[i] = prox_nuclear(Z[:, :, i] + W[:, :, i] / mu, 1 / mu)
            S[:, :, i] = invXtXI[:, :, i] @ (XtX[:, :, i] - X[:, :, i].T @ (E[:, :, i] - Y[:, :, i] / mu) + Z[:, :, i] + (V[:, :, i] - W[:, :, i]) / mu)

        # Super block {Z, E}
        Z = prox_tensor_l21((J + S - (W + V) / mu) / 2, alpha / (2 * mu))
        XmXS = np.zeros_like(E)

        for i in range(K):
            XmXS[:, :, i] = X[:, :, i] - X[:, :, i] @ S[:, :, i]

        if loss == 'l1':
            for i in range(K):
                E[:, :, i] = prox_l1(XmXS[:, :, i] + Y[:, :, i] / mu, lambd / mu)
        elif loss == 'l21':
            for i in range(K):
                E[:, :, i] = prox_l21(XmXS[:, :, i] + Y[:, :, i] / mu, lambd / mu)
        elif loss == 'l2':
            for i in range(K):
                E[:, :, i] = (XmXS[:, :, i] + Y[:, :, i] / mu) / (lambd / mu + 1)

        dY = XmXS - E
        dW = Z - J
        dV = Z - S

        chg = max([np.max(np.abs(Zk - Z)), np.max(np.abs(Ek - E)), np.max(np.abs(Jk - J)), np.max(np.abs(Sk - S)),
                   np.max(np.abs(dY)), np.max(np.abs(dW)), np.max(np.abs(dV))])

        if DEBUG and (iter == 0 or iter % 10 == 0):
            obj = np.sum(nuclearnormJ) + lambd * comp_loss(E, loss) + alpha * comp_loss(Z, 'l21')
            err = np.sqrt(np.linalg.norm(dY)**2 + np.linalg.norm(dW)**2 + np.linalg.norm(dV)**2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        W += mu * dW
        V += mu * dV
        mu = min(rho * mu, max_mu)

    obj = np.sum(nuclearnormJ) + lambd * comp_loss(E, loss) + alpha * comp_loss(Z, 'l21')
    err = np.sqrt(np.linalg.norm(dY)**2 + np.linalg.norm(dW)**2 + np.linalg.norm(dV)**2)
    
    return Z, E, obj, err, iter

# Example usage
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
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1'
}

n1, n2, K = 100, 200, 10
X = np.random.rand(n1, n2, K)
lambd = 0.1
alpha = 0.2
Z, E, obj, err, iterations = mlap(X, lambd, alpha, opts)

print(f'Iterations: {iterations}')
print(f'Objective: {obj}')
print(f'Error: {err}')

