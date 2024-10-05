import numpy as np

# Proximal operator for nuclear norm
def prox_nuclear(B, lambda_):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S = np.maximum(S - lambda_, 0)
    svp = np.sum(S > 0)
    if svp >= 1:
        S = S[:svp]
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0

    return X, nuclearnorm

# Proximal operator for L1 norm
def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

# Proximal operator for L21 norm
def prox_l21(B, lambda_):
    X = np.zeros_like(B)
    for i in range(B.shape[1]):
        nxi = np.linalg.norm(B[:, i])
        if nxi > lambda_:
            X[:, i] = (1 - lambda_ / nxi) * B[:, i]
    return X

# Function to compute the loss based on the selected type
def comp_loss(E, loss):
    if loss == 'l1':
        return np.sum(np.abs(E))
    elif loss == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError("Unsupported loss function")

# Latent Low-Rank Representation (LatLRR) function
def latlrr(X, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    eta1 = 1.02 * 2 * np.linalg.norm(X, 2)**2  # for Z
    eta2 = eta1  # for L
    eta3 = 1.02 * 2  # for E

    d, n = X.shape
    E = np.zeros((d, n))
    Z = np.zeros((n, n))
    L = np.zeros((d, d))
    Y = np.zeros_like(E)

    XtX = X.T @ X
    XXt = X @ X.T

    for iter in range(max_iter):
        Lk = L.copy()
        Ek = E.copy()
        Zk = Z.copy()

        # First super block {Z}
        Z, nuclearnormZ = prox_nuclear(Zk - (X.T @ (Y / mu + L @ X - X - E) + XtX @ Z) / eta1, 1 / (mu * eta1))

        # Second super block {L, E}
        temp = Lk - ((Y / mu + X @ Z - E) @ X.T + Lk @ XXt - XXt) / eta2
        L, nuclearnormL = prox_nuclear(temp, 1 / (mu * eta2))

        if loss == 'l1':
            E = prox_l1(Ek + (Y / mu + X @ Z + Lk @ X - X - Ek) / eta3, lambda_ / (mu * eta3))
        elif loss == 'l21':
            E = prox_l21(Ek + (Y / mu + X @ Z + Lk @ X - X - Ek) / eta3, lambda_ / (mu * eta3))
        elif loss == 'l2':
            E = (Y + mu * (X @ Z + Lk @ X - X + (eta3 - 1) * Ek)) / (lambda_ + mu * eta3)
        else:
            raise ValueError('Unsupported loss function')

        dY = X @ Z + L @ X - X - E
        chgL = np.max(np.abs(Lk - L))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgL, chgE, chgZ, np.max(np.abs(dY))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormZ + nuclearnormL + lambda_ * comp_loss(E, loss)
                err = np.linalg.norm(dY, 'fro')**2
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        mu = min(rho * mu, max_mu)

    obj = nuclearnormZ + nuclearnormL + lambda_ * comp_loss(E, loss)
    err = np.linalg.norm(dY, 'fro')**2

    return Z, L, obj, err, iter

# 示例代码，生成测试数据
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
    'DEBUG': 0
}

# Latent Low-Rank Representation (LatLRR)
lambda_ = 0.1
opts['loss'] = 'l1'

Z, L, obj, err, iter = latlrr(A, lambda_, opts)

print(f'Objective: {obj}')
print(f'Error: {err}')
print(f'Iterations: {iter}')
