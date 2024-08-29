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
def comp_loss(E, normtype):
    if normtype == 'l1':
        return np.sum(np.abs(E))
    elif normtype == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    elif normtype == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError("Unsupported loss function")

# Low-Rank and Sparse Representation (LRSR) function
def lrsr(A, B, lambda1, lambda2, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l21')

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((nb, na))
    E = np.zeros((d, na))
    Z = np.zeros_like(X)
    J = np.zeros_like(X)

    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)
    Y3 = np.zeros_like(X)

    BtB = B.T @ B
    BtA = B.T @ A
    I = np.eye(nb)
    invBtBI = np.linalg.inv(BtB + 2 * I)

    for iter in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        Ek = E.copy()
        Jk = J.copy()

        # First super block {Z, J, E}
        Z, nuclearnormZ = prox_nuclear(X + Y2 / mu, 1 / mu)
        J = prox_l1(X + Y3 / mu, lambda1 / mu)

        if loss == 'l1':
            E = prox_l1(A - B @ X + Y1 / mu, lambda2 / mu)
        elif loss == 'l21':
            E = prox_l21(A - B @ X + Y1 / mu, lambda2 / mu)
        elif loss == 'l2':
            E = mu * (A - B @ X + Y1 / mu) / (lambda2 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {X}
        X = invBtBI @ (B.T @ (Y1 / mu - E) + BtA - (Y2 + Y3) / mu + Z + J)

        dY1 = A - B @ X - E
        dY2 = X - Z
        dY3 = X - J

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chgJ = np.max(np.abs(Jk - J))
        chg = np.max([chgX, chgE, chgZ, chgJ, np.max(np.abs(dY1)), np.max(np.abs(dY2)), np.max(np.abs(dY3))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormZ + lambda1 * np.sum(np.abs(J)) + lambda2 * comp_loss(E, loss)
                err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2 + np.linalg.norm(dY3, 'fro')**2)
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        Y3 += mu * dY3
        mu = min(rho * mu, max_mu)

    obj = nuclearnormZ + lambda1 * np.sum(np.abs(J)) + lambda2 * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2 + np.linalg.norm(dY3, 'fro')**2)

    return X, E, obj, err, iter

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
    'DEBUG': 0,
    'loss': 'l21'
}

# Low-Rank and Sparse Representation (LRSR)
lambda1 = 0.1
lambda2 = 4

Xhat, Ehat, obj, err, iter = lrsr(A, B, lambda1, lambda2, opts)

print(f'Objective: {obj}')
print(f'Error: {err}')
print(f'Iterations: {iter}')
