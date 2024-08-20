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

# Proximal operator for tensor l21-norm
def prox_tensor_l21(B, lambda_):
    n1, n2, n3 = B.shape
    X = np.zeros_like(B)
    for i in range(n1):
        for j in range(n2):
            v = B[i, j, :]
            nxi = np.linalg.norm(v)
            if nxi > lambda_:
                X[i, j, :] = (1 - lambda_ / nxi) * v
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

# Multi-task Low-rank Affinity Pursuit (MLAP) function
def mlap(X, lambda_, alpha, opts):
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
    J = np.zeros_like(Z)
    S = np.zeros_like(Z)
    Y = np.zeros_like(E)
    W = np.zeros_like(Z)
    V = np.zeros_like(Z)
    XtX = np.zeros((n, n, K))
    invXtXI = np.zeros((n, n, K))
    I = np.eye(n)

    for i in range(K):
        XtX[:, :, i] = X[:, :, i].T @ X[:, :, i]
        invXtXI[:, :, i] = np.linalg.inv(XtX[:, :, i] + I)

    nuclearnormJ = np.zeros(K)

    for iter in range(max_iter):
        Zk = Z.copy()
        Ek = E.copy()
        Jk = J.copy()
        Sk = S.copy()

        # First super block {J, S}
        for i in range(K):
            J[:, :, i], nuclearnormJ[i] = prox_nuclear(Z[:, :, i] + W[:, :, i] / mu, 1 / mu)
            S[:, :, i] = invXtXI[:, :, i] @ (XtX[:, :, i] - X[:, :, i].T @ (E[:, :, i] - Y[:, :, i] / mu) + Z[:, :, i] + (V[:, :, i] - W[:, :, i]) / mu)

        # Second super block {Z, E}
        Z = prox_tensor_l21((J + S - (W + V) / mu) / 2, alpha / (2 * mu))

        XmXS = np.zeros_like(E)
        for i in range(K):
            XmXS[:, :, i] = X[:, :, i] - X[:, :, i] @ S[:, :, i]

        if loss == 'l1':
            for i in range(K):
                E[:, :, i] = prox_l1(XmXS[:, :, i] + Y[:, :, i] / mu, lambda_ / mu)
        elif loss == 'l21':
            for i in range(K):
                E[:, :, i] = prox_l21(XmXS[:, :, i] + Y[:, :, i] / mu, lambda_ / mu)
        elif loss == 'l2':
            for i in range(K):
                E[:, :, i] = (XmXS[:, :, i] + Y[:, :, i] / mu) / (lambda_ / mu + 1)
        else:
            raise ValueError('Unsupported loss function')

        dY = XmXS - E
        dW = Z - J
        dV = Z - S

        chgZ = np.max(np.abs(Zk - Z))
        chgE = np.max(np.abs(Ek - E))
        chgJ = np.max(np.abs(Jk - J))
        chgS = np.max(np.abs(Sk - S))
        chg = np.max([chgZ, chgE, chgJ, chgS, np.max(np.abs(dY)), np.max(np.abs(dW)), np.max(np.abs(dV))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = np.sum(nuclearnormJ) + lambda_ * comp_loss(E, loss) + alpha * comp_loss(Z, 'l21')
                err = np.sqrt(np.linalg.norm(dY)**2 + np.linalg.norm(dW)**2 + np.linalg.norm(dV)**2)
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        W += mu * dW
        V += mu * dV
        mu = min(rho * mu, max_mu)

    obj = np.sum(nuclearnormJ) + lambda_ * comp_loss(E, loss) + alpha * comp_loss(Z, 'l21')
    err = np.sqrt(np.linalg.norm(dY)**2 + np.linalg.norm(dW)**2 + np.linalg.norm(dV)**2)

    return Z, E, obj, err, iter

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

# Multi-task Low-rank Affinity Pursuit (MLAP)
n1 = 100
n2 = 200
K = 10
X = np.random.randn(n1, n2, K)
lambda_ = 0.1
alpha = 0.2

Z, E, obj, err, iter = mlap(X, lambda_, alpha, opts)

print(f'Error: {err}')
print(f'Iterations: {iter}')
