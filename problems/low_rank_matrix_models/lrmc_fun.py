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

# Low-Rank Matrix Completion (LRMC) function
def lrmc(MM, omega, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n = MM.shape
    M = np.zeros((d, n))
    M[omega] = MM[omega]
    X = np.zeros((d, n))
    E = np.zeros((d, n))
    Y = np.zeros((d, n))
    rho = 1.1

    def update_rho(rho, dY, tol):
        norm_dY = np.linalg.norm(dY, 'fro')
        if norm_dY > tol:
            rho *= 1.1
        elif norm_dY < tol:
            rho *= 0.9
        return rho

    for iter in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()

        # Update X
        X, nuclearnormX = prox_nuclear(-(E - M + Y / mu), 1 / mu)

        # Update E
        E = -(X - M + Y / mu)
        E[omega] = 0

        dY = X + E - M
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = np.max([chgX, chgE, np.max(np.abs(dY))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = nuclearnormX
            err = np.linalg.norm(dY, 'fro')
            print(f'iter {iter + 1}, mu={mu}, rho={rho}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        mu = min(rho * mu, max_mu)
        rho = update_rho(rho, dY, tol)

    obj = nuclearnormX
    err = np.linalg.norm(dY, 'fro')
    return X, obj, err, iter



# Regularized Low-Rank Matrix Completion (LRMC) function
def lrmcR(M, omega, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = M.shape
    X = np.zeros((d, n))
    Z = np.zeros((d, n))
    E = np.zeros((d, n))
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((d, n))
    omegac = np.setdiff1d(np.arange(d * n), np.ravel_multi_index(omega, (d, n)))

    for iter in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        Ek = E.copy()

        # First super block {X, E}
        X, nuclearnormX = prox_nuclear(Z - Y2 / mu, 1 / mu)
        temp = M - Y1 / mu
        temp[omega] -= Z[omega]

        if loss == 'l1':
            E = prox_l1(temp, lambda_ / mu)
        elif loss == 'l21':
            E = prox_l21(temp, lambda_ / mu)
        elif loss == 'l2':
            E = temp * (mu / (lambda_ + mu))
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z[omega] = (-E[omega] + M[omega] - (Y1[omega] - Y2[omega]) / mu + X[omega]) / 2
        Z.flat[omegac] = X.flat[omegac] + Y2.flat[omegac] / mu

        dY1 = E - M
        dY1[omega] += Z[omega]
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormX + lambda_ * comp_loss(E, loss)
                err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormX + lambda_ * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)

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
    'DEBUG': 0
}

# Low-Rank Matrix Completion (LRMC)
n1, n2 = 100, 200
r = 5
X = np.random.randn(n1, r) @ np.random.randn(r, n2)

p = 0.6
omega = np.where(np.random.rand(n1, n2) < p)
M = np.zeros((n1, n2))
M[omega] = X[omega]

Xhat, obj, err, iter = lrmc(M, omega, opts)
rel_err_X = np.linalg.norm(Xhat - X, 'fro') / np.linalg.norm(X, 'fro')

#print(f'rel_err_X: {rel_err_X}')

print(f'Objective: {obj}')
print(f'Error: {err}')
print(f'Iterations: {iter}')

# # Regularized LRMC
# E = np.random.randn(n1, n2) / 100
# M = X + E
# lambda_ = 0.1

# Xhat, Ehat, obj, err, iter = lrmcR(M, omega, lambda_, opts)
