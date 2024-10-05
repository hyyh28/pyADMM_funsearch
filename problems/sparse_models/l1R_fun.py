import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambda_val):
    """
    The proximal operator of the l1 norm

    min_x lambda*||x||_1 + 0.5*||x - b||_2^2

    Parameters:
        b : numpy.ndarray
            Input vector or matrix
        lambda_val : float
            Regularization parameter

    Returns:
        x : numpy.ndarray
            Output vector or matrix after applying the proximal operator
    """
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    """
    Compute the loss for a given matrix E and loss type

    Parameters:
        E : numpy.ndarray
            Error matrix
        loss : str
            Type of loss ('l1', 'l21', 'l2')

    Returns:
        out : float
            Computed loss value
    """
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l21':
        return np.sum([np.linalg.norm(E[:, i]) for i in range(E.shape[1])])
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def l1R(A, B, lambd, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')
    tau_incr = opts.get('tau_incr', 2)
    tau_decr = opts.get('tau_decr', 2)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # First super block {X, E}
        X = prox_l1(Z - Y2 / mu, lambd / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(E, loss) + lambd * np.linalg.norm(X, 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}")

        if chg < tol:
            break

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2

        # Update penalty parameter rho
        r_norm = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
        s_norm = mu * np.sqrt(np.linalg.norm(np.dot(A.T, dY1), 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if r_norm > tau_incr * s_norm:
            rho *= tau_incr
            Y1 /= tau_incr
            Y2 /= tau_incr
        elif s_norm > tau_decr * r_norm:
            rho /= tau_decr
            Y1 *= tau_decr
            Y2 *= tau_decr

        mu = min(rho * mu, max_mu)

    obj = comp_loss(E, loss) + lambd * np.linalg.norm(X, 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    
    return X, E, obj, err, iter


# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0,
    'loss': 'l1'
}

# 生成玩具数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# regularized l1
lambda_val = 0.01
X, E, obj, err, iter = l1R(A, B, lambda_val, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
plt.stem(X[:, 0])
plt.title('Regularized L1 Norm Result')
plt.show()
