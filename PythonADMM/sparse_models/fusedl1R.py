import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

def fused_l1R(A, b, lambda1, lambda2, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    e = np.zeros(d)
    z = np.copy(x)
    Y1 = np.zeros(d)
    Y2 = np.copy(x)

    Atb = A.T @ b
    I = np.eye(n)
    invAtAI = inv(A.T @ A + I) @ I

    for iter in range(max_iter):
        xk = np.copy(x)
        ek = np.copy(e)
        zk = np.copy(z)
        # update x
        x = flsa(z - Y2 / mu, lambda1 / mu, lambda2 / mu, n)
        # update e
        if loss == 'l1':
            e = prox_l1(b - A @ z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')
        # update z
        z = invAtAI @ (-A.T @ (Y1 / mu + e) + Atb + Y2 / mu + x)
        dY1 = A @ z + e - b
        dY2 = x - z
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgz = np.max(np.abs(zk - z))
        chg = max([chgx, chge, chgz, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(e, loss) + comp_fused_l1(x, lambda1, lambda2)
            err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        if chg < tol:
            break
        
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)
    
    obj = comp_loss(e, loss) + comp_fused_l1(x, lambda1, lambda2)
    err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
    
    return x, e, obj, err, iter

def flsa(v, lambda1, lambda2, n):
    # Solve the fused Lasso signal approximator (FLSA)
    x = np.copy(v)
    for _ in range(50):  # max_step = 50
        x_old = np.copy(x)
        z = np.clip(x[:-1] - x[1:], -lambda2, lambda2)
        x[:-1] = v[:-1] - z
        x[1:] = v[1:] + z
        x = np.sign(x) * np.maximum(np.abs(x) - lambda1, 0)
        if np.linalg.norm(x - x_old, ord=np.inf) < 1e-10:  # tol2 = 1e-10
            break
    return x

def comp_fused_l1(x, lambda1, lambda2):
    f = lambda1 * np.sum(np.abs(x))
    f += lambda2 * np.sum(np.abs(np.diff(x)))
    return f

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError('Loss function not supported')

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

# Options for the regularized fused L1 minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1'
}

# Regularization parameters
lambda1 = 10
lambda2 = 10

# Perform regularized fused L1 minimization
X, E, obj, err, iter = fused_l1R(A, b, lambda1, lambda2, opts)

# Plot the solution vector X and error vector E
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.stem(X)
plt.title('Solution vector X')

plt.subplot(1, 2, 2)
plt.stem(E)
plt.title('Error vector E')

plt.show()
