import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='mu_value.log', level=logging.INFO)

def prox_gl1(b, G, lambda_val):
    x = np.zeros_like(b)
    for g in G:
        nxg = np.linalg.norm(b[g])
        if nxg > lambda_val:
            x[g] = b[g] * (1 - lambda_val / nxg)
    return x

def prox_l1(b, lambda_val):
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l21':
        return np.sum([np.linalg.norm(E[:, i]) for i in range(E.shape[1])])
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def compute_groupl1(X, G):
    obj = 0
    for i in range(X.shape[1]):
        x = X[:, i]
        for g in G:
            obj += np.linalg.norm(x[g])
    return obj

def groupl1R(A, B, G, lambd, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

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
        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
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
            obj = comp_loss(E, loss) + lambd * compute_groupl1(X, G)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}")

        if chg < tol:
            break

        # Update penalty parameter rho dynamically
        rho_update_factor = 1.01  # Example factor for updating rho
        rho = min(rho * rho_update_factor, max_mu)

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(E, loss) + lambd * compute_groupl1(X, G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    
    return X, E, obj, err, iter


def evaluate(opts, lambd, d=10, na=200, nb=100, g_num=5):
    # Generate random problem instance
    A = np.random.randn(d, na)
    X_true = np.random.randn(na, nb)
    B = A @ X_true

    # Create group indices
    g_len = round(na / g_num)
    G = [list(range((i - 1) * g_len, i * g_len)) for i in range(1, g_num)]
    G.append(list(range((g_num - 1) * g_len, na)))

    # Solve using groupl1R
    _, _, obj, err, iter = groupl1R(A, B, G, lambd, opts)
    return iter

def evaluate_multiple(opts, lambd, num_trials=10):
    total_iterations = 0
    for _ in range(num_trials):
        total_iterations += evaluate(opts, lambd)
    
    average_iterations = total_iterations / num_trials
    print(f'Average Iterations over {num_trials} trials: {average_iterations:.2f}')
    return average_iterations

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'min_mu': 1e-10,
    'rho': 1.1,
    'DEBUG': 1,
    'loss': 'l1'
}

# Run the evaluation over multiple trials
lambda_val = 1
evaluate_multiple(opts, lambda_val)