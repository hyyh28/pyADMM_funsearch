import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu):
    return np.linalg.norm(X.ravel(), 1) + np.sum(Y1 * (A @ Z - B)) + np.sum(Y2 * (X - Z)) + (mu / 2) * (np.linalg.norm(A @ Z - B, 'fro')**2 + np.linalg.norm(X - Z, 'fro')**2)

def l1_adaptive_anderson(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    history_X = []
    history_Z = []

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        Y1 += mu * dY1
        Y2 += mu * dY2

        if len(history_X) == anderson_m:
            history_X.pop(0)
            history_Z.pop(0)
        
        history_X.append(X.copy())
        history_Z.append(Z.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            G = np.vstack([delta_X, delta_Z]).T
            g = np.hstack([(X - history_X[0]).flatten(), (Z - history_Z[0]).flatten()])
            if G.shape[0] == g.shape[0]:  # Ensure compatibility before lstsq
                alpha = np.linalg.lstsq(G, g, rcond=None)[0]
                X -= np.dot(alpha, delta_X[-1])
                Z -= np.dot(alpha, delta_Z[-1])

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        if chg < tol:
            break

        # Adaptive mu update
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter

def evaluate(instances: dict) -> float:
    d = instances['d']
    na = instances['na']
    nb = instances['nb']
    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X
    opts = instances['opts']
    
    X2, obj, err, iter = l1_adaptive_anderson(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter

datasets = {
    'l1': {'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.1,
        'mu': 1e-4,
        'max_mu': 1e10,
        'DEBUG': 1,
        'anderson_m': 5
    },
    'd': 10,
    'na': 200,
    'nb': 100}
}

evaluate(datasets['l1'])