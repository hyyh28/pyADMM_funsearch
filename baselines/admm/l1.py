import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    
    # 自适应惩罚参数更新，根据当前阶段数动态调整
    if iter_count >= stage_count:
        mu = min(rho * mu, max_mu)  # 增大惩罚参数
        iter_count = 0  # 重置迭代计数器
        
    
    return Y1, Y2, mu, rho, iter_count + 1

def l1_adaptive(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    stage_count = opts.get('stage_count', 100)  # 每个阶段的最大迭代次数

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    iter_count = 0  # 当前阶段的迭代计数器

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        
        # update X
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        
        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        # 自适应更新惩罚参数
        Y1, Y2, mu, rho, iter_count = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count)

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

    # 使用自适应的 l1 解法
    X2, obj, err, iter = l1_adaptive(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter

datasets = {
    'l1': {
        'opts': {
            'tol': 1e-6,
            'max_iter': 1000,
            'rho': 1.1,
            'mu': 1e-4,
            'max_mu': 1e10,
            'DEBUG': 1,
            'stage_count': 100
        },
        'd': 10,
        'na': 200,
        'nb': 100
    }
}

evaluate(datasets['l1'])