import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def l1(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

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

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        # mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter


def la_admm(A, B, opts):
    # Set default options
    stages = opts.get('stages', 5)  
    stage_iter = opts.get('stage_iter', 200)  

    mu = opts['mu']  # 初始惩罚参数
    rho = opts['rho']
    max_mu = opts['max_mu']

    total_iter = 0  # 总迭代计数
    X_final = None
    obj_final = None
    err_final = None

    for stage in range(stages):
        print(f"Stage {stage + 1}, mu={mu}")

        # 调用 l1 函数进行阶段性优化
        opts_stage = opts.copy()
        opts_stage['mu'] = mu
        opts_stage['max_iter'] = stage_iter  # 每阶段运行固定次数

        X, obj, err, iter = l1(A, B, opts_stage)  # 调用标准 ADMM 作为阶段性优化

        # 更新惩罚参数 mu
        mu = min(rho * mu, max_mu)

        # 更新总的迭代次数
        total_iter += iter

        # 保存阶段性结果
        X_final = X
        obj_final = obj
        err_final = err

    return X_final, obj_final, err_final, total_iter



def evaluate_la_admm(instances: dict) -> float:
    # Generate toy data
    d = instances['d']
    na = instances['na']
    nb = instances['nb']

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X

    # Options for the l1 minimization
    opts = instances['opts']

    # Perform la_admm minimization using LA-ADMM
    X2, obj, err, iter = la_admm(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter


datasets = {}

datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 2,
    'mu': 100,
    'max_mu': 1e10,
    'stage_iter': 200,  # Number of iterations per stage
    'stages': 5,  # Number of stages
    'DEBUG': 1
},
'd': 100,
'na': 200,
'nb': 100}

evaluate_la_admm(datasets['l1'])