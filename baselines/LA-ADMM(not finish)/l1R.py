import numpy as np
import matplotlib.pyplot as plt

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2

    # 每阶段更新惩罚参数
    if iter_count >= stage_count:
        mu = min(rho * mu, max_mu)
        iter_count = 0  # 重置阶段计数

    return Y1, Y2, mu, rho, iter_count + 1

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

def l1R_adaptive(A, B, lambda_val, opts):
    # 设置默认参数
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')
    stage_count = opts.get('stage_count', 100)  # 每个阶段的迭代次数

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

    iter_count = 0  # 当前阶段计数

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # 更新 X 和 E
        X = prox_l1(Z - Y2 / mu, lambda_val / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # 更新 Z
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # 计算残差和变化
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(E, loss) + lambda_val * np.linalg.norm(X, 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        # 自适应更新惩罚参数
        Y1, Y2, mu, rho, iter_count = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count)

    obj = comp_loss(E, loss) + lambda_val * np.linalg.norm(X, 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 1,
    'loss': 'l1',
    'stage_count': 100
}

# 生成toy数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# 使用自适应的 l1 正则化
lambda_val = 0.01
X, E, obj, err, iter = l1R_adaptive(A, B, lambda_val, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('Adaptive Regularized L1 Norm Result')
# plt.show()