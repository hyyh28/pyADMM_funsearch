import numpy as np
import matplotlib.pyplot as plt

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu):
    # 更新对偶变量
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    # 动态调整惩罚参数
    mu = min(mu * beta_factor, max_mu)
    return Y1, Y2, mu

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

def groupl1R_la_admm(A, B, G, lambda_val, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 2)  # 每阶段惩罚参数增长倍数
    stage_iter = opts.get('stage_iter', 50)   # 每阶段最大迭代次数
    max_mu = opts.get('max_mu', 1e10)
    initial_mu = opts.get('mu', 1e-4)
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

    mu = initial_mu
    total_iter = 0

    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

                # 第一块变量更新 {X, E}
                for i in range(nb):
                    X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
                if loss == 'l1':
                    E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
                elif loss == 'l2':
                    E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
                else:
                    raise ValueError('Unsupported loss function')

                # 第二块变量更新 {Z}
                Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

                # 计算残差和变化
                dY1 = A @ Z + E - B
                dY2 = X - Z
                chgX = np.max(np.abs(Xk - X))
                chgE = np.max(np.abs(Ek - E))
                chgZ = np.max(np.abs(Zk - Z))
                chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
                    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                    print(f"Stage {stage}, Iter {total_iter}, mu={mu:.2e}, obj={obj}, err={err}")

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 每阶段更新惩罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, total_iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'beta_factor': 1.5,  # 动态增长因子
    'stage_iter': 50,    # 每阶段最大迭代次数
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1,
    'loss': 'l1'
}

# 生成 toy 数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# 创建组索引
g_num = 5
g_len = round(na / g_num)
G = [list(range((i - 1) * g_len, i * g_len)) for i in range(1, g_num)]
G.append(list(range((g_num - 1) * g_len, na)))

# Regularized group Lasso
lambda_val = 1
X, E, obj, err, iter = groupl1R_la_admm(A, B, G, lambda_val, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('Regularized Group Lasso Result (LA-ADMM)')
# plt.show()