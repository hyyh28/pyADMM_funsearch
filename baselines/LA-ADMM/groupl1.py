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
    """
    The proximal operator of the group l1 norm.
    """
    x = np.zeros_like(b)
    for g in G:
        b_g = b[g]
        norm_b_g = np.linalg.norm(b_g)
        if norm_b_g > lambda_val:
            x[g] = b_g * (1 - lambda_val / norm_b_g)
    return x

def groupl1_la_admm(A, B, G, opts):
    """
    Solve the group l1-minimization problem by LA-ADMM
    """
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 2)  # 每阶段惩罚参数增长倍数
    stage_iter = opts.get('stage_iter', 50)   # 每阶段最大迭代次数
    max_mu = opts.get('max_mu', 1e10)
    initial_mu = opts.get('mu', 1e-4)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = X.copy()
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    mu = initial_mu
    total_iter = 0

    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                Xk = X.copy()
                Zk = Z.copy()

                # 更新 X
                X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T

                # 更新 Z
                Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

                # 计算残差和变化
                dY1 = A @ Z - B
                dY2 = X - Z
                chgX = np.max(np.abs(Xk - X))
                chgZ = np.max(np.abs(Zk - Z))
                chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
                    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                    print(f"Stage {stage}, Iter {total_iter}, mu={mu:.2e}, obj={obj}, err={err}")

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 每阶段更新惩罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, obj, err, total_iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'beta_factor': 1.5,  # 动态增长因子
    'stage_iter': 50,    # 每阶段最大迭代次数
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
}

# 生成 toy 数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# Group L1 正则化
g_num = 5
g_len = round(na / g_num)
G = [list(range(i * g_len, (i + 1) * g_len)) for i in range(g_num - 1)]
G.append(list(range((g_num - 1) * g_len, na)))

X, obj, err, iter = groupl1_la_admm(A, B, G, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('Group L1 Regularization Result (LA-ADMM)')
# plt.show()