import numpy as np
import matplotlib.pyplot as plt

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = min(mu * beta_factor, max_mu)  # 按阶段性调整
    return Y1, Y2, mu

def prox_nuclear(B, lambda_val):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S_threshold = np.maximum(S - lambda_val, 0)
    X = np.dot(U, np.dot(np.diag(S_threshold), Vt))
    nuclearnorm = np.sum(S_threshold)
    return X, nuclearnorm

def diagAtB(A, B):
    return np.einsum('ij,ij->j', A, B)

def trace_lasso_la_admm(A, b, opts):
    """
    LA-ADMM for Trace Lasso Regularization
    """
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 2)  # 每阶段罚参数增长因子
    stage_iter = opts.get('stage_iter', 50)  # 每阶段的迭代步数
    max_mu = opts.get('max_mu', 1e10)
    initial_mu = opts.get('mu', 1e-4)
    DEBUG = opts.get('DEBUG', 0)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))

    mu = initial_mu
    total_iter = 0

    # 主循环
    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                xk, Zk = x.copy(), Z.copy()

                # 更新 x
                x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))

                # 更新 Z
                Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

                # 计算残差和变化
                dY1 = A @ x - b
                dY2 = A @ np.diag(x) - Z
                chgx = np.max(np.abs(xk - x))
                chgZ = np.max(np.abs(Zk - Z))
                chg = max(chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = nuclearnorm
                    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
                    print(f"Stage {stage}, Iter {total_iter}, mu={mu:.2e}, obj={obj}, err={err}")

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 阶段性更新罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    obj = nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return x, obj, err, total_iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'beta_factor': 1.5,  # 每阶段的罚参数倍增因子
    'stage_iter': 50,  # 每阶段的迭代步数
    'DEBUG': 1
}

# 生成 toy 数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true
b = B[:, 0]

# Trace Lasso 正则化
x, obj, err, iter = trace_lasso_la_admm(A, b, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# plt.stem(x)
# plt.title('Trace Lasso Regularization Result (LA-ADMM)')
# plt.show()