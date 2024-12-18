import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu):
    # 更新对偶变量
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    # 动态调整罚参数
    mu = min(mu * beta_factor, max_mu)
    return Y1, Y2, mu

def tracelassoR_LA_ADMM(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 2)  # 每阶段罚参数增长倍数
    stage_iter = opts.get('stage_iter', 50)  # 每阶段最大迭代次数
    max_mu = opts.get('max_mu', 1e10)
    initial_mu = opts.get('mu', 1e-4)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))

    mu = initial_mu
    total_iter = 0

    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                xk, ek, Zk = x.copy(), e.copy(), Z.copy()

                # 第一块变量更新 {Z, e}
                Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / mu, lambda_ / mu)
                if loss == 'l1':
                    e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
                elif loss == 'l2':
                    e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
                else:
                    raise ValueError('Loss function not supported')

                # 第二块变量更新 {x}
                x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + diagAtB(A, Y2 / mu + Z))

                # 计算残差和变化
                dY1 = A @ x + e - b
                dY2 = Z - A @ np.diag(x)
                chgx = np.max(np.abs(xk - x))
                chge = np.max(np.abs(ek - e))
                chgZ = np.max(np.abs(Zk - Z))
                chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = comp_loss(e, loss) + lambda_ * nuclearnorm
                    err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
                    print(f'Stage {stage}, Iter {total_iter}, mu={mu:.2e}, obj={obj}, err={err}')

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 每阶段结束后更新罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    obj = comp_loss(e, loss) + lambda_ * nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)

    return x, e, obj, err, total_iter

def prox_nuclear(B, lambda_):
    U, S, Vt = svd(B, full_matrices=False)
    svp = np.sum(S > lambda_)
    if svp >= 1:
        S = S[:svp] - lambda_
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def diagAtB(A, B):
    return np.array([np.dot(A[:, i], B[:, i]) for i in range(A.shape[1])])

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

# Options for the regularized trace Lasso minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'beta_factor': 1.5,  # 动态增长因子
    'stage_iter': 50,    # 每阶段迭代次数
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1,
    'loss': 'l1'
}

# Regularization parameter
lambda_ = 0.1

# Perform regularized trace Lasso minimization using LA-ADMM
x, e, obj, err, iter = tracelassoR_LA_ADMM(A, b, lambda_, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# Plot the solution vector x
# plt.stem(x)
# plt.title('Regularized Trace Lasso Result (LA-ADMM)')
# plt.show()