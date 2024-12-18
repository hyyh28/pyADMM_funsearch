import numpy as np
import matplotlib.pyplot as plt

def prox_elasticnet(b, lambda1, lambda2):
    """
    Proximal operator for the elastic net.
    """
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu):
    # 更新对偶变量
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    # 动态调整惩罚参数
    mu = min(mu * beta_factor, max_mu)
    return Y1, Y2, mu

def elasticnet_la_admm(A, B, lambda_, opts):
    """
    Solve the elastic net minimization problem using LA-ADMM.
    """
    # Set options
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 1.5)  # 每阶段惩罚参数增长倍数
    stage_iter = opts.get('stage_iter', 50)     # 每阶段最大迭代次数
    max_mu = opts.get('max_mu', 1e10)
    initial_mu = opts.get('mu', 1e-4)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    mu = initial_mu
    total_iter = 0

    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                Xk = X.copy()
                Zk = Z.copy()

                # 更新 X
                X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_ / mu)

                # 更新 Z
                Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)

                # 计算残差和变化
                dY1 = A @ Z - B
                dY2 = X - Z
                chgX = np.max(np.abs(Xk - X))
                chgZ = np.max(np.abs(Zk - Z))
                chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = np.linalg.norm(X.ravel(), 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
                    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                    print(f"Stage {stage}, Iter {total_iter}, mu={mu:.2e}, obj={obj}, err={err}")

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 每阶段更新惩罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    obj = np.linalg.norm(X.ravel(), 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, obj, err, total_iter

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X

# Options for the elastic net minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'beta_factor': 2,  # 动态增长因子
    'stage_iter': 50,    # 每阶段最大迭代次数
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
}

# Regularization parameter for elastic net
lambda_ = 0.01

# Perform elastic net minimization using LA-ADMM
X2, obj, err, iter = elasticnet_la_admm(A, B, lambda_, opts)
print(f"Iterations: {iter}, Objective: {obj}, Error: {err}")

# # Plot the first column of X2
# plt.stem(X2[:, 0])
# plt.title('Elastic Net Regularization Result (LA-ADMM)')
# plt.show()