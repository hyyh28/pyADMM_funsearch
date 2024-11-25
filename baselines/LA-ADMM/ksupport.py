import numpy as np
import matplotlib.pyplot as plt

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu):
    # 更新对偶变量
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    # 动态调整惩罚参数
    mu = min(mu * beta_factor, max_mu)
    return Y1, Y2, mu

def prox_ksupport(v, k, lambda_val):
    """
    k-support norm proximal operator.
    """
    L = 1 / lambda_val
    d = len(v)
    if k >= d:
        return L * v / (1 + L)
    elif k <= 1:
        k = 1

    z = np.sort(np.abs(v))[::-1]
    z *= L
    ar = np.cumsum(z)
    z = np.append(z, -np.inf)
    diff = 0
    err = np.inf
    found = False

    for r in range(k - 1, -1, -1):
        l, T = bsearch(z, ar, k - r, d, diff, k, r, L)
        if ((L + 1) * T >= (l - k + (L + 1) * r + L + 1) * z[k - r]) and \
           (((k - r - 1 == 0) or (L + 1) * T < (l - k + (L + 1) * r + L + 1) * z[k - r - 1])):
            found = True
            break
        diff += z[k - r]
        err_tmp = max(0, (l - k + (L + 1) * r + L + 1) * z[k - r] - (L + 1) * T) + \
                  max(0, - (l - k + (L + 1) * r + L + 1) * z[k - r - 1] + (L + 1) * T)
        if err > err_tmp:
            err_r, err_l, err_T, err = r, l, T, err_tmp

    if not found:
        r, l, T = err_r, err_l, err_T

    p = np.zeros(d)
    if k - r - 1 > 0:
        p[:k - r - 1] = z[:k - r - 1] / (L + 1)
    p[k - r - 1:l + 1] = T / (l - k + (L + 1) * r + L + 1)

    if l + 1 < d:
        p[l + 1:] = z[l + 1:d]  # Ensure the right size for p

    ind = np.argsort(np.abs(v))[::-1]
    rev = np.zeros_like(ind)
    rev[ind] = np.arange(d)

    p = np.sign(v) * p[rev]
    return v - 1 / L * p

def bsearch(z, array, low, high, diff, k, r, L):
    if z[low] == 0:
        return low, 0
    while low < high:
        mid = (low + high) // 2 + 1
        tmp = mid - k + r + 1 + L * (r + 1)
        if z[mid] * tmp - (array[mid] - diff) > 0:
            low = mid
        else:
            high = mid - 1
    return low, array[low] - diff

def ksupport_la_admm(A, B, k, opts):
    """
    Solve the k-support norm minimization problem by LA-ADMM
    """
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 1000)
    beta_factor = opts.get('beta_factor', 2)  # 每阶段惩罚参数增长倍数
    stage_iter = opts.get('stage_iter', 50)   # 每阶段迭代次数
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
                temp = Z - Y2 / mu
                temp = prox_ksupport(temp.flatten(), k, 1 / mu)
                X = temp.reshape(na, nb)

                # 更新 Z
                Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

                # 计算残差和变化
                dY1 = A @ Z - B
                dY2 = X - Z
                chgX = np.max(np.abs(Xk - X))
                chgZ = np.max(np.abs(Zk - Z))
                chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
                    print(f"Stage {stage}, Iter {total_iter}, mu={mu:.2e}, err={err}")

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # 每阶段更新惩罚参数
            Y1, Y2, mu = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, max_mu)

    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return X, err, total_iter

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

# 生成玩具数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# k-support norm 正则化
k = 10
X, err, iter = ksupport_la_admm(A, B, k, opts)
print(f"Final Iteration: {iter}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('k-support Norm Regularization Result (LA-ADMM)')
# plt.show()