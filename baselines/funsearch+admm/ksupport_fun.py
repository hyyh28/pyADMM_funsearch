import numpy as np
import matplotlib.pyplot as plt

def prox_ksupport(v, k, lambda_val):
    # The proximal operator for the k-support norm
    # (Implementation remains as is)

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

def ksupport_admm(A, B, k, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
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
    invAtAI = np.linalg.inv(A.T @ A + I)
    rho = mu

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # Update X
        temp = Z - Y2 / rho
        temp = prox_ksupport(temp.flatten(), k, 1 / rho)
        X = temp.reshape(na, nb)
        
        # Update Z
        Z = invAtAI @ (-A.T @ Y1 / rho + AtB + Y2 / rho + X)

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
            print(f"iter {iter}, mu={mu}, rho={rho}, err={err}")

        if chg < tol:
            break

        # Update rho dynamically
        rho_update = 2 * (np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2) / (np.linalg.norm(X - Z, 'fro')**2 + np.linalg.norm(A @ Z - B, 'fro')**2)
        rho = min(max(rho_update, 1.1), 10)

    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return X, err, iter

def evaluate(opts, k, d=100, na=200, nb=100):
    # Generate random problem instance
    A = np.random.randn(d, na)
    X_true = np.random.randn(na, nb)
    B = A @ X_true

    # Solve using ksupport_admm
    _, err, iter = ksupport_admm(A, B, k, opts)
    return iter

def evaluate_multiple(opts, k, num_trials=10):
    total_iterations = 0
    for _ in range(num_trials):
        total_iterations += evaluate(opts, k)
    
    average_iterations = total_iterations / num_trials
    print(f'Average Iterations over {num_trials} trials: {average_iterations:.2f}')
    return average_iterations

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0
}

# 运行10次评估并输出平均结果
k = 10
evaluate_multiple(opts, k)