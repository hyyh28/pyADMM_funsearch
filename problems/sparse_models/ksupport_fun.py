import numpy as np
import matplotlib.pyplot as plt

def prox_ksupport(v, k, lambda_val):
    """
    The proximal operator of the k-support norm of a vector

    min_x 0.5*lambda*||x||_{ksp}^2 + 0.5*||x - v||_2^2

    Parameters:
        v : numpy.ndarray
            Input vector
        k : int
            k parameter for k-support norm
        lambda_val : float
            Regularization parameter

    Returns:
        B : numpy.ndarray
            Output vector after applying the proximal operator
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
    """
    Binary search helper for prox_ksupport.

    Parameters:
        z : numpy.ndarray
            Sorted absolute values of input vector scaled by L
        array : numpy.ndarray
            Cumulative sum array
        low : int
            Low index for search
        high : int
            High index for search
        diff : float
            Accumulated difference
        k : int
            k parameter
        r : int
            r parameter
        L : float
            1 / lambda

    Returns:
        l : int
            Index found
        T : float
            Corresponding cumulative sum value
    """
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
    rho_update_factor = opts.get('rho_update_factor', 1.1)
    rho_tol = opts.get('rho_tol', 1e-3)

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
        if np.linalg.norm(dY1) > rho_tol * np.linalg.norm(dY2):
            rho *= rho_update_factor
        elif np.linalg.norm(dY2) > rho_tol * np.linalg.norm(dY1):
            rho /= rho_update_factor

        # Update dual variables
        Y1 += rho * dY1
        Y2 += rho * dY2
        mu = min(rho_update_factor * mu, max_mu)

    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return X, err, iter


# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0
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
X, err, iter = ksupport_admm(A, B, k, opts)
print(f"Final Iteration: {iter}, Error: {err}")
plt.stem(X[:, 0])
plt.title('k-support Norm Regularization Result')
plt.show()
