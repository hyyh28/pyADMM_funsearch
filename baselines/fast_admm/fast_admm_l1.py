import numpy as np

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def admm_l1(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.0)  # 使用 rho 作为惩罚参数
    lam = opts.get('lam', 1.0)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    U = np.zeros_like(X)  # 拉格朗日乘子

    AtA = A.T @ A
    I = np.eye(na)
    invAtA_I = np.linalg.inv(AtA + I)  # 预计算矩阵逆

    for iter in range(max_iter):
        # x-update: 更新 X
        Xk = X.copy()
        X = prox_l1(Z - U, lam/rho)

        # z-update: 更新 Z
        Zk = Z.copy()
        Z = invAtA_I @ (A.T @ B + lam/rho * (U + X))

        # u-update: 更新拉格朗日乘子 U
        U = U + (X - Z)
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))

        # 计算残差
        primal_resid = A @ Z - B
        dual_resid = X - Z

        err = np.sqrt(np.linalg.norm(primal_resid, 'fro') ** 2 + np.linalg.norm(dual_resid, 'fro') ** 2)
        chg = max([chgX, chgZ, np.max(np.abs(primal_resid)), np.max(np.abs(dual_resid))])

        if DEBUG and (iter == 0 or (iter + 1) % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            print(f'iter {iter+1}, obj={obj}, err={err}')

        # 检查收敛条件
        if chg < tol:
            break
        Z_prev = Z.copy()  # 保存上一次的 Z 用于计算双重残差

        r_norm = np.linalg.norm(np.concatenate([primal_resid.ravel(), dual_resid.ravel()]))
        s_norm = np.linalg.norm(np.concatenate([(X - Xk).ravel(), (Z - Zk).ravel()]))
        if r_norm > s_norm:
            rho = (1 + rho) * 2
        if s_norm < r_norm:
            rho /= 2



    obj = np.linalg.norm(X.ravel(), 1)
    return X, obj, iter

# 评估函数
def evaluate(instances: dict) -> float:
    d = instances['d']
    na = instances['na']
    nb = instances['nb']

    A = np.random.randn(d, na)
    X_true = np.random.randn(na, nb)
    B = A @ X_true

    opts = instances['opts']
    X2, obj, iter = admm_l1(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}')
    return -iter

datasets = {
    'l1': {'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1,
        'lambda': 1,
        'DEBUG': 1
    },
    'd': 10,
    'na': 200,
    'nb': 100}
}

evaluate(datasets['l1'])