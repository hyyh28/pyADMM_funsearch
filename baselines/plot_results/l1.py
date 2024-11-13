import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu):
    return np.linalg.norm(X.ravel(), 1) + np.sum(Y1 * (A @ Z - B)) + np.sum(Y2 * (X - Z)) + (mu / 2) * (np.linalg.norm(A @ Z - B, 'fro')**2 + np.linalg.norm(X - Z, 'fro')**2)


def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    
    # 自适应惩罚参数更新，根据当前阶段数动态调整
    if iter_count >= stage_count:
        mu = min(rho * mu, max_mu)  # 增大惩罚参数
        iter_count = 0  # 重置迭代计数器
    return Y1, Y2, mu, rho, iter_count + 1

def l1_accelerated(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    history_X = []
    history_Z = []

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        Y1 += mu * dY1
        Y2 += mu * dY2

        if len(history_X) == anderson_m:
            history_X.pop(0)
            history_Z.pop(0)
        
        history_X.append(X.copy())
        history_Z.append(Z.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            G = np.vstack([delta_X, delta_Z]).T
            g = np.hstack([(X - history_X[0]).flatten(), (Z - history_Z[0]).flatten()])
            if G.shape[0] == g.shape[0]:  # Ensure compatibility before lstsq
                alpha = np.linalg.lstsq(G, g, rcond=None)[0]
                X -= np.dot(alpha, delta_X[-1])
                Z -= np.dot(alpha, delta_Z[-1])

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        obj = np.linalg.norm(X.ravel(), 1)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)
        
        if chg < tol:
            break

        # Adaptive mu update
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

    return errors_list, objective_list

def l1_admm(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    stage_count = opts.get('stage_count', 100)  # 每个阶段的最大迭代次数

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    iter_count = 0  # 当前阶段的迭代计数器
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        
        # update X
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        
        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = np.linalg.norm(X.ravel(), 1)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # 自适应更新惩罚参数
        Y1, Y2, mu, rho, iter_count = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu, iter_count, stage_count)

    return errors_list, objective_list

def l1_fast_admm(A, B, opts):
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
    objective_list = []
    errors_list = []

    for iter in range(max_iter):
        # x-update: 更新 X
        Xk = X.copy()
        X = prox_l1(Z - U, lam/rho)

        # z-update: 更新 Z
        Zk = Z.copy()
        Z = invAtA_I @ (A.T @ B + (lam / rho) * (U + X))

        # u-update: 更新拉格朗日乘子 U
        U = U + (X - Z)
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))

        # 计算残差
        primal_resid = np.linalg.norm(X - Z, 'fro')
        dual_resid = np.linalg.norm(rho * (Z - Z_prev), 'fro') if iter > 0 else 0
        err = np.sqrt(primal_resid ** 2 + dual_resid ** 2)
        chg = max([chgX, chgZ, np.max(np.abs(primal_resid)), np.max(np.abs(dual_resid))])
        obj = np.linalg.norm(X.ravel(), 1)

        if DEBUG and (iter == 0 or (iter + 1) % 10 == 0):
            print(f'iter {iter+1}, obj={obj}, err={err}, primal_resid={primal_resid}, dual_resid={dual_resid}')
        errors_list.append(err)
        objective_list.append(obj)

        # 检查收敛条件
        if chg < tol:
            break

        Z_prev = Z.copy()  # 保存上一次的 Z 用于计算双重残差

    obj = np.linalg.norm(X.ravel(), 1)
    return errors_list, objective_list

def l1_autoAdmm(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
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
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        # update X
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = np.linalg.norm(X.ravel(), 1)
        err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)
        
        if chg < tol:
            break

        # Adjust rho dynamically
        # rho_update = 2 * (np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2) / (np.linalg.norm(X - Z, 'fro')**2 + np.linalg.norm(A @ Z - B, 'fro')**2)
        # rho = min(max(rho_update, 1.1), 10)
        rho_update = 1.01 + 0.1 * (chgX / chgZ)
        rho = min(max(rho * rho_update, 1.1), 10.0)
        
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)
    
    return errors_list, objective_list

if __name__ == "__main__":
    # Generate toy data
    d = 10
    na = 200
    nb = 100

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X
    b = B[:, 0]

    # Options for the elastic net minimization
    opts = {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.1,
        'mu': 1e-4,
        'max_mu': 1e10,
        'DEBUG': 1
    }
    

    # Regularization parameter for elastic net
    lambda_ = 0.01
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = l1_accelerated(A, B, opts)
        elif key == "ADMM":
            error_list, objective_list = l1_admm(A, B, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = l1_fast_admm(A, B, opts)
        else:
            error_list, objective_list = l1_autoAdmm(A, B, opts)

        if len(objective_list) < 1000:
            objective_list.extend([objective_list[-1]] * (1000 - len(objective_list)))
        objective_list_map[key] = objective_list

    # Plotting the objective lists with different line styles and markers
    plt.figure(figsize=(10, 6))
    for key, objective_list in objective_list_map.items():
        plt.plot(objective_list, label=key)

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Value Convergence for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig("./l1.pdf")

