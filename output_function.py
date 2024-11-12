def trace_lasso_admm(A, b, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    tau_incr = opts.get('tau_incr', 2)
    tau_decr = opts.get('tau_decr', 2)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))

    for iter in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()

        # Update x
        x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))
        
        # Update Z
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

        # Compute residuals
        dY1 = A @ x - b
        dY2 = A @ np.diag(x) - Z
        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        # Update rho based on residuals
        r_norm = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
        s_norm = mu * np.sqrt(np.linalg.norm(x - xk) ** 2 + np.linalg.norm(Z - Zk) ** 2)

        if r_norm > tau_incr * s_norm:
            rho *= tau_incr
            Y1 /= tau_incr
            Y2 /= tau_incr
        elif s_norm > tau_decr * r_norm:
            rho /= tau_decr
            Y1 *= tau_decr
            Y2 *= tau_decr

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = nuclearnorm
            err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
            print(f"iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}")

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    
    return x, obj, err, iter

