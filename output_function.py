def sparsesc(L, lambda_, k, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    n = L.shape[0]
    P = np.zeros((n, n))
    Q = np.zeros_like(P)
    Y = np.zeros_like(P)

    def update_rho(rho, dY, P, Q):
        norm_dY = np.linalg.norm(dY, 'fro')
        norm_P = np.linalg.norm(P, 'fro')
        norm_Q = np.linalg.norm(Q, 'fro')
        if norm_dY > 10 * max(norm_P, norm_Q):
            return rho * 2
        elif max(norm_P, norm_Q) > 10 * norm_dY:
            return rho / 2
        return rho

    for iter_ in range(max_iter):
        Pk = P.copy()
        Qk = Q.copy()

        # Update P
        P = prox_l1(Q - (Y + L) / mu, lambda_ / mu)

        # Update Q
        temp = (P + Y / mu)
        temp = (temp + temp.T) / 2
        Q = project_fantope(temp, k)

        dY = P - Q
        chgP = np.max(np.abs(Pk - P))
        chgQ = np.max(np.abs(Qk - Q))
        chg = max(chgP, chgQ, np.max(np.abs(dY)))

        if DEBUG and (iter_ == 0 or (iter_ + 1) % 10 == 0):
            obj = np.trace(np.dot(P.T, L)) + lambda_ * np.sum(np.abs(Q))
            err = np.linalg.norm(dY, 'fro')
            print(f"iter {iter_+1}, mu={mu}, rho={rho}, obj={obj}, err={err}")

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)
        rho = update_rho(rho, dY, P, Q)

    obj = np.trace(np.dot(P.T, L)) + lambda_ * np.sum(np.abs(Q))
    err = np.linalg.norm(dY, 'fro')

    return P, obj, err, iter_

