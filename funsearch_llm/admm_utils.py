from __future__ import annotations
import numpy as np
from typing import Tuple
datasets = {}


datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'd': 10,
    'na': 200,
    'nb': 100
},}

datasets['groupl1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'd': 10,
    'na': 200,
    'nb': 100,
    'g_num': 5
},}

datasets['elasticnet'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda_': 0.01
},}

datasets['trace_lasso'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'd': 10,
    'na': 200
},}

datasets['ksupport'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'd': 10,
    'na': 200,
    'nb': 100,
    'k': 10
},}

datasets['l1R'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1',
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda_val': 0.01
},}

datasets['groupl1R'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1',
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda_val': 1,
    'G': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]  # example group partition
},}

datasets['elasticnetR'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1',
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda1': 10,
    'lambda2': 10
},}

datasets['tracelassoR'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1',
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda_': 0.1
},}

datasets['rpca'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1,
    'loss': 'l1',
    'n1': 100,
    'n2': 200,
    'r': 10,
    'p': 0.1,
    'lambda_': 1 / np.sqrt(max(100, 200))
},}

datasets['lrmc'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l1',
    'n1': 100,
    'n2': 200,
    'r': 5,
    'p': 0.6,
    'lambda_': 0.1
    }
}

datasets['lrr'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0,
    'loss': 'l21',
    'd': 10,
    'na': 200,
    'nb': 100,
    'lambda_': 0.001
    }
}

datasets['latlrr'] = {
    'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.1,
        'mu': 1e-3,
        'max_mu': 1e10,
        'DEBUG': 0,
        'loss': 'l1',
        'd': 10,
        'na': 200,
        'nb': 100,
        'lambda_': 0.1
    }
}


datasets['lrsr'] = {
    'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.2,
        'mu': 1e-3,
        'max_mu': 1e10,
        'DEBUG': 0,
        'loss': 'l21',
        'd': 10,
        'na': 200,
        'nb': 100,
        'lambda1': 0.1,
        'lambda2': 4
    }
}

datasets['igc'] = {
    'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.2,
        'mu': 1e-3,
        'max_mu': 1e10,
        'DEBUG': 1,
        'loss': 'l1',
        'd': 10,
        'na': 200,
        'nb': 100,
        'n': 100,
        'r': 5,
        'lambda_': 0.1  # Since lambda_ = 1 / sqrt(n), and n = 100
    },
}


datasets['mlap'] = {
    'opts': {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.2,
        'mu': 1e-3,
        'max_mu': 1e10,
        'DEBUG': 0,
        'loss': 'l21',
        'n1': 100,
        'n2': 200,
        'K': 10,
        'lambda_': 0.1,
        'alpha': 0.2
    },
}



datasets['rmsc'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 1,
    'd': 10,
    'n': 100,
    'm': 10
},}

datasets['sparsesc'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 0,
    'n': 100,
    'k': 5,
    'lambda_': 0.001
},}

