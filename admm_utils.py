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
    'd': 100,
    'na': 200,
    'nb': 100
},}