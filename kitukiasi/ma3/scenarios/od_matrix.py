from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def scale_od(od_matrix: csr_matrix, od_log_multipliers: np.ndarray, global_scale: float) -> coo_matrix:
    coo = od_matrix.tocoo()
    if od_log_multipliers.size == 0:
        mult = np.ones(coo.nnz)
    else:
        mult = np.exp(np.asarray(od_log_multipliers, dtype=np.float64))
        if mult.shape[0] != coo.nnz:
            raise ValueError(f"od_log_multipliers len {mult.shape[0]} != od nnz {coo.nnz}")
    scaled = coo.data.astype(np.float64) * float(global_scale) * mult
    return coo_matrix((scaled, (coo.row, coo.col)), shape=coo.shape).tocoo()
