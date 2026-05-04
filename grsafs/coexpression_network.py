# grsafs/coexpression_network.py
import numpy as np
from scipy import sparse
import time


def build_coexpression_laplacian(G_std, tau=0.3):
    """
    Build the Graph Laplacian matrix from standardized gene expression data.
    """
    n, p = G_std.shape
    print(f"[Network] Building co-expression network (n={n}, p={p})...")
    t0 = time.time()

    R = G_std.T @ G_std / (n - 1)
    np.fill_diagonal(R, 1.0)
    R = (R + R.T) / 2.0

    A = np.maximum(np.abs(R) - tau, 0.0)
    np.fill_diagonal(A, 0.0)

    nnz = np.count_nonzero(A[np.triu_indices(p, k=1)])
    total = p * (p - 1) // 2
    print(f"  Sparsity = {1 - nnz / total:.2%}")

    D = A.sum(axis=1)
    L = np.diag(D) - A
    L_sparse = sparse.csr_matrix(L)

    print(f"  Completed in {time.time() - t0:.1f}s")
    return L_sparse, D