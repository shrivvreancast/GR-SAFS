# grsafs/graph_lasso.py
import numpy as np
import time
from scipy.sparse.linalg import eigsh, LinearOperator


def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def graph_lasso_pgd(G_std, y, L_sparse, lambda1, lambda2,
                    max_iter=1500, tol=1e-4, verbose=False):
    """
    Proximal Gradient Descent solver for the Graph-Lasso objective.
    """
    n, p = G_std.shape

    def H_matvec(v):
        Gv = G_std @ v
        return (G_std.T @ Gv) / n + lambda2 * (L_sparse @ v)

    H_op = LinearOperator((p, p), matvec=H_matvec)
    t0 = time.time()

    # Estimate Lipschitz constant
    L_lip = eigsh(H_op, k=1, which='LM', tol=1e-3, maxiter=300,
                  return_eigenvectors=False)[0]
    if verbose:
        print(f"  [Lip] {time.time() - t0:.1f}s, L={L_lip:.2f}")

    eta = 1.0 / L_lip
    Gtx = G_std.T @ y / n
    beta = np.zeros(p)

    for k in range(max_iter):
        grad = H_matvec(beta) - Gtx
        beta_new = soft_threshold(beta - eta * grad, eta * lambda1)
        diff = np.linalg.norm(beta_new - beta)
        rel = diff / max(np.linalg.norm(beta), 1e-10)
        beta = beta_new

        if rel < tol and k > 10:
            break

    nnz = np.sum(np.abs(beta) > 1e-8)
    if verbose:
        print(f"  [PGD] converged in {k + 1} iters, nnz={nnz}")

    return beta, {'n_iter': k + 1, 'n_nonzero': nnz, 'converged': rel < tol}