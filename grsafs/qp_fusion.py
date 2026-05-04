# grsafs/qp_fusion.py
import numpy as np
from scipy.optimize import minimize


def ledoit_wolf_shrink(S):
    """
    Ledoit-Wolf shrinkage for the residual covariance matrix.
    """
    p = S.shape[0]
    mu = np.trace(S) / p
    delta = S - mu * np.eye(p)
    denom = np.trace(S @ S) * p
    alpha = max(0.0, min(1.0, 1.0 - np.linalg.norm(delta, 'fro') ** 2 / denom)) if denom > 1e-12 else 1.0
    return alpha * S + (1 - alpha) * mu * np.eye(p), alpha


def solve_qp_weights(P_oof, y, gamma=1.0):
    """
    Solve the diversity-penalized Quadratic Programming problem.
    """
    n, K = P_oof.shape
    residuals = y[:, None] - P_oof
    Omega_star, alpha_lw = ledoit_wolf_shrink(np.cov(residuals.T))

    H = (P_oof.T @ P_oof) / n + gamma * Omega_star
    f = -(P_oof.T @ y) / n

    result = minimize(lambda w: 0.5 * w @ H @ w + f @ w,
                      np.ones(K) / K,
                      jac=lambda w: H @ w + f,
                      method='SLSQP',
                      bounds=[(0.10, 1.0)] * K,
                      constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                      options={'maxiter': 1000, 'ftol': 1e-12})
    w = result.x
    if np.any(np.isnan(w)):
        w = np.ones(K) / K
    w = np.maximum(w, 0)
    w /= w.sum()

    print(f"[QP Fusion] w_GL={w[0]:.4f}, w_RF={w[1]:.4f}, γ={gamma}")
    return w, Omega_star