# grsafs/ecdf_alignment.py
import numpy as np
from scipy.stats import rankdata


def compute_ensemble_scores(imp_gl, imp_rf, w, p):
    
    # eCDF zero-value truncated mapping
    pct_gl = rankdata(np.abs(imp_gl), method='min') / p
    pct_rf = rankdata(np.abs(imp_rf), method='min') / p

    # Weight transfer from prediction space to feature space
    fused_scores = (w[0] / w.sum()) * pct_gl + (w[1] / w.sum()) * pct_rf
    return fused_scores