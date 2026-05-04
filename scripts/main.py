"""
========================================================================
Experiment: TCGA-LUAD Main Analysis for GR-SAFS
========================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import json
import warnings
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as lifelines_cindex

# ======================================================================
# 导入我们刚刚拆分封装好的 GR-SAFS 核心算法包
# ======================================================================
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))  # 确保能找到 grsafs 包

from grsafs.coexpression_network import build_coexpression_laplacian
from grsafs.graph_lasso import graph_lasso_pgd
from grsafs.qp_fusion import solve_qp_weights
from grsafs.ecdf_alignment import compute_ensemble_scores

warnings.filterwarnings('ignore')
np.random.seed(42)


# ======================================================================
# 辅助函数：计算 OOF 预测 (由于涉及交叉验证，放在实验脚本中)
# ======================================================================
def compute_oof_predictions(G_std, y, L, lambda1, lambda2, n_folds=5, seed=42):
    n, p = G_std.shape
    P_oof = np.zeros((n, 2))
    imp_gl_acc = np.zeros(p)
    imp_rf_acc = np.zeros(p)
    imp_gl_signed_acc = np.zeros(p)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_i, (tr, val) in enumerate(kf.split(G_std)):
        print(f"    OOF fold {fold_i + 1}/{n_folds}...")
        G_tr, G_val = G_std[tr], G_std[val]
        y_tr, y_val = y[tr], y[val]

        beta, _ = graph_lasso_pgd(G_tr, y_tr, L, lambda1, lambda2)
        P_oof[val, 0] = G_val @ beta
        imp_gl_acc += np.abs(beta)
        imp_gl_signed_acc += beta

        rf = RandomForestRegressor(n_estimators=300, max_depth=5,
                                   min_samples_leaf=3, random_state=seed, n_jobs=-1)
        rf.fit(G_tr, y_tr)
        P_oof[val, 1] = rf.predict(G_val)
        imp_rf_acc += rf.feature_importances_

    return P_oof, imp_gl_acc / n_folds, imp_rf_acc / n_folds, imp_gl_signed_acc / n_folds


# ======================================================================
# 辅助函数：生存分析与画图 (精简保留了你原来的代码逻辑)
# ======================================================================
def kaplan_meier_validation(risk_scores, t, e, output_path, title="KM"):
    median_score = np.median(risk_scores)
    high = risk_scores >= median_score;
    low = ~high
    lr = logrank_test(t[high], t[low], event_observed_A=e[high], event_observed_B=e[low])
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_h = KaplanMeierFitter()
    kmf_h.fit(t[high], event_observed=e[high], label='High Risk')
    kmf_h.plot_survival_function(ax=ax, color='red', linewidth=2)
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(t[low], event_observed=e[low], label='Low Risk')
    kmf_l.plot_survival_function(ax=ax, color='blue', linewidth=2)
    ax.set_title(f'{title}\nLog-rank p = {lr.p_value:.2e}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (days)', fontsize=12);
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.legend(fontsize=12);
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight');
    plt.close()
    return lr.p_value


def cox_cindex(risk_scores, t, e):
    return lifelines_cindex(t, -risk_scores, e)


def plot_top_genes(gene_names, scores, output_path, top_k=20):
    idx = np.argsort(scores)[::-1][:top_k]
    names = [gene_names[i] for i in idx];
    vals = scores[idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_k))
    bars = ax.barh(range(top_k), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k));
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel('GR-SAFS Score', fontsize=12)
    ax.set_title(f'Top {top_k} Candidate Prognostic Genes', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    plt.tight_layout();
    plt.savefig(output_path, dpi=200, bbox_inches='tight');
    plt.close()


# ======================================================================
# 主流水线
# ======================================================================
def main(args):
    # 【改造点：告别硬编码绝对路径】
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  TCGA-LUAD GR-SAFS Analysis Pipeline")
    print("=" * 65)

    # Step 1: 加载
    print("\n[Step 1] 加载数据...")
    G_df = pd.read_csv(data_dir / "TCGA_LUAD_G.csv", index_col=0)
    y_df = pd.read_csv(data_dir / "TCGA_LUAD_y.csv", index_col=0)
    print(f"  G: {G_df.shape}, y: {y_df.shape}")

    # Step 2: 方差过滤 + 标准化
    print(f"\n[Step 2] 方差过滤 Top-10000...")
    top_genes_idx = G_df.var(axis=0).nlargest(10000).index
    G_filtered = G_df[top_genes_idx]
    gene_names_filtered = np.array(top_genes_idx.tolist())
    p = len(gene_names_filtered)

    scaler = StandardScaler()
    G_std = scaler.fit_transform(G_filtered.values).astype(np.float64)
    y_dev = y_df['deviance_residual'].values.astype(np.float64)
    y_time = y_df['time'].values.astype(np.float64)
    y_event = y_df['event'].values.astype(int)
    n = G_std.shape[0]

    # Step 3: 共表达网络 (调用 grsafs 库)
    print(f"\n[Step 3] 构建共表达网络...")
    L_sparse, D = build_coexpression_laplacian(G_std, tau=0.3)

    # Step 4: GR-SAFS 主分析 (预设已搜索好的最优参数)
    print(f"\n[Step 4] GR-SAFS 主分析 (λ1=0.2, λ2=0.05, γ=10)...")
    P_oof, imp_gl, imp_rf, imp_gl_signed = compute_oof_predictions(
        G_std, y_dev, L_sparse, lambda1=0.2, lambda2=0.05, n_folds=5, seed=42)

    w_opt, Omega = solve_qp_weights(P_oof, y_dev, gamma=10)

    # eCDF 融合 (调用 grsafs 库)
    final_scores = compute_ensemble_scores(imp_gl, imp_rf, w_opt, p)

    # Step 5: 输出结果
    print(f"\n[Step 5] Top-20 候选基因输出与生存验证...")
    ranking = np.argsort(final_scores)[::-1]
    top_k_idx = ranking[:20]

    # 画图
    plot_top_genes(gene_names_filtered, final_scores, output_dir / "top_genes_bar.png", top_k=20)

    # 风险评分
    G_topk = G_std[:, top_k_idx]
    direction = np.sign(imp_gl_signed[top_k_idx])
    scores_20 = final_scores[top_k_idx]
    risk_scores = G_topk @ (direction * scores_20)

    km_p = kaplan_meier_validation(
        risk_scores, y_time, y_event, output_dir / "KM_survival.png",
        title="TCGA-LUAD: GR-SAFS Risk Stratification")
    c_index = cox_cindex(risk_scores, y_time, y_event)

    print(f"\n{'=' * 65}")
    print(f"  完成! C-index: {c_index:.4f}, KM p-value: {km_p:.2e}")
    print(f"  Top-10 Genes: {', '.join(gene_names_filtered[top_k_idx][:10])}")
    print(f"  输出已保存至: {output_dir}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR-SAFS on TCGA-LUAD")
    # 使用相对路径，假设运行脚本时在项目根目录
    parser.add_argument("--data_dir", type=str, default="../data/processed",
                        help="Directory containing preprocessed data")
    parser.add_argument("--output_dir", type=str, default="../outputs/tcga_luad", help="Directory to save outputs")
    args = parser.parse_args()

    main(args)