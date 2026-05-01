import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi

def generate_advanced_plots():
    with open('outputs/ablation_results.json', 'r') as f:
        results = json.load(f)

    os.makedirs('report/figures', exist_ok=True)
    
    # Set academic style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    colors = sns.color_palette("deep")

    # Helper to calculate mean across folds for a specific metric
    def get_mean_metric(model_id, metric):
        folds = ['scifact', 'nfcorpus', 'scidocs']
        vals = [results[model_id]['folds'][fold][metric] for fold in folds]
        return np.mean(vals)

    # ---------------------------------------------------------
    # 1. Fusion Strategy (Grouped Bar Chart)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fusion_ids = ['baseline', 'abl2_diff_norm', 'abl2_inter_norm']
    fusion_labels = ['Concat\n(Baseline)', 'Difference\n(|q-d|)', 'Interaction\n(q⊙d)']
    
    ndcg10 = [results[fid]['summary_ndcg'] for fid in fusion_ids]
    mrr10 = [results[fid]['summary_mrr'] for fid in fusion_ids]
    recall10 = [get_mean_metric(fid, 'Recall@10') for fid in fusion_ids]

    x = np.arange(len(fusion_labels))
    width = 0.25

    ax.bar(x - width, ndcg10, width, label='NDCG@10', color=colors[0], edgecolor='black')
    ax.bar(x, mrr10, width, label='MRR@10', color=colors[1], edgecolor='black')
    ax.bar(x + width, recall10, width, label='Recall@10', color=colors[2], edgecolor='black')

    ax.set_ylabel('Metric Score', fontweight='bold')
    ax.set_title('Figure 1: Performance Comparison by Fusion Strategy', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fusion_labels, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('report/figures/fig1_fusion.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Layer Selection (Bar Chart with Trendline)
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    layer_ids = ['abl1_final_only', 'abl1_extremes', 'abl1_sparse', 'baseline', 'abl1_all_layers']
    layer_labels = ['Final Only\n[12]', 'Extremes\n[0,12]', 'Sparse\n[0,6,12]', 'Base\n[0,3,6,9,12]', 'All Layers\n[0-12]']
    
    layer_ndcg = [results[lid]['summary_ndcg'] for lid in layer_ids]
    
    ax1.bar(layer_labels, layer_ndcg, color=colors[3], alpha=0.7, edgecolor='black', label='NDCG@10')
    ax1.plot(layer_labels, layer_ndcg, marker='o', color='maroon', linewidth=2, markersize=8, label='Trend')
    
    ax1.set_ylabel('Summary NDCG@10', fontweight='bold')
    ax1.set_title('Figure 2: Impact of Layer Selection (Representation Drift)', pad=15, fontweight='bold')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig('report/figures/fig2_layers.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Architecture Complexity (Line Plot showing Overfitting)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    arch_ids = ['abl3_linear', 'baseline', 'abl3_wide', 'abl3_deep']
    arch_labels = ['Linear\n(0-Hidden)', 'Base\n(256)', 'Wide\n(512)', 'Deep\n(512-256-128)']
    
    arch_ndcg = [results[aid]['summary_ndcg'] for aid in arch_ids]
    arch_mrr = [results[aid]['summary_mrr'] for aid in arch_ids]

    ax.plot(arch_labels, arch_ndcg, marker='s', linewidth=2.5, markersize=8, color=colors[0], label='NDCG@10')
    ax.plot(arch_labels, arch_mrr, marker='^', linewidth=2.5, markersize=8, color=colors[1], label='MRR@10', linestyle='--')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Model Complexity', fontweight='bold')
    ax.set_title('Figure 3: Architecture Complexity vs Performance', pad=15, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('report/figures/fig3_architecture.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Normalization Impact (Horizontal Bar Chart)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    norm_ids = ['baseline', 'abl5_l2norm', 'abl5_layernorm']
    norm_labels = ['Raw (No Norm)', 'L2 Normalization', 'Layer Normalization']
    
    norm_ndcg = [results[nid]['summary_ndcg'] for nid in norm_ids]
    
    y_pos = np.arange(len(norm_labels))
    ax.barh(y_pos, norm_ndcg, align='center', color=sns.color_palette("pastel")[4], edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(norm_labels, fontweight='bold')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Summary NDCG@10', fontweight='bold')
    ax.set_title('Figure 4: Impact of Input Normalization', pad=15, fontweight='bold')
    
    # Add values on bars
    for i, v in enumerate(norm_ndcg):
        ax.text(v + 0.002, i + 0.05, f'{v:.4f}', fontweight='bold')

    plt.tight_layout()
    plt.savefig('report/figures/fig4_normalization.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 5. Radar Chart: Baseline vs Best Model
    # ---------------------------------------------------------
    categories = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10', 'Recall@10']
    N = len(categories)
    
    base_vals = [get_mean_metric('baseline', cat) for cat in categories]
    best_vals = [get_mean_metric('abl2_inter_norm', cat) for cat in categories]
    
    # Repeat first value to close the circle
    base_vals += base_vals[:1]
    best_vals += best_vals[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10, fontweight='bold')
    ax.set_rlabel_position(0)
    
    ax.plot(angles, base_vals, linewidth=2, linestyle='solid', label='Baseline (Concat)', color=colors[0])
    ax.fill(angles, base_vals, color=colors[0], alpha=0.2)
    
    ax.plot(angles, best_vals, linewidth=2, linestyle='solid', label='Best Model (Interaction)', color=colors[2])
    ax.fill(angles, best_vals, color=colors[2], alpha=0.2)
    
    plt.title('Figure 5: Multi-metric Radar Comparison', size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('report/figures/fig5_radar.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 6. Domain Generalization Heatmap (Fold Analysis)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = ['scifact', 'nfcorpus', 'scidocs']
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@10', 'Recall@10']
    
    heatmap_data = np.zeros((len(folds), len(metrics)))
    for i, fold in enumerate(folds):
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = results['abl2_inter_norm']['folds'][fold][metric]
            
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=metrics, yticklabels=[f.upper() for f in folds], ax=ax)
    
    ax.set_title('Figure 6: Best Model Domain Generalization (Heatmap)', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/figures/fig6_heatmap.png', dpi=300)
    plt.close()

    print("Advanced figures generated successfully.")

if __name__ == "__main__":
    generate_advanced_plots()
