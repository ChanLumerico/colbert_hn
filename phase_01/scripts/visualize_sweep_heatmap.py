import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_sweep_heatmap():
    # 1. Load results
    results_path = "phase_01/outputs/intervention/gamma_sweep_results.json"
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # 2. Process data for Heatmap (Relative Improvement %)
    heatmap_data = []
    datasets = list(data.keys())
    gammas = [item["gamma"] for item in data[datasets[0]]]
    
    for ds in datasets:
        baseline_ndcg = data[ds][0]["ndcg@10"] 
        row = []
        for item in data[ds]:
            rel_imp = ((item["ndcg@10"] - baseline_ndcg) / (baseline_ndcg + 1e-9)) * 100
            row.append(rel_imp)
        heatmap_data.append(row)
        
    df = pd.DataFrame(heatmap_data, index=datasets, columns=gammas)
    
    # 3. Plot
    plt.figure(figsize=(14, 6))
    
    # Revert to full scale but keep it centered at 0
    ax = sns.heatmap(df, annot=False, cmap="RdBu", center=0, 
                     linewidths=.5, linecolor='white',
                     cbar_kws={'label': 'Relative Improvement (%)'})
    
    # 4. Custom Annotations: Bold the best in each row
    for i in range(len(datasets)):
        row_values = df.iloc[i, :].values
        max_val = np.max(row_values)
        for j in range(len(gammas)):
            val = row_values[j]
            is_best = (val == max_val and val > 0)
            
            text_color = "white" if abs(val) > 20 else "black"
            font_weight = "bold" if is_best else "normal"
            font_size = 11 if is_best else 10
            
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", 
                    ha="center", va="center", 
                    color=text_color, 
                    fontweight=font_weight,
                    fontsize=font_size)
    
    plt.title("Intervention Performance Sweep: NDCG@10 Relative Improvement (%)", fontsize=14, pad=20)
    plt.xlabel("Intervention Strength (Gamma)", fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    
    # 5. Save
    out_dir = "report/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig7_sweep_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Heatmap saved to {out_path}")

if __name__ == "__main__":
    visualize_sweep_heatmap()
