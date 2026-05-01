import sys
import os
import yaml

# Set root to phase_01
PHASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PHASE_ROOT not in sys.path:
    sys.path.append(PHASE_ROOT)

from shared.colbert_inspector import ColBERTInspector
from shared.data_utils import save_json
from confusion_rate import compute_confusion_by_dataset, summarize_across_datasets

def main():
    # 1. Load config
    config_path = os.path.join(PHASE_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Initialize ColBERTInspector
    inspector = ColBERTInspector(
        model_name=config["model"]["colbert"],
        device=config["model"]["device"]
    )
    
    # 3. Run analysis for each dataset
    all_results = {}
    for ds_name in config["datasets"]:
        res = compute_confusion_by_dataset(
            ds_name, 
            inspector, 
            top_k=config["retrieval"]["top_k"]
        )
        all_results[ds_name] = res
        
    # 4. Generate summary
    summary = summarize_across_datasets(all_results)
    
    # 5. Save summary
    out_path = os.path.join(PHASE_ROOT, "outputs/01_confusion_analysis/summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json(summary, out_path)
    
    # 6. Print summary table
    print("\n" + "="*80)
    print(f"{'Dataset':<15} | {'Global CR':<12} | {'Query CR':<12} | {'Mean Margin':<12}")
    print("-" * 80)
    for ds_name, stats in summary.items():
        print(f"{ds_name:<15} | {stats['global_confusion_rate']:<12.3f} | {stats['query_confusion_rate']:<12.3f} | {stats['mean_margin']:<12.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
