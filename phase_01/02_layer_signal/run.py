import sys
import os
import yaml
import numpy as np

# Resolve path for shared module
PHASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PHASE_ROOT not in sys.path:
    sys.path.append(PHASE_ROOT)

from shared.data_utils import load_beir_dataset, load_json, save_json
from shared.colbert_inspector import ColBERTInspector
from layer_auroc import compute_layer_auroc
from representation_geometry import analyze_geometry

def main():
    # 1. Load config
    config_path = os.path.join(PHASE_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    datasets = config["datasets"]
    model_name = config["model"]["colbert"]
    device = config["model"]["device"]
    
    # 2. Initialize Inspector
    print(f"Initializing ColBERT Inspector ({model_name}) on {device}...")
    inspector = ColBERTInspector(model_name=model_name, device=device)
    
    all_dataset_results = {}
    
    for ds in datasets:
        print(f"\n--- Analyzing Layer Signal for {ds} ---")
        
        # Load raw data to get texts
        corpus, queries, _ = load_beir_dataset(ds, split="test")
        
        # Load triplets from Step 1
        triplet_path = os.path.join(PHASE_ROOT, f"outputs/01_confusion_analysis/{ds}/results.json")
        if not os.path.exists(triplet_path):
            print(f"Step 1 results not found for {ds}. Please run 01_confusion_analysis/run.py first.")
            continue
            
        step1_data = load_json(triplet_path)
        # Reconstruct triplet list from Step 1 results 
        # Actually Step 1 results.json from previous implementation only saved stats, 
        # not the full triplet list. Let's assume we need to rebuild triplets or modify Step 1.
        # Wait, Step 1's compute_confusion_by_dataset saved results.json which has "per_query".
        # Let's rebuild triplets from "per_query" if possible, but that doesn't have doc_ids for all HNs.
        
        # To be robust, let's re-run building triplets or assume results.json should have had them.
        # Actually, let's just re-retrieve or re-build triplets since we have the inspector and data.
        print("Re-building triplets for signal analysis...")
        from shared.data_utils import build_triplets
        # We need the retrieved results. Let's just re-retrieve for a small sample of queries 
        # to keep Step 2 fast and self-contained.
        sample_query_ids = list(queries.keys())[:50] # Sample 50 queries for deep analysis
        sample_queries = {qid: queries[qid] for qid in sample_query_ids}
        
        # We need qrels too
        _, _, qrels = load_beir_dataset(ds, split="test")
        
        retrieved = inspector.batch_retrieve(sample_queries, corpus, top_k=50)
        triplets = build_triplets(sample_queries, qrels, retrieved, k=50)
        
        if not triplets:
            print(f"No triplets found for {ds}, skipping.")
            continue
            
        # 3. Run Analyses
        print("Running AUROC analysis...")
        auroc_results = compute_layer_auroc(triplets, corpus, queries, inspector)
        
        print("Running Geometry analysis...")
        geometry_results = analyze_geometry(triplets, corpus, queries, inspector)
        
        ds_results = {
            "auroc": auroc_results,
            "geometry": geometry_results
        }
        
        # Save dataset results
        output_path = os.path.join(PHASE_ROOT, f"outputs/02_layer_signal/{ds}/results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_json(ds_results, output_path)
        all_dataset_results[ds] = ds_results
        
    # 4. Save global summary
    summary_path = os.path.join(PHASE_ROOT, "outputs/02_layer_signal/summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    save_json(all_dataset_results, summary_path)
    
    # 5. Print Summary Table
    print("\n" + "="*80)
    print(f"{'Dataset':<12} | {'Layer':<6} | {'AUROC':<8} | {'Sim Separation':<15} | {'Dist Separation':<15}")
    print("-" * 80)
    for ds, results in all_dataset_results.items():
        for layer in sorted(results["auroc"].keys()):
            auroc = results["auroc"][layer]
            geom = results["geometry"][layer]
            print(f"{ds:<12} | {layer:<6} | {auroc:<8.3f} | {geom['sim_separation']:<15.3f} | {geom['dist_separation']:<15.3f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
