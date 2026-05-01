import sys
import os
import torch
import yaml
import importlib

# Resolve paths
PHASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PHASE_ROOT not in sys.path:
    sys.path.append(PHASE_ROOT)

from shared.data_utils import load_beir_dataset, save_json
from shared.colbert_inspector import ColBERTInspector
from failure_breakdown import categorize_queries, get_query_details

# Dynamic import for modules starting with digits
router_mod = importlib.import_module("03_router_training.router_model")
LayerRouter = router_mod.LayerRouter

rerank_mod = importlib.import_module("04_intervention.reranking")
rerank_with_router = rerank_mod.rerank_with_router

def main():
    # 1. Load config
    config_path = os.path.join(PHASE_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    datasets = config["datasets"]
    model_name = config["model"]["colbert"]
    device = config["model"]["device"]
    
    # 2. Initialize Models
    inspector = ColBERTInspector(model_name=model_name, device=device)
    router = LayerRouter(num_layers=5, embed_dim=128)
    model_pth = os.path.join(PHASE_ROOT, "outputs/03_router_training/router_model.pt")
    router.load_state_dict(torch.load(model_pth, map_location=device))
    router.to(device)
    router.eval()
    
    analysis_results = {}
    
    # 3. Analyze each dataset
    for ds in datasets:
        print(f"\n--- Qualitative Analysis for {ds} ---")
        corpus, queries, qrels = load_beir_dataset(ds, split="test")
        
        # Sample queries for analysis
        test_query_ids = list(queries.keys())[:30]
        test_queries = {qid: queries[qid] for qid in test_query_ids}
        
        # Run Retrieval and Intervention
        vanilla_results = inspector.batch_retrieve(test_queries, corpus, top_k=50)
        intervention_results = rerank_with_router(test_queries, corpus, vanilla_results, inspector, router, device=device)
        
        # Categorize
        categories = categorize_queries(qrels, vanilla_results, intervention_results)
        
        # Pick examples
        examples = {}
        for cat, qids in categories.items():
            if qids:
                target_qid = qids[0]
                examples[cat] = get_query_details(
                    target_qid, queries, corpus, qrels, 
                    vanilla_results[target_qid], intervention_results[target_qid]
                )
        
        analysis_results[ds] = {
            "counts": {cat: len(qids) for cat, qids in categories.items()},
            "examples": examples
        }
        
        print(f"Stats for {ds}:")
        for cat, count in analysis_results[ds]["counts"].items():
            print(f"  {cat:<15}: {count}")
            
    # 4. Save analysis
    out_dir = os.path.join(PHASE_ROOT, "outputs/05_analysis")
    os.makedirs(out_dir, exist_ok=True)
    save_json(analysis_results, os.path.join(out_dir, "breakdown.json"))
    
    print("\n" + "="*80)
    print("Failure Analysis Complete!")
    print(f"Detailed breakdown saved to {out_dir}/breakdown.json")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
