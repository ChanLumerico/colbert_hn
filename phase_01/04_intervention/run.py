import sys
import os
import torch
import yaml
import importlib

# Resolve paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from shared.data_utils import load_beir_dataset, save_json
from shared.colbert_inspector import ColBERTInspector
from reranking import rerank_with_router
from baselines import compute_all_metrics

# Dynamic import for module starting with a digit
router_mod = importlib.import_module("03_router_training.router_model")
LayerRouter = router_mod.LayerRouter

def main():
    # 1. Load config
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    datasets = config["datasets"]
    model_name = config["model"]["colbert"]
    device = config["model"]["device"]
    
    # [Ablation Configs]
    abl_cfg = config["ablation"]
    exp_id = abl_cfg["experiment_id"]
    target_layers = abl_cfg["layers"]
    fusion_type = abl_cfg["fusion_type"]
    hidden_dims = abl_cfg["hidden_dims"]
    norm_type = abl_cfg["norm_type"]
    
    # 2. Initialize Models
    inspector = ColBERTInspector(model_name=model_name, device=device)
    
    router = LayerRouter(
        num_layers=len(target_layers), 
        embed_dim=128, 
        hidden_dims=hidden_dims, 
        fusion_type=fusion_type
    )
    model_path = f"outputs/03_router_training/{exp_id}/router_model.pt"
    if not os.path.exists(model_path):
        print(f"Trained router not found at {model_path}. Please run 03_router_training/run.py first.")
        return
    router.load_state_dict(torch.load(model_path, map_location=device))
    router.to(device)
    router.eval()
    
    all_results = {}
    
    # 3. Evaluate each dataset
    for ds in datasets:
        print(f"\n--- Evaluating Intervention for {ds} ---")
        corpus, queries, qrels = load_beir_dataset(ds, split="test")
        
        # Select a test sample for evaluation
        test_query_ids = list(queries.keys())[:50]
        test_queries = {qid: queries[qid] for qid in test_query_ids}
        
        # A. Vanilla ColBERT
        print(f"Retrieving with Vanilla ColBERT for {len(test_queries)} queries...")
        vanilla_results = inspector.batch_retrieve(test_queries, corpus, top_k=top_k_retrieval)
        vanilla_metrics = compute_all_metrics(qrels, vanilla_results)
        
        # B. Intervention (ColBERT + LayerRouter)
        print("Applying LayerRouter Intervention (Reranking)...")
        intervention_results = rerank_with_router(
            test_queries, corpus, vanilla_results, 
            inspector, router, device=device,
            target_layers=target_layers,
            norm_type=norm_type
        )
        intervention_metrics = compute_all_metrics(qrels, intervention_results)
        
        all_results[ds] = {
            "vanilla": vanilla_metrics,
            "intervention": intervention_metrics,
            "improvement": {m: intervention_metrics[m] - vanilla_metrics[m] for m in vanilla_metrics}
        }
        
        print(f"\nResults for {ds}:")
        print(f"  NDCG@10: {vanilla_metrics['NDCG@10']:.4f} -> {intervention_metrics['NDCG@10']:.4f} "
              f"({all_results[ds]['improvement']['NDCG@10']:+.4f})")
        print(f"  MRR@10:  {vanilla_metrics['MRR@10']:.4f} -> {intervention_metrics['MRR@10']:.4f} "
              f"({all_results[ds]['improvement']['MRR@10']:+.4f})")
              
    # 4. Save summary
    os.makedirs("outputs/04_intervention", exist_ok=True)
    save_json(all_results, "outputs/04_intervention/summary.json")
    
    print("\n" + "="*80)
    print("Intervention Evaluation Complete!")
    print("Summary saved to outputs/04_intervention/summary.json")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
