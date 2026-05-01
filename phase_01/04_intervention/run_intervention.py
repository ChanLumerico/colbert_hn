import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
PHASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PHASE_ROOT)

import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from phase_01.shared.data_utils import load_beir_dataset
from phase_01.shared.colbert_inspector import ColBERTInspector
from phase_01.shared.metrics import ndcg_at_k, mrr_at_k
import importlib
router_module = importlib.import_module("phase_01.03_router_training.router_model")
LayerRouter = router_module.LayerRouter
label_module = importlib.import_module("phase_01.03_router_training.label_design")
apply_normalization = label_module.apply_normalization

def run_intervention(dataset_name: str, 
                     checkpoint_dir: str, 
                     gamma: float = 1.0, 
                     top_k: int = 100):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    corpus, queries, qrels = load_beir_dataset(dataset_name, split="test")
    # Speed up for demonstration purposes: limit to first 50 queries
    queries = {k: queries[k] for k in list(queries.keys())[:50]}
    
    inspector = ColBERTInspector(device=device)
    
    # 1. Baseline Retrieval
    print(f"\n[{dataset_name}] Running ColBERT Baseline Retrieval...")
    baseline_results = inspector.batch_retrieve(queries, corpus, top_k=top_k)
    
    # Calculate baseline metrics
    baseline_metrics = {
        "ndcg@10": ndcg_at_k(qrels, baseline_results, k=10),
        "mrr@10": mrr_at_k(qrels, baseline_results, k=10)
    }
    print(f"Baseline: NDCG@10: {baseline_metrics['ndcg@10']:.4f}, MRR@10: {baseline_metrics['mrr@10']:.4f}")
    
    # 2. Load Router
    # Golden recipe: abl2_inter_norm (Interaction, LayerNorm, Hidden 256)
    router = LayerRouter(num_layers=5, embed_dim=128, hidden_dims=[256], fusion_type="interaction")
    model_path = os.path.join(checkpoint_dir, "router_model.pth")
    router.load_state_dict(torch.load(model_path, map_location=device))
    router.to(device)
    router.eval()
    
    # 3. Intervention (Re-ranking)
    print(f"[{dataset_name}] Running Router Intervention (gamma={gamma})...")
    intervened_results = {}
    
    target_layers = [0, 3, 6, 9, 12]
    
    for qid, qtext in tqdm(queries.items(), desc="Re-ranking"):
        retrieved_docs = baseline_results[qid]
        if not retrieved_docs:
            intervened_results[qid] = []
            continue
            
        doc_ids = [d[0] for d in retrieved_docs]
        base_scores = torch.tensor([d[1] for d in retrieved_docs], dtype=torch.float32).to(device)
        dtexts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        # Extract features
        q_layer_dict = inspector.get_all_layer_reprs([qtext])
        d_layer_dict = inspector.get_all_layer_reprs(dtexts)
        
        q_reps = torch.stack([q_layer_dict[l] for l in target_layers], dim=1).to(device) # (1, 5, 128)
        d_reps = torch.stack([d_layer_dict[l] for l in target_layers], dim=1).to(device) # (K, 5, 128)
        
        # Expand q_reps to match d_reps batch size
        q_reps = q_reps.expand(len(doc_ids), -1, -1)
        
        # Apply layer norm (Golden Recipe)
        q_reps = apply_normalization(q_reps, "layer_norm")
        d_reps = apply_normalization(d_reps, "layer_norm")
        
        with torch.no_grad():
            router_logits = router(q_reps, d_reps) # shape: (K,)
            
        # Intervention logic: New_Score = Base_Score + gamma * Router_Logit
        # If router thinks it's a Hard Negative, logit is negative -> penalizes score
        # If router thinks it's a Positive, logit is positive -> boosts score
        new_scores = base_scores + gamma * router_logits
        
        # Re-sort
        sorted_indices = torch.argsort(new_scores, descending=True)
        reranked_docs = [(doc_ids[idx], new_scores[idx].item()) for idx in sorted_indices]
        intervened_results[qid] = reranked_docs
        
    intervened_metrics = {
        "ndcg@10": ndcg_at_k(qrels, intervened_results, k=10),
        "mrr@10": mrr_at_k(qrels, intervened_results, k=10)
    }
    print(f"Intervened: NDCG@10: {intervened_metrics['ndcg@10']:.4f}, MRR@10: {intervened_metrics['mrr@10']:.4f}")
    
    return baseline_metrics, intervened_metrics

if __name__ == "__main__":
    datasets = ["scifact", "nfcorpus", "scidocs"]
    
    results = {}
    for ds in datasets:
        # Cross-validation setup
        chkpt = os.path.join(PHASE_ROOT, f"outputs/ablation/abl2_inter_norm/fold_{ds}/training")
        base_m, int_m = run_intervention(ds, chkpt, gamma=1.0, top_k=50) 
        results[ds] = {"baseline": base_m, "intervened": int_m}
        
    out_dir = os.path.join(PHASE_ROOT, "outputs/intervention")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nIntervention completed successfully!")
