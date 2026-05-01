import sys
import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add phase_01 root to sys.path
PHASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PHASE_ROOT not in sys.path:
    sys.path.append(PHASE_ROOT)

from shared.data_utils import load_beir_dataset
from shared.colbert_inspector import ColBERTInspector
from shared.metrics import ndcg_at_k, mrr_at_k
import importlib
router_module = importlib.import_module("03_router_training.router_model")
LayerRouter = router_module.LayerRouter
label_module = importlib.import_module("03_router_training.label_design")
apply_normalization = label_module.apply_normalization

def run_gamma_sweep(dataset_name: str, 
                    checkpoint_dir: str, 
                    gammas: List[float],
                    top_k: int = 50,
                    num_queries: int = 100):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    corpus, queries, qrels = load_beir_dataset(dataset_name, split="test")
    test_qids = list(queries.keys())[:num_queries]
    test_queries = {qid: queries[qid] for qid in test_qids}
    
    inspector = ColBERTInspector(device=device)
    
    # 2. Baseline Retrieval
    print(f"\n[{dataset_name}] 1/3: Running Baseline Retrieval & Feature Extraction...")
    baseline_results = inspector.batch_retrieve(test_queries, corpus, top_k=top_k)
    
    # 3. Load Router
    router = LayerRouter(num_layers=5, embed_dim=128, hidden_dims=[256], fusion_type="interaction")
    model_path = os.path.join(checkpoint_dir, "router_model.pth")
    router.load_state_dict(torch.load(model_path, map_location=device))
    router.to(device)
    router.eval()
    
    # 4. Cache Base Scores and Router Logits
    print(f"[{dataset_name}] 2/3: Caching Router Logits for {len(test_qids)} queries...")
    cached_data = {}
    target_layers = [0, 3, 6, 9, 12]
    
    for qid in tqdm(test_qids, desc="Caching"):
        qtext = test_queries[qid]
        retrieved_docs = baseline_results[qid]
        if not retrieved_docs:
            cached_data[qid] = None
            continue
            
        doc_ids = [d[0] for d in retrieved_docs]
        base_scores = torch.tensor([d[1] for d in retrieved_docs], dtype=torch.float32).to(device)
        dtexts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        # Extract features
        q_layer_dict = inspector.get_all_layer_reprs([qtext])
        d_layer_dict = inspector.get_all_layer_reprs(dtexts)
        
        q_reps = torch.stack([q_layer_dict[l] for l in target_layers], dim=1).to(device) # (1, 5, 128)
        d_reps = torch.stack([d_layer_dict[l] for l in target_layers], dim=1).to(device) # (K, 5, 128)
        q_reps = q_reps.expand(len(doc_ids), -1, -1)
        
        # Norm
        q_reps = apply_normalization(q_reps, "layer_norm")
        d_reps = apply_normalization(d_reps, "layer_norm")
        
        with torch.no_grad():
            router_logits = router(q_reps, d_reps).cpu() # Move to CPU for caching
            
        cached_data[qid] = {
            "doc_ids": doc_ids,
            "base_scores": base_scores.cpu(),
            "router_logits": router_logits
        }
    
    # 5. Sweep Gamma
    print(f"[{dataset_name}] 3/3: Sweeping Gamma values {gammas}...")
    sweep_results = []
    
    for g in gammas:
        intervened_results = {}
        for qid in test_qids:
            data = cached_data[qid]
            if data is None:
                intervened_results[qid] = []
                continue
                
            new_scores = data["base_scores"] + g * data["router_logits"]
            sorted_indices = torch.argsort(new_scores, descending=True)
            reranked_docs = [(data["doc_ids"][idx], new_scores[idx].item()) for idx in sorted_indices]
            intervened_results[qid] = reranked_docs
            
        metrics = {
            "gamma": g,
            "ndcg@10": ndcg_at_k(qrels, intervened_results, k=10),
            "mrr@10": mrr_at_k(qrels, intervened_results, k=10)
        }
        sweep_results.append(metrics)
        print(f"  Gamma={g:<5} | NDCG@10: {metrics['ndcg@10']:.4f} | MRR@10: {metrics['mrr@10']:.4f}")
        
    return sweep_results

if __name__ == "__main__":
    datasets = ["scifact", "nfcorpus", "scidocs"]
    gammas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0] # Baseline is gamma=0.0
    
    all_sweep_results = {}
    for ds in datasets:
        chkpt = os.path.join(PHASE_ROOT, f"outputs/ablation/abl2_inter_norm/fold_{ds}/training")
        if not os.path.exists(chkpt):
            print(f"Checkpoint not found for {ds}, skipping.")
            continue
            
        all_sweep_results[ds] = run_gamma_sweep(ds, chkpt, gammas, num_queries=50)
        
    out_dir = os.path.join(PHASE_ROOT, "outputs/intervention")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gamma_sweep_results.json"), "w") as f:
        json.dump(all_sweep_results, f, indent=4)
        
    print(f"\nGamma Sweep completed! Results saved to {out_dir}/gamma_sweep_results.json")
