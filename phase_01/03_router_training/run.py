import sys
import os
import torch
import yaml

# Resolve path for shared module
PHASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PHASE_ROOT not in sys.path:
    sys.path.append(PHASE_ROOT)

from shared.data_utils import load_beir_dataset, build_triplets, save_json
from shared.colbert_inspector import ColBERTInspector
from router_model import LayerRouter
from label_design import prepare_router_data, RouterDataset
from train import train_router

def main():
    # 1. Load config
    config_path = os.path.join(PHASE_ROOT, "config.yaml")
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
    
    # 2. Initialize Inspector
    inspector = ColBERTInspector(model_name=model_name, device=device)
    
    # 3. Collect and Prepare Data
    all_q_reps = []
    all_d_reps = []
    all_labels = []
    
    for ds in datasets:
        print(f"\n--- Preparing training data from {ds} ---")
        corpus, queries, qrels = load_beir_dataset(ds, split="test")
        
        sample_query_ids = list(queries.keys())[:100]
        sample_queries = {qid: queries[qid] for qid in sample_query_ids}
        
        retrieved = inspector.batch_retrieve(sample_queries, corpus, top_k=50)
        triplets = build_triplets(sample_queries, qrels, retrieved, k=50)
        
        if not triplets: continue
            
        q_reps, d_reps, labels = prepare_router_data(
            triplets, corpus, queries, inspector, 
            max_samples=1000, 
            target_layers=target_layers, 
            norm_type=norm_type
        )
        all_q_reps.append(q_reps)
        all_d_reps.append(d_reps)
        all_labels.append(labels)
        
    full_q_reps = torch.cat(all_q_reps, dim=0)
    full_d_reps = torch.cat(all_d_reps, dim=0)
    full_labels = torch.cat(all_labels, dim=0)
    
    dataset = RouterDataset(full_q_reps, full_d_reps, full_labels)
    print(f"\nTotal training samples: {len(dataset)}")
    
    # 4. Initialize and Train Flexible Router
    model = LayerRouter(
        num_layers=len(target_layers), 
        embed_dim=128, 
        hidden_dims=hidden_dims, 
        fusion_type=fusion_type
    ).to(device)
    
    history = train_router(model, dataset, epochs=20, batch_size=64, lr=1e-3, device=device)
    
    # 5. Save Results (Hierarchical)
    out_dir = os.path.join(PHASE_ROOT, f"outputs/03_router_training/{exp_id}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "router_model.pt"))
    save_json(history, os.path.join(out_dir, "history.json"))
    
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
