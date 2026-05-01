import torch
import numpy as np
from typing import List, Dict, Any
from shared.colbert_inspector import ColBERTInspector

def analyze_geometry(triplets: List[dict], corpus: Dict, queries: Dict, inspector: ColBERTInspector) -> Dict[int, dict]:
    """
    Analyze the geometry of representations across layers.
    Calculates mean cosine similarity and L2 distance for (Q, P) and (Q, HN).
    """
    if len(triplets) > 500:
        indices = np.random.choice(len(triplets), 500, replace=False)
        sampled_triplets = [triplets[i] for i in indices]
    else:
        sampled_triplets = triplets

    q_texts = [queries[t["query_id"]] for t in sampled_triplets]
    p_texts = [corpus[t["pos_id"]]["title"] + " " + corpus[t["pos_id"]]["text"] for t in sampled_triplets]
    n_texts = [corpus[t["hn_id"]]["title"] + " " + corpus[t["hn_id"]]["text"] for t in sampled_triplets]

    q_layer_reps = inspector.get_all_layer_reprs(q_texts)
    p_layer_reps = inspector.get_all_layer_reprs(p_texts)
    n_layer_reps = inspector.get_all_layer_reprs(n_texts)

    results = {}
    for layer in sorted(q_layer_reps.keys()):
        q_reps = q_layer_reps[layer]
        p_reps = p_layer_reps[layer]
        n_reps = n_layer_reps[layer]

        # Cosine similarity
        pos_sims = torch.sum(q_reps * p_reps, dim=-1).detach().cpu().numpy()
        neg_sims = torch.sum(q_reps * n_reps, dim=-1).detach().cpu().numpy()

        # L2 Distance
        pos_dists = torch.norm(q_reps - p_reps, p=2, dim=-1).detach().cpu().numpy()
        neg_dists = torch.norm(q_reps - n_reps, p=2, dim=-1).detach().cpu().numpy()

        results[layer] = {
            "mean_pos_sim": float(np.mean(pos_sims)),
            "mean_neg_sim": float(np.mean(neg_sims)),
            "mean_pos_dist": float(np.mean(pos_dists)),
            "mean_neg_dist": float(np.mean(neg_dists)),
            "sim_separation": float(np.mean(pos_sims - neg_sims)),
            "dist_separation": float(np.mean(neg_dists - pos_dists)) # Positive if HN is further
        }
        
    return results
