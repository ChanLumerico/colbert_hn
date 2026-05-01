import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Any
from shared.colbert_inspector import ColBERTInspector
from tqdm import tqdm

def compute_layer_auroc(triplets: List[dict], corpus: Dict, queries: Dict, inspector: ColBERTInspector) -> Dict[int, float]:
    """
    Compute AUROC for each layer using pooled representations.
    For each triplet (q, p, hn), we compare sim(q, p) and sim(q, hn).
    We treat (q, p) as positive class and (q, hn) as negative class or simply calculate
    how often sim(q, p) > sim(q, hn) across the dataset.
    """
    # Sample triplets for efficiency if needed
    if len(triplets) > 1000:
        indices = np.random.choice(len(triplets), 1000, replace=False)
        sampled_triplets = [triplets[i] for i in indices]
    else:
        sampled_triplets = triplets

    # Batch process to get all layer representations
    q_texts = [queries[t["query_id"]] for t in sampled_triplets]
    p_texts = [corpus[t["pos_id"]]["title"] + " " + corpus[t["pos_id"]]["text"] for t in sampled_triplets]
    n_texts = [corpus[t["hn_id"]]["title"] + " " + corpus[t["hn_id"]]["text"] for t in sampled_triplets]

    print(f"Extracting layer representations for {len(sampled_triplets)} triplets...")
    q_layer_reps = inspector.get_all_layer_reprs(q_texts)
    p_layer_reps = inspector.get_all_layer_reprs(p_texts)
    n_layer_reps = inspector.get_all_layer_reprs(n_texts)

    results = {}
    for layer in sorted(q_layer_reps.keys()):
        q_reps = q_layer_reps[layer] # (N, 128)
        p_reps = p_layer_reps[layer] # (N, 128)
        n_reps = n_layer_reps[layer] # (N, 128)

        # Compute cosine similarity (dot product since they are L2 normalized in encode)
        # Note: inspector.get_all_layer_reprs already performs normalization
        pos_sims = torch.sum(q_reps * p_reps, dim=-1).detach().cpu().numpy()
        neg_sims = torch.sum(q_reps * n_reps, dim=-1).detach().cpu().numpy()

        # AUROC calculation
        # Labels: 1 for positive pairs, 0 for negative pairs
        y_true = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
        y_scores = np.concatenate([pos_sims, neg_sims])
        
        auroc = roc_auc_score(y_true, y_scores)
        results[layer] = float(auroc)
        
    return results
