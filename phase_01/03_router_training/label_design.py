import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple
from shared.colbert_inspector import ColBERTInspector

class RouterDataset(Dataset):
    def __init__(self, q_reps: torch.Tensor, d_reps: torch.Tensor, labels: torch.Tensor):
        self.q_reps = q_reps
        self.d_reps = d_reps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.q_reps[idx], self.d_reps[idx], self.labels[idx]

def apply_normalization(reps, norm_type="raw"):
    """ [Ablation 5] Apply normalization """
    if norm_type == "raw":
        return reps
    elif norm_type == "l2":
        return torch.nn.functional.normalize(reps, p=2, dim=-1)
    elif norm_type == "layer_norm":
        means = reps.mean(dim=-1, keepdim=True)
        stds = reps.std(dim=-1, keepdim=True) + 1e-6
        return (reps - means) / stds
    return reps

def prepare_router_data(triplets: List[dict], corpus: Dict, queries: Dict, 
                        inspector: ColBERTInspector, 
                        max_samples: int = 2000,
                        target_layers: List[int] = [0, 3, 6, 9, 12],
                        norm_type: str = "raw",
                        pbar_desc: str = "      Extracting Features") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare (Query, Doc, Label) training pairs with Ablation support.
    """
    from tqdm.notebook import tqdm
    
    if len(triplets) > max_samples:
        indices = np.random.choice(len(triplets), max_samples, replace=False)
        triplets = [triplets[i] for i in indices]

    q_ids = [t["query_id"] for t in triplets]
    p_ids = [t["pos_id"] for t in triplets]
    n_ids = [t["hn_id"] for t in triplets]

    q_texts = [queries[qid] for qid in q_ids]
    p_texts = [corpus[pid]["title"] + " " + corpus[pid]["text"] for pid in p_ids]
    n_texts = [corpus[nid]["title"] + " " + corpus[nid]["text"] for nid in n_ids]

    # [Ablation 1] Select target layers during extraction
    q_layer_dict = inspector.get_all_layer_reprs(q_texts, pbar_desc=f"{pbar_desc}:Q")
    p_layer_dict = inspector.get_all_layer_reprs(p_texts, pbar_desc=f"{pbar_desc}:P")
    n_layer_dict = inspector.get_all_layer_reprs(n_texts, pbar_desc=f"{pbar_desc}:N")

    # Filter only target layers
    q_reps_all = torch.stack([q_layer_dict[l] for l in sorted(target_layers)], dim=1).cpu()
    p_reps_all = torch.stack([p_layer_dict[l] for l in sorted(target_layers)], dim=1).cpu()
    n_reps_all = torch.stack([n_layer_dict[l] for l in sorted(target_layers)], dim=1).cpu()

    # [Ablation 5] Apply normalization
    q_reps_all = apply_normalization(q_reps_all, norm_type)
    p_reps_all = apply_normalization(p_reps_all, norm_type)
    n_reps_all = apply_normalization(n_reps_all, norm_type)

    # Combine Positives and Negatives
    final_q_reps = torch.cat([q_reps_all, q_reps_all], dim=0)
    final_d_reps = torch.cat([p_reps_all, n_reps_all], dim=0)
    final_labels = torch.cat([torch.ones(len(q_reps_all)), torch.zeros(len(n_reps_all))], dim=0)

    return final_q_reps, final_d_reps, final_labels
