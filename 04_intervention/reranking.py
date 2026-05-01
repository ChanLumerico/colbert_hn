import sys
import os
import torch
from typing import Dict, List, Tuple

# Import normalization from 03 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
label_mod = importlib.import_module("03_router_training.label_design")
apply_normalization = label_mod.apply_normalization

def rerank_with_router(queries: Dict[str, str], corpus: Dict, 
                       retrieved_results: Dict[str, List[Tuple[str, float]]], 
                       inspector, router_model, device: str = "cpu", 
                       batch_size: int = 32,
                       target_layers: List[int] = [0, 3, 6, 9, 12],
                       norm_type: str = "raw",
                       pbar_desc: str = "      Reranking queries") -> Dict[str, List[Tuple[str, float]]]:
    """
    Use the trained LayerRouter to re-rank the retrieved documents with Ablation support.
    """
    from tqdm.notebook import tqdm
    router_model.to(device)
    router_model.eval()
    
    new_results = {}
    
    for qid, results in tqdm(retrieved_results.items(), desc=pbar_desc, leave=False):
        if not results:
            new_results[qid] = []
            continue
            
        q_text = queries[qid]
        doc_ids = [r[0] for r in results]
        doc_texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        # 1. Extract multi-layer representations for Query
        with torch.no_grad():
            q_layer_dict = inspector.get_all_layer_reprs([q_text], pbar_desc=f"{pbar_desc}:Q")
            q_reps = torch.stack([q_layer_dict[l] for l in sorted(target_layers)], dim=1) # (1, L, 128)
            q_reps = apply_normalization(q_reps, norm_type)
            
        # 2. Extract multi-layer representations for Documents in batches
        all_doc_reps = []
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i+batch_size]
            with torch.no_grad():
                d_layer_dict = inspector.get_all_layer_reprs(batch_texts, pbar_desc=f"{pbar_desc}:D_batch")
                d_reps = torch.stack([d_layer_dict[l] for l in sorted(target_layers)], dim=1) # (B, L, 128)
                d_reps = apply_normalization(d_reps, norm_type)
                all_doc_reps.append(d_reps.cpu())
        
        all_doc_reps = torch.cat(all_doc_reps, dim=0) # (K, L, 128)
        
        # 3. Compute Router scores
        router_scores = []
        for i in range(0, len(all_doc_reps), batch_size):
            d_batch = all_doc_reps[i:i+batch_size].to(device)
            q_batch = q_reps.expand(d_batch.size(0), -1, -1).to(device)
            
            with torch.no_grad():
                logits = router_model(q_batch, d_batch)
                scores = torch.sigmoid(logits).cpu().tolist()
                router_scores.extend(scores)
                
        # 4. Re-sort documents based on router scores
        reranked = sorted(zip(doc_ids, router_scores), key=lambda x: x[1], reverse=True)
        new_results[qid] = reranked
        
    return new_results
