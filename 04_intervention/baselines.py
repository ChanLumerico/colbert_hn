from shared.metrics import ndcg_at_k, mrr_at_k, recall_at_k
from typing import Dict, List

def compute_all_metrics(qrels: Dict, results: Dict, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute multiple IR metrics for a given set of results.
    """
    metrics = {}
    for k in k_list:
        metrics[f"NDCG@{k}"] = ndcg_at_k(qrels, results, k=k)
        metrics[f"MRR@{k}"] = mrr_at_k(qrels, results, k=k)
        metrics[f"Recall@{k}"] = recall_at_k(qrels, results, k=k)
    return metrics
