import numpy as np
from typing import List, Dict, Tuple

def pairwise_accuracy(y_true: List[float], y_pred: List[float],
                      query_ids: List[str]) -> float:
    """
    같은 쿼리 내 쌍(i, j)에서 y_true[i] > y_true[j]일 때
    y_pred[i] > y_pred[j]인 비율을 반환한다.
    PairAcc = |correct pairs| / |total pairs|
    """
    correct = 0
    total = 0
    
    # Group by query_id
    q_data = {}
    for i in range(len(query_ids)):
        qid = query_ids[i]
        if qid not in q_data:
            q_data[qid] = []
        q_data[qid].append((y_true[i], y_pred[i]))
        
    for qid, items in q_data.items():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                t1, p1 = items[i]
                t2, p2 = items[j]
                
                if t1 == t2:
                    continue
                
                total += 1
                if (t1 > t2 and p1 > p2) or (t1 < t2 and p1 < p2):
                    correct += 1
                    
    return correct / total if total > 0 else 0.0

def ndcg_at_k(qrels: Dict[str, Dict[str, int]], 
              results: Dict[str, List[Tuple[str, float]]], 
              k: int = 10) -> float:
    """
    표준 NDCG@K를 계산한다.
    """
    scores = []
    for qid, res in results.items():
        if qid not in qrels:
            continue
            
        # DCG
        dcg = 0.0
        for i, (doc_id, score) in enumerate(res[:k]):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2**rel - 1) / np.log2(i + 2)
            
        # IDCG
        idcg = 0.0
        rel_scores = sorted(qrels[qid].values(), reverse=True)
        for i, rel in enumerate(rel_scores[:k]):
            idcg += (2**rel - 1) / np.log2(i + 2)
            
        if idcg > 0:
            scores.append(dcg / idcg)
        else:
            scores.append(0.0)
            
    return np.mean(scores) if scores else 0.0

def mrr_at_k(qrels: Dict[str, Dict[str, int]], 
             results: Dict[str, List[Tuple[str, float]]], 
             k: int = 10) -> float:
    """MRR@K를 계산한다."""
    scores = []
    for qid, res in results.items():
        if qid not in qrels:
            continue
            
        found = False
        for i, (doc_id, score) in enumerate(res[:k]):
            if qrels[qid].get(doc_id, 0) > 0:
                scores.append(1.0 / (i + 1))
                found = True
                break
        if not found:
            scores.append(0.0)
            
    return np.mean(scores) if scores else 0.0

def recall_at_k(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, List[Tuple[str, float]]], 
                k: int) -> float:
    """Recall@K를 계산한다."""
    scores = []
    for qid, res in results.items():
        if qid not in qrels:
            continue
            
        relevant_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        if not relevant_docs:
            continue
            
        retrieved_at_k = {doc_id for doc_id, score in res[:k]}
        hits = len(relevant_docs.intersection(retrieved_at_k))
        scores.append(hits / len(relevant_docs))
        
    return np.mean(scores) if scores else 0.0
