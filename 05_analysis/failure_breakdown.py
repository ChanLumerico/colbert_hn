from typing import Dict, List, Tuple

def categorize_queries(qrels: Dict, vanilla_results: Dict, intervention_results: Dict) -> Dict[str, List[str]]:
    """
    Categorize queries into Rescued, Still Confused, and Harmful.
    """
    categories = {
        "rescued": [],       # Vanilla Top-1 is wrong, Intervention Top-1 is correct
        "still_confused": [], # Both Top-1 are wrong
        "harmful": [],        # Vanilla Top-1 is correct, Intervention Top-1 is wrong
        "always_correct": []  # Both Top-1 are correct
    }
    
    for qid in vanilla_results:
        if qid not in qrels or qid not in intervention_results:
            continue
            
        gt_ids = list(qrels[qid].keys())
        
        v_top1 = vanilla_results[qid][0][0] if vanilla_results[qid] else None
        i_top1 = intervention_results[qid][0][0] if intervention_results[qid] else None
        
        v_is_correct = v_top1 in gt_ids
        i_is_correct = i_top1 in gt_ids
        
        if not v_is_correct and i_is_correct:
            categories["rescued"].append(qid)
        elif not v_is_correct and not i_is_correct:
            categories["still_confused"].append(qid)
        elif v_is_correct and not i_is_correct:
            categories["harmful"].append(qid)
        else:
            categories["always_correct"].append(qid)
            
    return categories

def get_query_details(qid: str, queries: Dict, corpus: Dict, qrels: Dict, 
                      vanilla_res: List[Tuple], intervention_res: List[Tuple], top_n: int = 3) -> Dict:
    """
    Get detailed text for a specific query and its top results.
    """
    gt_ids = list(qrels[qid].keys())
    
    details = {
        "query": queries[qid],
        "vanilla_top": [],
        "intervention_top": [],
        "ground_truth_samples": [corpus[gid]["title"] + " " + corpus[gid]["text"] for gid in gt_ids[:1]]
    }
    
    for rid, score in vanilla_res[:top_n]:
        details["vanilla_top"].append({
            "id": rid,
            "text": corpus[rid]["title"] + " " + corpus[rid]["text"],
            "score": score,
            "is_correct": rid in gt_ids
        })
        
    for rid, score in intervention_res[:top_n]:
        details["intervention_top"].append({
            "id": rid,
            "text": corpus[rid]["title"] + " " + corpus[rid]["text"],
            "score": score,
            "is_correct": rid in gt_ids
        })
        
    return details
