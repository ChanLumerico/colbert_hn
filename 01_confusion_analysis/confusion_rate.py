import os
from typing import List, Dict, Any
import numpy as np
from shared.data_utils import load_beir_dataset, build_triplets, save_json
from shared.colbert_inspector import ColBERTInspector

def compute_confusion_rate(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    triplet 리스트를 받아 쿼리별 confusion 여부를 계산한다.

    confusion 정의:
      margin = pos_score - hn_score
      margin < 0이면 해당 HN은 confused (HN이 positive보다 높게 랭킹됨)

    반환 dict 구조:
    {
      "global_confusion_rate": float,        # 전체 HN 쌍 기준
      "query_confusion_rate": float,         # confused 쿼리 비율 (1개 이상 confused HN 보유)
      "per_query": {
        query_id: {
          "n_total_hn": int,
          "n_confused": int,
          "confusion_rate": float,
          "margins": List[float],
          "worst_margin": float,
        }
      },
      "margin_distribution": List[float],   # 전체 margin 값 리스트
    }
    """
    per_query = {}
    all_margins = []
    total_hn_pairs = 0
    confused_hn_pairs = 0
    
    for trip in triplets:
        qid = trip["query_id"]
        pos_score = trip["pos_score"]
        hn_score = trip["hn_score"]
        margin = pos_score - hn_score
        
        if qid not in per_query:
            per_query[qid] = {
                "n_total_hn": 0,
                "n_confused": 0,
                "margins": [],
                "worst_margin": float('inf')
            }
            
        per_query[qid]["n_total_hn"] += 1
        per_query[qid]["margins"].append(margin)
        all_margins.append(margin)
        total_hn_pairs += 1
        
        if margin < 0:
            per_query[qid]["n_confused"] += 1
            confused_hn_pairs += 1
            
        if margin < per_query[qid]["worst_margin"]:
            per_query[qid]["worst_margin"] = margin
            
    # Post-process per_query
    n_confused_queries = 0
    for qid in per_query:
        stats = per_query[qid]
        stats["confusion_rate"] = stats["n_confused"] / stats["n_total_hn"]
        if stats["n_confused"] > 0:
            n_confused_queries += 1
            
    return {
        "global_confusion_rate": confused_hn_pairs / total_hn_pairs if total_hn_pairs > 0 else 0.0,
        "query_confusion_rate": n_confused_queries / len(per_query) if per_query else 0.0,
        "per_query": per_query,
        "margin_distribution": all_margins
    }

def compute_confusion_by_dataset(dataset_name: str, inspector: ColBERTInspector,
                                  top_k: int = 100) -> Dict[str, Any]:
    """
    단일 데이터셋에 대해 전체 파이프라인을 실행한다:
    1. 데이터 로드
    2. batch_retrieve로 전체 쿼리 검색
    3. build_triplets으로 triplet 구성
    4. compute_confusion_rate 실행
    5. 결과를 outputs/01_confusion_analysis/{dataset_name}/results.json으로 저장
    """
    print(f"\nProcessing dataset: {dataset_name}")
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    
    retrieved_results = inspector.batch_retrieve(queries, corpus, top_k=top_k)
    
    triplets = build_triplets(queries, qrels, retrieved_results, k=top_k)
    
    results = compute_confusion_rate(triplets)
    
    # Save results
    output_path = f"outputs/01_confusion_analysis/{dataset_name}/results.json"
    save_json(results, output_path)
    
    return results

def summarize_across_datasets(results_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    3개 데이터셋 결과를 받아 비교 요약 테이블을 반환한다.
    """
    summary = {}
    for ds_name, res in results_map.items():
        margins = res["margin_distribution"]
        summary[ds_name] = {
            "global_confusion_rate": res["global_confusion_rate"],
            "query_confusion_rate": res["query_confusion_rate"],
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
            "std_margin": float(np.std(margins)) if margins else 0.0,
            "n_queries": len(res["per_query"]),
            "n_triplets": len(margins)
        }
    return summary
