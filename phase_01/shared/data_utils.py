import os
import json
import logging
from typing import Dict, List, Tuple, Any
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)

def load_beir_dataset(dataset_name: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    """
    beir 라이브러리를 사용하여 corpus, queries, qrels를 로드한다.
    dataset_name: "scifact" | "nfcorpus" | "scidocs"
    반환: (corpus, queries, qrels)
    corpus: Dict[doc_id, {"title": str, "text": str}]
    queries: Dict[query_id, str]
    qrels: Dict[query_id, Dict[doc_id, int]]
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    # Get true project root (parent of phase_01/)
    shared_dir = os.path.dirname(os.path.abspath(__file__))
    # shared -> phase_01 -> root
    project_root = os.path.dirname(os.path.dirname(shared_dir))
    cache_dir = os.path.join(project_root, "data")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    data_path = util.download_and_unzip(url, cache_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    return corpus, queries, qrels

def build_triplets(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], 
                   retrieved_results: Dict[str, List[Tuple[str, float]]], 
                   k: int = 100) -> List[Dict[str, Any]]:
    """
    쿼리별로 (query_id, pos_id, hn_id, pos_score, hn_score) triplet을 구성한다.
    retrieved_results: Dict[query_id, List[Tuple[doc_id, score]]] — ColBERT top-K 결과
    qrels에서 relevance > 0인 문서를 positive로 정의한다.
    top-K 내에서 positive가 아닌 문서를 HN candidate으로 정의한다.
    positive가 retrieved 결과에 없는 쿼리는 건너뛴다.
    """
    triplets = []
    skipped_no_pos = 0
    
    for qid, results in retrieved_results.items():
        if qid not in qrels:
            continue
            
        pos_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        top_k_results = results[:k]
        
        retrieved_positives = [(doc_id, score) for doc_id, score in top_k_results if doc_id in pos_docs]
        
        if not retrieved_positives:
            skipped_no_pos += 1
            continue
            
        retrieved_hns = [(doc_id, score) for doc_id, score in top_k_results if doc_id not in pos_docs]
        
        for pos_id, pos_score in retrieved_positives:
            for hn_id, hn_score in retrieved_hns:
                triplets.append({
                    "query_id": qid,
                    "pos_id": pos_id,
                    "hn_id": hn_id,
                    "pos_score": pos_score,
                    "hn_score": hn_score
                })
                
    if skipped_no_pos > 0:
        print(f"Warning: Skipped {skipped_no_pos} queries because no positive document was found in top-{k} results.")
        
    return triplets

def save_json(data: Any, path: str):
    """결과를 JSON으로 저장. 경로가 없으면 자동 생성."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    """JSON 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
