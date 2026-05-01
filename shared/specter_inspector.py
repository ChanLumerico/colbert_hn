import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from tqdm import tqdm

class SPECTERInspector:
    """
    Wrapper for SPECTER v2 (Scientific Document Embeddings).
    Inference only.
    """
    def __init__(self, model_name: str = "allenai/specter2_base", device: str = "mps"):
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into normalized CLS embeddings."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # SPECTER uses the CLS token (index 0)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
        return embeddings

    def batch_retrieve(self, queries: Dict[str, str], corpus: Dict, top_k: int = 100, 
                       batch_size: int = 32) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch retrieve top-K documents using Cosine Similarity between SPECTER embeddings.
        """
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        print("Pre-encoding corpus with SPECTER...")
        all_d_embs = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding docs"):
            d_embs = self.encode(doc_texts[i:i+batch_size])
            all_d_embs.append(d_embs.cpu())
        
        all_d_embs = torch.cat(all_d_embs, dim=0) # (Num_docs, dim)
        
        print("Computing Cosine Similarity for queries...")
        out_results = {}
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        for i in tqdm(range(0, len(query_ids), batch_size), desc="Querying"):
            batch_q_ids = query_ids[i:i+batch_size]
            batch_q_texts = query_texts[i:i+batch_size]
            
            q_embs = self.encode(batch_q_texts).cpu() # (B, dim)
            
            # Cosine similarity (dot product of normalized embeddings)
            scores = torch.matmul(q_embs, all_d_embs.T) # (B, Num_docs)
            
            for j, qid in enumerate(batch_q_ids):
                q_scores = scores[j].tolist()
                q_results = sorted(zip(doc_ids, q_scores), key=lambda x: x[1], reverse=True)
                out_results[qid] = q_results[:top_k]
                
        return out_results
