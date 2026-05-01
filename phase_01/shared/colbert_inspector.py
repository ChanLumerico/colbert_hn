import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertPreTrainedModel, BertModel
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import os
import yaml

class ColBERTInspector:
    """
    ColBERT 모델 래퍼. 추론 전용 (학습 없음).
    """
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # ColBERT v2.0 is based on BERT with a linear layer
        class ColBERTConfig(BertPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.bert = BertModel(config)
                self.linear = nn.Linear(config.hidden_size, 128, bias=False)
                self.post_init()
        
        hf_model = ColBERTConfig.from_pretrained(model_name)
        self.model = hf_model.bert
        self.linear = hf_model.linear
        
        self.model.to(self.device)
        self.linear.to(self.device)
        self.model.eval()
        self.linear.eval()
        
        # Load target layers from config if possible, else default
        self.target_layers = [0, 3, 6, 9, 12]
        
    def encode(self, texts: List[str], layer: int = 12) -> torch.Tensor:
        """
        텍스트 리스트를 인코딩하여 지정 레이어의 mean-pooled 표현을 반환한다.
        반환 shape: (len(texts), 128)
        layer: 0부터 12까지. 0이면 embedding layer 출력.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            # hidden_states contains (layer 0, layer 1, ..., layer 12)
            hidden_state = outputs.hidden_states[layer]
            
            # Apply ColBERT's linear projection
            projected = self.linear(hidden_state)
            # ColBERT uses L2 normalization
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
            
            # Mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            
        return pooled

    def get_all_layer_reprs(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        """
        모든 레이어(0, 3, 6, 9, 12)의 표현을 한 번에 반환한다.
        반환: {layer_idx: tensor of shape (len(texts), 128)}
        분석 레이어는 config.yaml의 target_layers에서 읽는다.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt"
        ).to(self.device)
        
        results = {}
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            
            for layer in self.target_layers:
                hidden_state = outputs.hidden_states[layer]
                projected = self.linear(hidden_state)
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
                pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                results[layer] = pooled
                
        return results

    def retrieve(self, query: str, corpus: Dict, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        단일 쿼리에 대해 corpus에서 top-K 문서를 검색한다.
        ColBERT MaxSim 스코어 기준으로 정렬하여 반환한다.
        반환: [(doc_id, score), ...]
        """
        # Small scale retrieval for analysis
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        # Batch retrieve with batch size 1 for simplicity here, but calling internal batch logic
        results = self.batch_retrieve({ "q1": query }, corpus, top_k=top_k)
        return results["q1"]

    def batch_retrieve(self, queries: Dict[str, str], corpus: Dict, top_k: int = 100,
                       batch_size: int = 32) -> Dict[str, List[Tuple[str, float]]]:
        """
        전체 쿼리에 대해 batch로 검색 결과를 반환한다.
        반환: {query_id: [(doc_id, score), ...]}
        진행상황을 tqdm으로 출력한다.
        """
        # Pre-encode all documents to save time
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        
        print(f"Pre-encoding {len(doc_texts)} documents...")
        all_doc_embs = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding Corpus"):
            batch_docs = doc_texts[i : i + batch_size]
            encoded = self.tokenizer(batch_docs, padding=True, truncation=True, max_length=150, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                projected = self.linear(outputs.last_hidden_state)
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
                all_doc_embs.append(projected.cpu()) # Store in CPU to save GPU memory
        
        results = {}
        query_ids = list(queries.keys())
        print(f"Retrieving for {len(query_ids)} queries...")
        for i in tqdm(range(len(query_ids)), desc="Retrieving"):
            qid = query_ids[i]
            query_text = queries[qid]
            
            # Encode query
            encoded_q = self.tokenizer([query_text], padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs_q = self.model(**encoded_q)
                projected_q = self.linear(outputs_q.last_hidden_state)
                projected_q = torch.nn.functional.normalize(projected_q, p=2, dim=-1)
                q_emb = projected_q[0] # (seq_len_q, 128)
                q_mask = encoded_q["attention_mask"][0].bool()
                q_emb = q_emb[q_mask] # Only keep non-padding tokens
            
            # Compute MaxSim against all docs
            q_scores = []
            doc_idx = 0
            for doc_batch in all_doc_embs:
                # doc_batch: (B, seq_len_d, 128)
                # q_emb: (seq_len_q_active, 128)
                # Compute dot product: (B, seq_len_q_active, seq_len_d)
                scores = torch.matmul(q_emb, doc_batch.transpose(1, 2)) 
                # MaxSim: max over doc tokens, sum over query tokens
                max_scores = scores.max(dim=-1).values # (B, seq_len_q_active)
                sum_scores = max_scores.sum(dim=-1) # (B)
                
                for j in range(len(sum_scores)):
                    q_scores.append((doc_ids[doc_idx + j], sum_scores[j].item()))
                doc_idx += len(sum_scores)
            
            q_scores.sort(key=lambda x: x[1], reverse=True)
            results[qid] = q_scores[:top_k]
            
        return results
