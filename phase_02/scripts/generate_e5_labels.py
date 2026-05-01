import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Resolve paths
PHASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PHASE_01_PATH = os.path.join(PHASE_ROOT, "phase_01")
if PHASE_01_PATH not in sys.path:
    sys.path.append(PHASE_01_PATH)

from shared.data_utils import load_beir_dataset, build_triplets
from shared.colbert_inspector import ColBERTInspector

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(query: str) -> str:
    return f'Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {query}'

class E5LabelGenerator:
    def __init__(self, model_id='intfloat/e5-mistral-7b-instruct', device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embeddings(self, texts: list, is_query: bool = False):
        if is_query:
            processed_texts = [get_detailed_instruct(t) for t in texts]
        else:
            processed_texts = texts
            
        batch_dict = self.tokenizer(processed_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    def process_dataset(self, dataset_name, max_queries=100):
        corpus, queries, qrels = load_beir_dataset(dataset_name)
        inspector = ColBERTInspector(device=self.device)
        
        # 1. Get triplets (Same as Phase 1)
        qids = list(queries.keys())[:max_queries]
        sample_queries = {qid: queries[qid] for qid in qids}
        retrieved = inspector.batch_retrieve(sample_queries, corpus, top_k=50)
        triplets = build_triplets(sample_queries, qrels, retrieved, k=50)
        
        print(f"Generating labels for {len(triplets)} triplets in {dataset_name}...")
        
        labeled_triplets = []
        for trip in tqdm(triplets):
            q_text = queries[trip['query_id']]
            p_text = corpus[trip['pos_id']]['title'] + " " + corpus[trip['pos_id']]['text']
            n_text = corpus[trip['hn_id']]['title'] + " " + corpus[trip['hn_id']]['text']
            
            # Get E5 Scores
            q_emb = self.get_embeddings([q_text], is_query=True)
            p_emb = self.get_embeddings([p_text], is_query=False)
            n_emb = self.get_embeddings([n_text], is_query=False)
            
            p_score = torch.matmul(q_emb, p_emb.T).item()
            n_score = torch.matmul(q_emb, n_emb.T).item()
            
            trip['e5_pos_score'] = p_score
            trip['e5_hn_score'] = n_score
            trip['e5_margin'] = p_score - n_score
            
            labeled_triplets.append(trip)
            
        return labeled_triplets

if __name__ == "__main__":
    generator = E5LabelGenerator()
    datasets = ["scifact", "nfcorpus", "scidocs"]
    
    all_labels = {}
    for ds in datasets:
        labels = generator.process_dataset(ds, max_queries=50) # Sample for testing
        all_labels[ds] = labels
        
    out_dir = os.path.join(PHASE_ROOT, "phase_02/data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e5_soft_labels.json"), "w") as f:
        json.dump(all_labels, f, indent=4)
        
    print(f"E5 Soft Labels saved to {out_dir}/e5_soft_labels.json")
