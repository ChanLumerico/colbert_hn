import sys
import os
import torch
import yaml
import importlib
from typing import List, Dict

# Resolve paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from shared.data_utils import load_beir_dataset, save_json, build_triplets
from shared.colbert_inspector import ColBERTInspector

# Dynamic imports
router_mod = importlib.import_module("03_router_training.router_model")
LayerRouter = router_mod.LayerRouter

label_mod = importlib.import_module("03_router_training.label_design")
prepare_router_data = label_mod.prepare_router_data
RouterDataset = label_mod.RouterDataset

train_mod = importlib.import_module("03_router_training.train")
train_router = train_mod.train_router

rerank_mod = importlib.import_module("04_intervention.reranking")
rerank_with_router = rerank_mod.rerank_with_router

baseline_mod = importlib.import_module("04_intervention.baselines")
compute_all_metrics = baseline_mod.compute_all_metrics


def main():
    # 1. Load config
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets = config["datasets"]
    model_name = config["model"]["colbert"]
    device = config["model"]["device"]

    inspector = ColBERTInspector(model_name=model_name, device=device)
    cross_val_results = {}

    # 2. Iterate each dataset as a test set (Leave-One-Out)
    for test_ds in datasets:
        train_ds_list = [d for d in datasets if d != test_ds]
        print(f"\n" + "=" * 50)
        print(f"FOLD: Testing on {test_ds} (Unseen)")
        print(f"Training on: {train_ds_list}")
        print("=" * 50)

        # A. Prepare Training Data (Build on the fly from other datasets)
        all_q_reps = []
        all_d_reps = []
        all_labels = []

        for ds in train_ds_list:
            print(f"\n--- Harvesting training data from {ds} ---")
            corpus, queries, qrels = load_beir_dataset(ds, split="test")

            # Sample queries to build training triplets
            sample_query_ids = list(queries.keys())[:100]
            sample_queries = {qid: queries[qid] for qid in sample_query_ids}

            print(f"Retrieving for {len(sample_queries)} queries in {ds}...")
            retrieved = inspector.batch_retrieve(sample_queries, corpus, top_k=50)
            triplets = build_triplets(sample_queries, qrels, retrieved, k=50)

            if triplets:
                q_reps, d_reps, labels = prepare_router_data(triplets, corpus, queries, inspector, max_samples=1000)
                all_q_reps.append(q_reps)
                all_d_reps.append(d_reps)
                all_labels.append(labels)

        # Build features for training
        full_q_reps = torch.cat(all_q_reps, dim=0)
        full_d_reps = torch.cat(all_d_reps, dim=0)
        full_labels = torch.cat(all_labels, dim=0)
        train_dataset = RouterDataset(full_q_reps, full_d_reps, full_labels)

        # B. Train Router
        print(f"Training Router for Fold {test_ds} (Total Samples: {len(train_dataset)})...")
        router = LayerRouter(num_layers=5, embed_dim=128).to(device)
        train_router(router, train_dataset, epochs=10, lr=1e-3, device=device)

        # C. Evaluate on Test Dataset (Zero-shot)
        print(f"Evaluating Zero-shot on {test_ds}...")
        corpus, queries, qrels = load_beir_dataset(test_ds, split="test")

        # Sample queries for evaluation
        test_query_ids = list(queries.keys())[:50]
        test_queries = {qid: queries[qid] for qid in test_query_ids}

        # Vanilla
        vanilla_results = inspector.batch_retrieve(test_queries, corpus, top_k=50)
        vanilla_metrics = compute_all_metrics(qrels, vanilla_results)

        # Intervention
        intervention_results = rerank_with_router(
            test_queries, corpus, vanilla_results, inspector, router, device=device
        )
        intervention_metrics = compute_all_metrics(qrels, intervention_results)

        cross_val_results[test_ds] = {
            "vanilla": vanilla_metrics,
            "zero_shot_intervention": intervention_metrics,
            "improvement": {
                m: intervention_metrics[m] - vanilla_metrics[m] for m in vanilla_metrics
            },
        }

        print(f"\nZero-shot Result for {test_ds}:")
        print(
            f"  NDCG@10 Improvement: {cross_val_results[test_ds]['improvement']['NDCG@10']:+.4f}"
        )

    # 3. Save Report
    os.makedirs("outputs/06_cross_validation", exist_ok=True)
    save_json(cross_val_results, "outputs/06_cross_validation/report.json")

    print("\n" + "!" * 80)
    print("CROSS-DATASET VALIDATION COMPLETE!")
    print("Results saved to outputs/06_cross_validation/report.json")
    print("!" * 80 + "\n")


if __name__ == "__main__":
    main()
