import os
import sys
import time
import torch
import pandas as pd
from pathlib import Path
import argparse

from benchmarks.config import BASE_CONFIG, ensure_dirs, set_global_seed, CACHE_DIR, result_subdir
from benchmarks.dataset_registry import DATASET_REGISTRY
from benchmarks.data_utils import load_or_preprocess_adata
from utils.data import DataSplitter
from benchmarks.train_utils import train_and_evaluate

def standardize_labels(adata, label_key):
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    elif "cell_type" not in adata.obs.columns:
        print(f"  WARNING: Neither '{label_key}' nor 'cell_type' found in obs.")
    return adata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=["setty", "endo", "dentate"])
    parser.add_argument("--models", type=str, nargs="+", default=["DPMM-Base", "DPMM-Contrastive"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(BASE_CONFIG.device)
    
    out_dirs = {
        "csv": result_subdir("warmup", "csv"),
        "cache": CACHE_DIR,
    }
    ensure_dirs(*out_dirs.values())
    results = []
    
    for ds_key in args.datasets:
        ds_info = DATASET_REGISTRY[ds_key]
        
        print(f"\n=== Warmup Ablation: {ds_key} ===")
        
        # Load data
        adata = load_or_preprocess_adata(
            ds_info["path"], max_cells=BASE_CONFIG.max_cells, hvg_top_genes=BASE_CONFIG.hvg_top_genes,
            seed=args.seed, cache_dir=str(out_dirs["cache"]), use_cache=True
        )
        adata = standardize_labels(adata, ds_info["label_key"])
        
        splitter = DataSplitter(
            adata=adata, layer="counts",
            train_size=0.7, val_size=0.15, test_size=0.15,
            batch_size=BASE_CONFIG.batch_size, latent_dim=BASE_CONFIG.latent_dim
        )
        
        for model_name in args.models:
            for warmup_ratio in [0.0, 0.3]:
                print(f"  Model: {model_name}, Warmup Ratio: {warmup_ratio}")
                from benchmarks.model_registry import MODELS
                model_info = MODELS[model_name]
                
                start_time = time.time()
                
                # Override warmup_ratio in params
                params = model_info["params"].copy()
                params["dpmm_warmup_ratio"] = warmup_ratio
                
                res = train_and_evaluate(
                    name=f"{model_name}_warmup{warmup_ratio}",
                    model_cls=model_info["class"],
                    params=params,
                    splitter=splitter,
                    device=device,
                    lr=BASE_CONFIG.lr,
                    epochs=500,
                    data_type=ds_info["data_type"]
                )
                
                end_time = time.time()
                train_time = end_time - start_time
                
                res["Dataset"] = ds_key
                res["WarmupRatio"] = warmup_ratio
                res["Model"] = model_name
                res["TrainTime"] = train_time
                res["Score"] = (res.get("NMI", 0) + res.get("ARI", 0) + res.get("ASW", 0)) / 3
                
                # Remove large objects
                res.pop("latent", None)
                res.pop("history", None)
                res.pop("model_obj", None)
                
                results.append(res)
                
                torch.cuda.empty_cache()
                
    df = pd.DataFrame(results)
    csv_path = out_dirs["csv"] / f"warmup_ablation_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved warmup ablation results to {csv_path}")

if __name__ == "__main__":
    main()
