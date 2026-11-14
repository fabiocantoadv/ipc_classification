#!/usr/bin/env python3
"""Compara embeddings de patentes na mesma divisão da linha de base TF-IDF."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from baseline import evaluate, labels_for, load_rows, select_threshold, split_indices


SOURCES = {
    "gemma_ta": "patentes_xml_vec_gemma2_ta.npy",
    "gemma_claims": "patentes_xml_vec_gemma2_claims.npy",
    "qwen_ta": "patentes_xml_vec_qwen3_06_ta.npy",
    "qwen_tac": "patentes_xml_vec_qwen3_06_tac.npy",
}
VIEWS = ("gemma_ta", "gemma_claims", "gemma_mean", "qwen_ta", "qwen_tac", "qwen_tac_routed")


def get_embeddings(view: str, folder: Path, indices: list[int]) -> np.ndarray:
    if view == "qwen_tac_routed":
        ta = np.load(folder / SOURCES["qwen_ta"], mmap_mode="r")
        tac = np.load(folder / SOURCES["qwen_tac"], mmap_mode="r")
        lengths = np.load(folder / "qwen_tac_token_lengths.npy", mmap_mode="r")
        if ta.shape != tac.shape or len(lengths) != len(ta):
            raise ValueError("Qwen TA, TAC e contagem de tokens não estão alinhados")
        use_ta = lengths[indices] > 4096
        return np.where(use_ta[:, None], ta[indices], tac[indices]).astype(np.float32)
    if view == "gemma_mean":
        ta = np.load(folder / SOURCES["gemma_ta"], mmap_mode="r")
        claims = np.load(folder / SOURCES["gemma_claims"], mmap_mode="r")
        if ta.shape != claims.shape:
            raise ValueError("Matrizes Gemma2 TA e Claims com formatos diferentes")
        return (np.asarray(ta[indices], dtype=np.float32) + np.asarray(claims[indices], dtype=np.float32)) * 0.5
    return np.asarray(np.load(folder / SOURCES[view], mmap_mode="r")[indices], dtype=np.float32)


def prepare_targets(rows: list[dict[str, str]], splits: dict[str, np.ndarray],
                    level: str, min_support: int):
    label_sets = [labels_for(row, level) for row in rows]
    counts = Counter(label for i in splits["train"] for label in label_sets[i])
    classes = sorted(label for label, count in counts.items() if count >= min_support)
    known = set(classes)
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    y = {part: mlb.transform([label_sets[i] & known for i in indices]).astype(bool)
         for part, indices in splits.items()}
    total_true = {part: np.asarray([len(label_sets[i]) for i in indices], dtype=np.int32)
                  for part, indices in splits.items()}
    coverage = {part: {
        "documents": len(indices),
        "label_assignments": int(total_true[part].sum()),
        "unmodeled_label_assignments": int(sum(len(label_sets[i] - known) for i in indices)),
    } for part, indices in splits.items()}
    return classes, y, total_true, coverage


def ridge_components(X: np.ndarray, Y: np.ndarray):
    n = X.shape[0]
    X64 = np.asarray(X, dtype=np.float64)
    x_mean = X64.mean(axis=0)
    y_mean = Y.mean(axis=0, dtype=np.float64)
    gram = X64.T @ X64 - n * np.outer(x_mean, x_mean)
    rhs = np.asarray((csr_matrix(Y.astype(np.float32)).T @ X64).T, dtype=np.float64)
    rhs -= n * np.outer(x_mean, y_mean)
    return gram, rhs, x_mean, y_mean


def fit_one_view(X: np.ndarray, splits: dict[str, np.ndarray], Y: dict[str, np.ndarray],
                 total_true: dict[str, np.ndarray], alphas: list[float]):
    X_train, X_val, X_test = (X[splits[part]] for part in ("train", "validation", "test"))
    gram, rhs, x_mean, y_mean = ridge_components(X_train, Y["train"])
    identity = np.eye(gram.shape[0], dtype=np.float64)
    best = None
    for alpha in alphas:
        factor = cho_factor(gram + alpha * identity, overwrite_a=True, check_finite=False)
        coef = cho_solve(factor, rhs, check_finite=False)
        val_scores = (X_val - x_mean) @ coef + y_mean
        threshold, val_f1 = select_threshold(val_scores, Y["validation"])
        if best is None or val_f1 > best["validation_micro_f1"]:
            best = {"alpha": alpha, "threshold": threshold,
                    "validation_micro_f1": val_f1, "coef": coef}
    assert best is not None
    coef = best.pop("coef")
    val_scores = (X_val - x_mean) @ coef + y_mean
    test_scores = (X_test - x_mean) @ coef + y_mean
    best["validation"] = evaluate(val_scores, Y["validation"], best["threshold"], total_true["validation"])
    best["test"] = evaluate(test_scores, Y["test"], best["threshold"], total_true["test"])
    return best, coef, x_mean, y_mean


def run(args: argparse.Namespace) -> None:
    rows, original_indices = load_rows(args.input)
    splits = split_indices(rows, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    targets = {level: prepare_targets(rows, splits, level, 1 if level == "subclass" else args.group_min_support)
               for level in args.levels}
    results_path = args.output / "results.json"
    summary = {
        "input": str(args.input.resolve()), "seed": args.seed, "alphas": args.alphas,
        "group_min_support": args.group_min_support, "labeled_documents": len(rows),
        "models": {},
    }
    if results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        for key in ("input", "seed", "alphas", "group_min_support"):
            if previous.get(key) != summary[key]:
                raise ValueError(f"Resultados anteriores usam outro valor de {key}")
        summary["models"] = previous["models"]
    for view in args.views:
        start = time.monotonic()
        X = get_embeddings(view, args.embeddings, original_indices)
        if X.shape[0] != len(rows) or not np.isfinite(X).all():
            raise ValueError(f"Matriz inválida para {view}: {X.shape}")
        print(f"[{view}] {X.shape}, norma mediana={np.median(np.linalg.norm(X, axis=1)):.4f}", flush=True)
        for level in args.levels:
            classes, y, total_true, coverage = targets[level]
            result, coef, x_mean, y_mean = fit_one_view(X, splits, y, total_true, args.alphas)
            result.update({"features": int(X.shape[1]), "training_labels": len(classes),
                           "split_coverage": coverage, "elapsed_seconds": round(time.monotonic() - start, 1)})
            key = f"{view}_{level}"
            summary["models"][key] = result
            results_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            np.savez(args.output / f"model_{key}.npz", coef=coef.astype(np.float32),
                     x_mean=x_mean.astype(np.float32), y_mean=y_mean.astype(np.float32),
                     classes=np.asarray(classes), threshold=result["threshold"],
                     alpha=result["alpha"], view=view, level=level)
            print(f"[{key}] alpha={result['alpha']} validação={result['validation_micro_f1']:.3f} "
                  f"teste={result['test']['micro_f1_all_labels']:.3f} (todos os rótulos)", flush=True)
        del X
    print(f"Resultados: {results_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("patentes_xml_embeddings_ipc.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("results/embeddings"))
    parser.add_argument("--output", type=Path, default=Path("results/embedding_baseline"))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=VIEWS)
    parser.add_argument("--levels", nargs="+", choices=("subclass", "main_group"), default=("subclass", "main_group"))
    parser.add_argument("--group-min-support", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
