#!/usr/bin/env python3
"""Baseline multilabel IPC por subclasse, com ablação das reivindicações."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


VIEWS = ("title_abstract", "first_claim", "independent_claims", "all_claims")
CLAIM_START = re.compile(r"(?m)^\s*(\d{1,3})\s*[.)]\s+")
DEPENDENCY = re.compile(r"\breivindicac(?:ao|oes)\b|\bclaim(?:s)?\b")


def unaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def split_claims(text: str) -> list[str]:
    starts = list(CLAIM_START.finditer(text))
    if not starts:
        return [text.strip()] if text.strip() else []
    parts = [text[m.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)].strip()
             for i, m in enumerate(starts)]
    # Alguns XMLs têm um cabeçalho numerado antes da primeira reivindicação real.
    return [part for part in parts if len(part) >= 40]


def is_dependent(claim: str) -> bool:
    head = unaccent(claim[:300]).lower()
    return bool(DEPENDENCY.search(head))


def make_text(row: dict[str, str], view: str) -> str:
    base = f"titulo {row['titulo']} resumo {row['resumo']}"
    if view == "title_abstract":
        return base
    claims = split_claims(row["reivindicacoes"])
    if view == "first_claim":
        chosen = claims[:1]
    elif view == "independent_claims":
        chosen = [claim for i, claim in enumerate(claims) if i == 0 or not is_dependent(claim)]
    elif view == "all_claims":
        chosen = claims
    else:
        raise ValueError(view)
    return base + " reivindicacoes " + " ".join(chosen)


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[int]]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    original_indices = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if not (row["ipc"] or "").strip():
                continue
            rows.append(row)
            original_indices.append(i)
    return rows, original_indices


def labels_for(row: dict[str, str], level: str) -> set[str]:
    codes = {code.strip().replace(" ", "") for code in row["ipc"].split(";") if code.strip()}
    if level == "subclass":
        return {code[:4] for code in codes}
    if level == "main_group":
        return {code.split("/")[0] + "/00" for code in codes}
    raise ValueError(level)


def split_indices(rows: list[dict[str, str]], seed: int) -> dict[str, np.ndarray]:
    groups = [hashlib.sha1(unaccent((r["titulo"] + " " + r["resumo"]).lower()).encode()).hexdigest()
              for r in rows]
    all_indices = np.arange(len(rows))
    train, held = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
                       .split(all_indices, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
                                 .split(held, groups=np.asarray(groups)[held]))
    return {"train": train, "validation": held[val_local], "test": held[test_local]}


def predict_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    predicted = scores >= threshold
    empty = ~predicted.any(axis=1)
    predicted[empty, scores[empty].argmax(axis=1)] = True
    return predicted


def select_threshold(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    quantiles = np.linspace(0.92, 0.9995, 70)
    candidates = np.unique(np.quantile(scores, quantiles))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        predicted = predict_at_threshold(scores, float(threshold))
        score = f1_score(truth, predicted, average="micro", zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def evaluate(scores: np.ndarray, truth: np.ndarray, threshold: float,
             total_true_labels: np.ndarray) -> dict[str, float]:
    predicted = predict_at_threshold(scores, threshold)
    result = {
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "macro_f1_all_train_labels": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(truth, predicted, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, predicted, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(predicted == truth, axis=1))),
        "mean_predicted_labels": float(predicted.sum(axis=1).mean()),
    }
    true_positives = int(np.count_nonzero(predicted & truth))
    result["micro_f1_all_labels"] = float(2 * true_positives / max(1, predicted.sum() + total_true_labels.sum()))
    for k in (1, 3, 5):
        top = np.argpartition(scores, -k, axis=1)[:, -k:]
        hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
        result[f"recall_at_{k}"] = float(np.mean(hits / np.maximum(1, truth.sum(axis=1))))
        result[f"recall_at_{k}_all_labels"] = float(np.mean(hits / np.maximum(1, total_true_labels)))
        result[f"hit_at_{k}"] = float(np.mean(hits > 0))
    return result


def run(args: argparse.Namespace) -> None:
    if args.output is None:
        suffix = "" if args.min_train_support == 1 else f"_min{args.min_train_support}"
        args.output = Path(f"results/baseline_{args.level}{suffix}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, original_indices = load_rows(args.input)
    splits = split_indices(rows, args.seed)
    label_sets = [labels_for(row, args.level) for row in rows]
    train_counts = Counter(label for i in splits["train"] for label in label_sets[i])
    training_labels = sorted(label for label, count in train_counts.items() if count >= args.min_train_support)
    mlb = MultiLabelBinarizer(classes=training_labels)
    mlb.fit([[]])
    with (args.output / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_row_zero_based", "numero_br", "split"])
        for part, indices in splits.items():
            for i in indices:
                writer.writerow([original_indices[i], rows[i]["numero_br"], part])
    known = set(training_labels)
    y = {part: mlb.transform([label_sets[i] & known for i in indices]).astype(bool)
         for part, indices in splits.items()}
    total_true = {part: np.asarray([len(label_sets[i]) for i in indices], dtype=np.int32)
                  for part, indices in splits.items()}
    coverage = {part: {
        "documents": int(len(indices)),
        "label_assignments": int(sum(len(label_sets[i]) for i in indices)),
        "unmodeled_label_assignments": int(sum(len(label_sets[i] - known) for i in indices)),
        "documents_with_unmodeled_label": int(sum(bool(label_sets[i] - known) for i in indices)),
        "never_seen_in_training_assignments": int(sum(len(label_sets[i] - train_counts.keys()) for i in indices)),
    } for part, indices in splits.items()}
    summary = {
        "input": str(args.input.resolve()), "seed": args.seed,
        "level": args.level, "min_train_support": args.min_train_support,
        "max_features": args.max_features, "linear_svc_c": args.c,
        "labeled_documents": len(rows), "training_labels": len(training_labels),
        "split_coverage": coverage, "views": {},
        "note": "Métricas sem sufixo all_labels consideram somente rótulos treinados; métricas all_labels incluem os rótulos não treinados no denominador.",
    }
    result_path = args.output / "results.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        settings = ("input", "seed", "max_features", "linear_svc_c")
        if any(previous.get(key) != summary[key] for key in settings) or any(
            previous.get(key, summary[key]) != summary[key] for key in ("level", "min_train_support")
        ):
            raise ValueError("Resultados anteriores usam outro arquivo, divisão ou configuração de modelo")
        summary["views"] = previous["views"]
    for view in args.views:
        print(f"\n[{view}] preparando textos", flush=True)
        start = time.monotonic()
        texts = [make_text(row, view) for row in rows]
        vectorizer = TfidfVectorizer(
            strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
            min_df=3, max_df=0.98, max_features=args.max_features,
            sublinear_tf=True, dtype=np.float32,
        )
        X_train = vectorizer.fit_transform([texts[i] for i in splits["train"]])
        X_val = vectorizer.transform([texts[i] for i in splits["validation"]])
        X_test = vectorizer.transform([texts[i] for i in splits["test"]])
        del texts
        print(f"[{view}] matriz {X_train.shape}, nnz={X_train.nnz:,}; treinando OvR", flush=True)
        model = OneVsRestClassifier(LinearSVC(C=args.c, dual="auto", class_weight="balanced", max_iter=2000),
                                    n_jobs=args.n_jobs)
        model.fit(X_train, y["train"])
        val_scores = np.asarray(model.decision_function(X_val))
        threshold, val_f1 = select_threshold(val_scores, y["validation"])
        test_scores = np.asarray(model.decision_function(X_test))
        result = {
            "features": int(X_train.shape[1]), "train_nonzeros": int(X_train.nnz),
            "threshold_from_validation": threshold, "validation_micro_f1": val_f1,
            "validation": evaluate(val_scores, y["validation"], threshold, total_true["validation"]),
            "test": evaluate(test_scores, y["test"], threshold, total_true["test"]),
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
        summary["views"][view] = result
        result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_models:
            joblib.dump({"vectorizer": vectorizer, "model": model, "classes": training_labels,
                         "threshold": threshold, "view": view, "level": args.level},
                        args.output / f"model_{view}.joblib", compress=3)
        print(f"[{view}] validação F1={val_f1:.3f}; teste F1={result['test']['micro_f1']:.3f}; "
              f"tempo={result['elapsed_seconds']}s", flush=True)
        del X_train, X_val, X_test, model, vectorizer, val_scores, test_scores
    print(f"\nResultados: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("patentes_xml_embeddings_ipc.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--level", choices=("subclass", "main_group"), default="subclass")
    parser.add_argument("--min-train-support", type=int, default=1)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=VIEWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-models", action="store_true")
    run(parser.parse_args())
