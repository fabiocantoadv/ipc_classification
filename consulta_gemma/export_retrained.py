#!/usr/bin/env python3
"""Retreina e exporta os classificadores IPC Gemma 2 (grupo principal e subclasse) para a API.

Receita escolhida no piloto hierárquico (consulta_gemma/hierarquia_piloto.py): logística
um-contra-todos com Adam, 30 épocas, sem penalidade, sobre vetores de documento Gemma 2 normalizados,
com a mesma divisão agrupada e os mesmos universos de rótulos dos modelos anteriores. Por escopo:
  tr   -> vetores TA        r -> vetores Claims        tr_r -> média de TA e Claims (TAC)
O limiar de margem e o top_n são escolhidos na validação; o teste é medido uma vez e registrado.
Os .npz anteriores são movidos para ipc_models/anteriores_sgd/ antes de gravar os novos.

    python3 consulta_gemma/export_retrained.py --scopes tr_r     (um escopo por vez, ~1 min cada)
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "consulta_gemma"))
from baseline import load_rows, select_threshold, split_indices
from embedding_experiment import prepare_targets
from hierarquia_piloto import train_flat
from train_query_classifier import f1_top_n

OUT = ROOT / "patenteia_search/semantic_api/ipc_models"
GEMMA_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EPOCHS, LAM = 30, 0.0


def vectors(scope, idx):
    emb = ROOT / "results/embeddings"
    ta = np.load(emb / "patentes_xml_vec_gemma2_ta.npy")[idx] if scope in ("tr", "tr_r") else None
    cl = np.load(emb / "patentes_xml_vec_gemma2_claims.npy")[idx] if scope in ("r", "tr_r") else None
    X = ta if scope == "tr" else cl if scope == "r" else (ta + cl) / 2
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def hit(scores, y, k):
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.take_along_axis(y, top, 1).any(1).mean())


def save(path, W, b, classes, threshold, top_n, scope, level, meta):
    sha = hashlib.sha256(W.tobytes() + b.tobytes() + "\n".join(classes).encode()).hexdigest()
    np.savez_compressed(
        path, coef=W.astype(np.float32), intercept=b.astype(np.float32), classes=np.array(classes),
        threshold=np.float32(threshold), top_n=np.int32(top_n), scope=np.array(scope), dimension=np.int32(W.shape[1]),
        normalization=np.array("unit"), level=np.array(level), alpha=np.float64(LAM),
        gemma_model=np.array("google/embeddinggemma-2"), gemma_revision=np.array(GEMMA_REVISION),
        prompt_name=np.array("documento: title: {title} | text: {text}"), weights_sha256=np.array(sha),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    return sha


def main(a):
    rows, orig = load_rows(ROOT / "patentes_xml_embeddings_ipc.csv")
    splits = split_indices(rows, 42); orig = np.asarray(orig)
    report_path = OUT / "exportacao_modelos_retreinados.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    backup = OUT / "anteriores_sgd"; backup.mkdir(exist_ok=True)
    for scope in a.scopes:
        t0 = time.monotonic()
        X = {k: vectors(scope, orig[v]) for k, v in splits.items()}
        report[scope] = {}
        for level, support in (("main_group", 2), ("subclass", 1)):
            classes, y, total_true, _ = prepare_targets(rows, splits, level, support)
            W, b = train_flat(X["train"], y["train"], LAM, EPOCHS)
            val = X["validation"] @ W.T + b
            thr, val_f1 = select_threshold(val, y["validation"])
            by_n = {n: f1_top_n(val, y["validation"], total_true["validation"], thr, n) for n in range(1, 6)}
            top_n = max(by_n, key=lambda n: (round(by_n[n], 6), -n))
            test = X["test"] @ W.T + b
            metrics = {"limiar": thr, "top_n": top_n, "validacao_micro_f1": val_f1,
                       "validacao_hit@1": hit(val, y["validation"], 1), "validacao_hit@3": hit(val, y["validation"], 3),
                       "teste_hit@1": hit(test, y["test"], 1), "teste_hit@3": hit(test, y["test"], 3),
                       "teste_hit@5": hit(test, y["test"], 5)}
            meta = {"scope": scope, "level": level, "receita": f"logística OvR, Adam, {EPOCHS} épocas, λ={LAM}",
                    "vetores": {"tr": "TA", "r": "Claims", "tr_r": "média TA + Claims"}[scope], "split_seed": 42,
                    "suporte_minimo": support, **{k: v for k, v in metrics.items() if k != "limiar"}}
            path = OUT / f"ipc_gemma2_{scope}_{level}.npz"
            if path.exists() and not (backup / path.name).exists():
                shutil.move(str(path), backup / path.name)
            metrics["weights_sha256"] = save(path, W, b, list(classes), thr, top_n, scope, level, meta)
            report[scope][level] = metrics
            print(f"[{scope} {level}] limiar={thr:.3f} top_n={top_n} teste Hit@1={metrics['teste_hit@1']:.3f} "
                  f"Hit@3={metrics['teste_hit@3']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scopes", nargs="+", choices=("tr", "r", "tr_r"), default=["tr_r"])
    main(p.parse_args())
