#!/usr/bin/env python3
"""Exporta para a API os classificadores IPC Gemma 2 treinados com v1 + v2 (2ª rodada).

Mesma receita e mesmo formato de .npz de export_retrained.py; muda só o treino:
treino do v1 (2026) + treino do v2 (XML 2014-17 e PDFs 2018+, rótulos atualizados pela concordância
da CIB, só os do universo do v1; consultas do benchmark de busca excluídas). Limiar e top_n escolhidos na validação do v1; teste do v1 medido uma vez.
Os .npz atuais (1ª rodada, só v1) vão para ipc_models/anteriores_v1/ antes de gravar os novos.

    python3 consulta_gemma/export_v2.py --scopes tr_r      (um escopo por vez, ~1 min cada)
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "consulta_gemma"))
from baseline import load_rows, select_threshold, split_indices
from embedding_experiment import prepare_targets
from hierarquia_piloto import train_flat
from train_query_classifier import f1_top_n
from export_retrained import OUT, EPOCHS, LAM, vectors, hit, save
import train_v2

V2 = ROOT / "consulta_gemma/v2"


def v2_vectors(scope):
    ta = np.load(V2 / "v2_gemma2_ta.npy") if scope in ("tr", "tr_r") else None
    cl = np.load(V2 / "v2_gemma2_claims.npy") if scope in ("r", "tr_r") else None
    X = ta if scope == "tr" else cl if scope == "r" else (ta + cl) / 2
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def main(a):
    rows, orig = load_rows(ROOT / "patentes_xml_embeddings_ipc.csv")
    splits = split_indices(rows, 42); orig = np.asarray(orig)
    report_path = OUT / "exportacao_modelos_v1v2.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    backup = OUT / "anteriores_v1"; backup.mkdir(exist_ok=True)
    targets = {level: prepare_targets(rows, splits, level, support) for level, support in (("main_group", 2), ("subclass", 1))}
    train_v2.ESCOPO = "ta"  # só para carregar rótulos do v2 (os vetores vêm de v2_vectors)
    df, _, Yg2, Ys2 = train_v2.load_v2({"gcls": targets["main_group"][0], "scls": targets["subclass"][0]})
    bench = train_v2.consultas_benchmark()
    is_train2 = (df["split"] == "train").to_numpy() & ~df["numero_base"].astype(str).isin(bench).to_numpy()
    print(f"consultas do benchmark fora do treino: {len(bench)} ({int(((df['split'] == 'train') & df['numero_base'].astype(str).isin(bench)).sum())} estavam no treino do v2)")
    for scope in a.scopes:
        t0 = time.monotonic()
        X = {k: vectors(scope, orig[v]) for k, v in splits.items()}
        X2 = v2_vectors(scope)
        report[scope] = {}
        for level in ("main_group", "subclass"):
            classes, y, total_true, _ = targets[level]
            Y2 = Yg2 if level == "main_group" else Ys2
            m2 = is_train2 & Y2.any(1)
            Xtr = np.vstack([X["train"], X2[m2]]); Ytr = np.vstack([y["train"], Y2[m2]])
            W, b = train_flat(Xtr, Ytr, LAM, EPOCHS)
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
                    "suporte_minimo": 2 if level == "main_group" else 1,
                    "treino": f"v1 ({len(y['train'])} pedidos de 2026) + v2 ({int(m2.sum())} pedidos 2014-2019+)",
                    "rodada": 2, **{k: v for k, v in metrics.items() if k != "limiar"}}
            path = OUT / f"ipc_gemma2_{scope}_{level}.npz"
            if path.exists() and not (backup / path.name).exists():
                shutil.move(str(path), backup / path.name)
            metrics["weights_sha256"] = save(path, W, b, list(classes), thr, top_n, scope, level, meta)
            metrics["n_treino"] = len(Xtr)
            report[scope][level] = metrics
            print(f"[{scope} {level}] {len(Xtr)} pedidos limiar={thr:.3f} top_n={top_n} teste Hit@1={metrics['teste_hit@1']:.3f} "
                  f"Hit@3={metrics['teste_hit@3']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scopes", nargs="+", choices=("tr", "r", "tr_r"), default=["tr_r"])
    main(p.parse_args())
