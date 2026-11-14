#!/usr/bin/env python3
"""Classificadores IPC (grupo principal) sobre vetores Gemma 2 de CONSULTA, por escopo.

1) Transferência: aplica os SGD treinados com vetores de documento aos vetores de consulta.
2) Treino de consulta: One-vs-Rest SGD (log_loss, L2) nos vetores SearchQuery do treino.
   Alpha, limiar global e número máximo de grupos (top_n) são escolhidos na validação;
   o teste é avaliado uma única vez, com a configuração escolhida.
3) Exporta .npz (pesos lineares, interceptos, classes, limiar, top_n, metadados e hash) para a
   inferência só com NumPy na API, e confere a paridade NumPy x scikit-learn no teste.
Mesma divisão agrupada do projeto (split_indices) e mesmo universo de rótulos (>= 2 no treino)."""
from __future__ import annotations
import argparse, hashlib, json, sys, time, warnings
from pathlib import Path
import joblib, numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baseline import evaluate, load_rows, select_threshold, split_indices
from embedding_experiment import prepare_targets

DOC_MODEL = {"tr": "gemma_ta", "r": "gemma_claims", "tr_r": "gemma_mean"}
GEMMA_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


def unit(X):
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def f1_top_n(scores, truth, total_true, threshold, n):
    """micro-F1 com todos os rótulos quando no máximo n grupos acima do limiar são devolvidos."""
    order = np.argsort(-scores, axis=1)[:, :n]
    top = np.take_along_axis(scores, order, 1)
    keep = top >= threshold
    keep[:, 0] = True  # mesmo critério da linha de base: sempre ao menos o maior escore
    pred = np.zeros_like(truth)
    np.put_along_axis(pred, order, keep, 1)
    tp = int(np.count_nonzero(pred & truth))
    return 2 * tp / max(1, int(pred.sum()) + int(total_true.sum()))


def export(path, model, classes, threshold, top_n, scope, alpha, meta, level="main_group", prompt_name="SearchQuery"):
    dim = next(est.coef_.shape[1] for est in model.estimators_ if hasattr(est, "coef_"))
    # Classe sem positivos vira _ConstantPredictor no scikit-learn: escore constante (coef 0).
    coef = np.vstack([est.coef_.ravel() if hasattr(est, "coef_") else np.zeros(dim)
                      for est in model.estimators_]).astype(np.float32)
    intercept = np.array([float(est.intercept_[0]) if hasattr(est, "coef_") else float(np.ravel(est.y_)[0])
                          for est in model.estimators_], dtype=np.float32)
    sha = hashlib.sha256(coef.tobytes() + intercept.tobytes() + "\n".join(classes).encode()).hexdigest()
    np.savez_compressed(
        path, coef=coef, intercept=intercept, classes=np.array(classes), threshold=np.float32(threshold),
        top_n=np.int32(top_n), scope=np.array(scope), dimension=np.int32(coef.shape[1]),
        normalization=np.array("unit"), level=np.array(level), alpha=np.float64(alpha),
        gemma_model=np.array("google/embeddinggemma-2"), gemma_revision=np.array(GEMMA_REVISION),
        prompt_name=np.array(prompt_name), weights_sha256=np.array(sha),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    return sha, coef, intercept


def parity(model, coef, intercept, X, threshold):
    ref = np.asarray(model.decision_function(X))
    ours = X @ coef.T + intercept
    top_ref = np.argsort(-ref, axis=1)[:, :3]; top_ours = np.argsort(-ours, axis=1)[:, :3]
    near = np.abs(ref - threshold) < 0.05
    return {"documentos": int(X.shape[0]), "dif_max_escore": float(np.abs(ref - ours).max()),
            "top3_identico": float(np.mean(np.all(top_ref == top_ours, axis=1))),
            "decisao_limiar_identica": float(np.mean((ref >= threshold) == (ours >= threshold))),
            "pares_perto_do_limiar": int(near.sum()),
            "decisao_perto_do_limiar_identica": float(np.mean((ref[near] >= threshold) == (ours[near] >= threshold))) if near.any() else None}


def main(a):
    rows, original = load_rows(a.input)
    splits = split_indices(rows, a.seed)
    classes, y, total_true, coverage = prepare_targets(rows, splits, "main_group", a.group_min_support)
    a.output.mkdir(parents=True, exist_ok=True)
    report = {"config": {k: str(v) for k, v in vars(a).items()}, "coverage": coverage,
              "training_labels": len(classes), "scopes": {}}
    out_json = a.output / "resultados_consulta.json"
    for scope in a.scopes:
        t0 = time.monotonic()
        X = np.load(a.vectors / f"query_{scope}.npy")[original]
        X = {part: unit(X[idx]) for part, idx in splits.items()}
        ytrain = y["train"]
        if a.max_train_docs:  # só para teste rápido do código; não usar nos modelos finais
            X["train"], ytrain = X["train"][: a.max_train_docs], ytrain[: a.max_train_docs]
        res = {}
        # 1) transferência dos modelos de documento
        doc_path = a.doc_models / f"model_{DOC_MODEL[scope]}_main_group.joblib"
        if doc_path.exists():
            art = joblib.load(doc_path)
            if list(art["classes"]) != list(classes):
                raise ValueError(f"Universo de rótulos diferente em {doc_path}")
            sc = {p: np.asarray(art["model"].decision_function(art["scaler"].transform(X[p])))
                  for p in ("validation", "test")}
            thr, _ = select_threshold(sc["validation"], y["validation"])
            res["transferencia_modelo_documento"] = {
                "modelo": doc_path.name,
                "teste_limiar_original": evaluate(sc["test"], y["test"], art["threshold"], total_true["test"]),
                "teste_limiar_revalidado": evaluate(sc["test"], y["test"], thr, total_true["test"]),
                "limiar_revalidado": thr}
            del art, sc
        # 2) treino com vetores de consulta
        best, best_model = None, None
        for alpha in a.alphas:
            est = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, max_iter=a.max_iter, tol=a.tol,
                                random_state=a.seed)
            model = OneVsRestClassifier(est, n_jobs=a.n_jobs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                with parallel_backend("threading"):
                    model.fit(X["train"], ytrain)
            val = np.asarray(model.decision_function(X["validation"]))
            thr, f1 = select_threshold(val, y["validation"])
            print(f"[{scope}] alpha={alpha:g} validação micro-F1={f1:.4f}", flush=True)
            if best is None or f1 > best["validation_micro_f1"]:
                best = {"alpha": alpha, "threshold": thr, "validation_micro_f1": f1}; best_model = model
        val = np.asarray(best_model.decision_function(X["validation"]))
        by_n = {n: f1_top_n(val, y["validation"], total_true["validation"], best["threshold"], n) for n in range(1, 6)}
        top_n = max(by_n, key=lambda n: (round(by_n[n], 6), -n))
        best["validation_f1_por_top_n"] = {str(k): round(v, 4) for k, v in by_n.items()}
        best["top_n"] = top_n
        test = np.asarray(best_model.decision_function(X["test"]))  # avaliação única do teste
        best["test"] = evaluate(test, y["test"], best["threshold"], total_true["test"])
        best["test"]["micro_f1_all_labels_top_n"] = f1_top_n(test, y["test"], total_true["test"], best["threshold"], top_n)
        best["nonconverged_labels"] = int(sum(getattr(e, 'n_iter_', 0) >= a.max_iter for e in best_model.estimators_))
        res["modelo_consulta"] = best
        meta = {"scope": scope, "trained_on": "vetores SearchQuery do treino", "split_seed": a.seed,
                "group_min_support": a.group_min_support, "validation_micro_f1": best["validation_micro_f1"],
                "test_micro_f1_all_labels": best["test"]["micro_f1_all_labels"], "test_hit_at_3": best["test"]["hit_at_3"],
                "vectors_meta": json.loads((a.vectors / f"query_{scope}.json").read_text(encoding="utf-8"))
                if (a.vectors / f"query_{scope}.json").exists() else None}
        npz = a.output / f"ipc_gemma2_{scope}_main_group.npz"
        sha, coef, intercept = export(npz, best_model, classes, best["threshold"], top_n, scope, best["alpha"], meta)
        res["exportacao"] = {"arquivo": npz.name, "weights_sha256": sha,
                             "paridade_teste": parity(best_model, coef, intercept, X["test"], best["threshold"])}
        if a.save_joblib:
            joblib.dump({"model": best_model, "classes": classes, "threshold": best["threshold"], "scope": scope},
                        a.output / f"ipc_gemma2_{scope}_main_group.joblib", compress=3)
        res["segundos"] = round(time.monotonic() - t0, 1)
        report["scopes"][scope] = res
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        tr = res.get("transferencia_modelo_documento", {}).get("teste_limiar_revalidado", {})
        print(f"[{scope}] teste consulta: F1 todos={best['test']['micro_f1_all_labels']:.3f} Hit@3={best['test']['hit_at_3']:.3f} "
              f"| transferência doc: F1 todos={tr.get('micro_f1_all_labels', float('nan')):.3f} Hit@3={tr.get('hit_at_3', float('nan')):.3f} "
              f"| paridade dif={res['exportacao']['paridade_teste']['dif_max_escore']:.2e}", flush=True)
    print("Relatório:", out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "patentes_xml_embeddings_ipc.csv")
    p.add_argument("--vectors", type=Path, default=ROOT / "consulta_gemma/vetores")
    p.add_argument("--doc-models", type=Path, default=ROOT / "results/embedding_sgd")
    p.add_argument("--output", type=Path, default=ROOT / "patenteia_search/semantic_api/ipc_models")
    p.add_argument("--scopes", nargs="+", default=["tr", "r", "tr_r"])
    p.add_argument("--alphas", nargs="+", type=float, default=[1e-4, 1e-5])
    p.add_argument("--group-min-support", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-joblib", action="store_true")
    p.add_argument("--max-train-docs", type=int, default=0, help=argparse.SUPPRESS)
    main(p.parse_args())
