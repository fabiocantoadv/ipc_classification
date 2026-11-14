#!/usr/bin/env python3
"""Piloto: classificador IPC hierárquico (subclasse -> grupo principal) sobre vetores Gemma 2 de documento.

Entrada: média dos vetores TA e Claims (como o modelo TAC), normalizada; mesma divisão agrupada do
projeto (split_indices, semente 42) e mesmos universos de rótulos dos modelos atuais (subclasse com
suporte >= 1 e grupo principal com suporte >= 2 no treino).

Variantes (todas decodificadas de forma coerente: subclasse ordenada pelo seu melhor grupo):
  base   modelos atuais independentes (SGD), nota log σ(grupo) + log σ(subclasse)          [API hoje]
  v1     condicional por subclasse: log σ(subclasse) + log P(grupo | subclasse), com um
         classificador por subclasse treinado só com os pedidos dela (negativos = grupos irmãos)
  v1b    v1 + log σ(grupo) do modelo global (três termos)
  v2     regularização hierárquica: pesos do grupo = pesos da subclasse-mãe + ajuste próprio,
         com penalidade L2 no ajuste (grupos raros ficam perto da mãe), decodificado como a base
  ctrl   mesmo otimizador do v2, sem a mãe (separa o efeito da hierarquia do efeito do treino)
Hiperparâmetros escolhidos na validação (Hit@3 de grupo); o teste é avaliado uma vez no fim.

Etapas (cada uma cabe em poucos minutos; resultados intermediários em consulta_gemma/hierarquia/):
  python3 consulta_gemma/hierarquia_piloto.py base
  python3 consulta_gemma/hierarquia_piloto.py v1
  python3 consulta_gemma/hierarquia_piloto.py v2 --lam 1e-3      (repetir para cada λ; idem ctrl)
  python3 consulta_gemma/hierarquia_piloto.py teste
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "consulta_gemma/hierarquia"
sys.path.insert(0, str(ROOT))
from baseline import load_rows, split_indices
from embedding_experiment import prepare_targets

MODELS = ROOT / "patenteia_search/semantic_api/ipc_models"
RARE = 10  # grupo raro: menos de 10 pedidos de treino


def ls(z):
    return -np.logaddexp(0.0, -z)


def data():
    rows, orig = load_rows(ROOT / "patentes_xml_embeddings_ipc.csv")
    splits = split_indices(rows, 42)
    gcls, yg, _, _ = prepare_targets(rows, splits, "main_group", 2)
    scls, ys, _, _ = prepare_targets(rows, splits, "subclass", 1)
    orig = np.asarray(orig)
    X = (np.load(ROOT / "results/embeddings/patentes_xml_vec_gemma2_ta.npy")[orig]
         + np.load(ROOT / "results/embeddings/patentes_xml_vec_gemma2_claims.npy")[orig]) / 2
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    sidx = {c: i for i, c in enumerate(scls)}
    parent = np.array([sidx[c[:4]] for c in gcls])
    return {"X": {k: X[v] for k, v in splits.items()}, "yg": yg, "ys": ys,
            "gcls": np.array(gcls), "scls": np.array(scls), "parent": parent}


def load_npz(name):
    with np.load(MODELS / name) as d:
        return d["coef"].astype(np.float32), d["intercept"].astype(np.float32), list(d["classes"].astype(str))


def decode_metrics(group_score, D, part):
    """group_score: (n, grupos) log-nota. Subclasse = melhor grupo (coerente)."""
    yg, ys, parent = D["yg"][part], D["ys"][part], D["parent"]
    n, K = group_score.shape
    order = np.argsort(-group_score, axis=1)[:, :5]
    hits = np.take_along_axis(yg, order, 1)
    res = {f"grupo_hit@{k}": float(np.mean(hits[:, :k].any(1))) for k in (1, 3, 5)}
    S = len(D["scls"])
    best = np.full((n, S), -np.inf, dtype=np.float64)
    for k in range(S):  # melhor grupo de cada subclasse
        m = parent == k
        if m.any():
            best[:, k] = group_score[:, m].max(1)
    sorder = np.argsort(-best, axis=1)[:, :3]
    shits = np.take_along_axis(ys, sorder, 1)
    res.update({f"subclasse_hit@{k}": float(np.mean(shits[:, :k].any(1))) for k in (1, 3)})
    # recall@5 por rótulo, separando grupos raros e frequentes
    train_support = D["yg"]["train"].sum(0)
    in5 = np.zeros_like(yg); np.put_along_axis(in5, order, True, 1)
    for name, mask in (("raros", train_support < RARE), ("frequentes", train_support >= RARE)):
        tot = yg[:, mask].sum()
        res[f"recall@5_{name}"] = float((yg[:, mask] & in5[:, mask]).sum() / max(1, tot))
        res[f"rotulos_{name}"] = int(tot)
    return res


def base_scores(D, part):
    gc, gb, gcls = load_npz("ipc_gemma2_tr_r_main_group.npz")
    sc, sb, scls = load_npz("ipc_gemma2_tr_r_subclass.npz")
    assert gcls == list(D["gcls"]) and scls == list(D["scls"]), "universo de rótulos diferente dos modelos atuais"
    X = D["X"][part]
    G = X @ gc.T + gb; S = X @ sc.T + sb
    return G, S


# ---------------------------------------------------------------- v1: condicional por subclasse
def train_conditional(D, C):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    X, yg, ys, parent = D["X"]["train"], D["yg"]["train"], D["ys"]["train"], D["parent"]
    K, dim = len(D["gcls"]), X.shape[1]
    W = np.zeros((K, dim), np.float32); b = np.zeros(K, np.float32)
    for s in range(len(D["scls"])):
        kids = np.where(parent == s)[0]
        if not len(kids):
            continue
        docs = ys[:, s]
        Xs, Ys = X[docs], yg[docs][:, kids]
        for j, g in enumerate(kids):
            pos = int(Ys[:, j].sum()); n = len(Ys)
            if pos == 0 or pos == n:  # constante: proporção suavizada
                p = (pos + 0.5) / (n + 1.0); b[g] = np.log(p / (1 - p)); continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m = LogisticRegression(C=C, max_iter=300, solver="liblinear").fit(Xs, Ys[:, j])
            W[g] = m.coef_[0]; b[g] = m.intercept_[0]
    return W, b


# ---------------------------------------------------------------- v2 / controle: SGD com prior da mãe
def train_hier(D, lam, use_parent, epochs=12, batch=1024, lr=0.02, seed=0, mother="base"):
    if mother == "base":
        sc, sb, _ = load_npz("ipc_gemma2_tr_r_subclass.npz")
    else:
        with np.load(OUT / f"{mother}.npz") as d:
            sc, sb = d["W"].astype(np.float32), d["b"].astype(np.float32)
    X, Y = D["X"]["train"], D["yg"]["train"].astype(np.float32)
    parent = D["parent"]
    K, dim = Y.shape[1], X.shape[1]
    Wp = sc[parent] if use_parent else np.zeros((K, dim), np.float32)
    bp = sb[parent] if use_parent else np.zeros(K, np.float32)
    V = np.zeros((K, dim), np.float32)
    prior = Y.mean(0).clip(1e-4, 1 - 1e-4)
    c = (np.log(prior / (1 - prior)) - (bp if use_parent else 0)).astype(np.float32)
    mV = np.zeros_like(V); vV = np.zeros_like(V); mc = np.zeros_like(c); vc = np.zeros_like(c)
    rng = np.random.default_rng(seed); n = len(X); t = 0
    history = []
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
            t += 1
            Xb, Yb = X[idx], Y[idx]
            z = Xb @ (Wp + V).T + bp + c
            err = (1 / (1 + np.exp(-z)) - Yb) / len(idx)
            gV = err.T @ Xb + lam * V
            gc = err.sum(0)
            for p_, g_, m_, v_ in ((V, gV, mV, vV), (c, gc, mc, vc)):
                m_ *= 0.9; m_ += 0.1 * g_; v_ *= 0.999; v_ += 0.001 * g_ * g_
                p_ -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
        history.append(ep)
    return (Wp + V).astype(np.float32), (bp + c).astype(np.float32)


def train_flat(X, Y, lam, epochs, batch=1024, lr=0.02, seed=0):
    """Logística multirrótulo (um-contra-todos) com Adam, mesmo otimizador do controle."""
    Y = Y.astype(np.float32); K, dim = Y.shape[1], X.shape[1]
    W = np.zeros((K, dim), np.float32)
    prior = Y.mean(0).clip(1e-4, 1 - 1e-4); c = np.log(prior / (1 - prior)).astype(np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mc = np.zeros_like(c); vc = np.zeros_like(c)
    rng = np.random.default_rng(seed); n = len(X); t = 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
            t += 1
            z = X[idx] @ W.T + c
            err = (1 / (1 + np.exp(-z)) - Y[idx]) / len(idx)
            for p_, g_, m_, v_ in ((W, err.T @ X[idx] + lam * W, mW, vW), (c, err.sum(0), mc, vc)):
                m_ *= 0.9; m_ += 0.1 * g_; v_ *= 0.999; v_ += 0.001 * g_ * g_
                p_ -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
    return W, c


def component_scores(D, part, recipe):
    """Notas de grupo e de subclasse (por grupo) de uma receita: grupo, subclasse, condicional, beta."""
    X = D["X"][part]; parent = D["parent"]
    G0, S0 = base_scores(D, part)
    if recipe["grupo"] == "base":
        G = G0
    else:
        with np.load(OUT / f"{recipe['grupo']}.npz") as d:
            G = X @ d["W"].T + d["b"]
    if recipe["subclasse"] == "base":
        S = S0
    else:
        with np.load(OUT / f"{recipe['subclasse']}.npz") as d:
            S = X @ d["W"].T + d["b"]
    score = ls(G) + ls(S[:, parent])
    if recipe.get("condicional"):
        with np.load(OUT / f"{recipe['condicional']}.npz") as d:
            score = score + recipe.get("beta", 1.0) * ls(X @ d["W"].T + d["b"])
    return score


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    D = data()
    rep_path = OUT / "validacao.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    t0 = time.monotonic()
    if a.etapa == "base":
        G, S = base_scores(D, "validation")
        sp = S[:, D["parent"]]
        rep["base_independente_margem"] = decode_metrics(G, D, "validation")
        rep["base_combinada"] = decode_metrics(ls(G) + ls(sp), D, "validation")
    elif a.etapa == "v1":
        G, S = base_scores(D, "validation"); sp = S[:, D["parent"]]
        for C in a.Cs:
            W, b = train_conditional(D, C)
            np.savez(OUT / f"v1_C{C:g}.npz", W=W, b=b)
            cond = ls(D["X"]["validation"] @ W.T + b)
            rep[f"v1_C{C:g}"] = decode_metrics(ls(sp) + cond, D, "validation")
            rep[f"v1b_C{C:g}"] = decode_metrics(ls(sp) + cond + ls(G), D, "validation")
            print(f"C={C:g} v1 {rep[f'v1_C{C:g}']['grupo_hit@3']:.3f} v1b {rep[f'v1b_C{C:g}']['grupo_hit@3']:.3f} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
    elif a.etapa in ("v2", "ctrl"):
        if a.subclasse == "base":
            _, S = base_scores(D, "validation")
        else:
            with np.load(OUT / f"{a.subclasse}.npz") as d:
                S = D["X"]["validation"] @ d["W"].T + d["b"]
        sp = S[:, D["parent"]]
        W, b = train_hier(D, a.lam, use_parent=a.etapa == "v2", epochs=a.epochs, mother=a.subclasse)
        key = f"{a.etapa}_lam{a.lam:g}_ep{a.epochs}" + ("" if a.subclasse == "base" else f"_mae[{a.subclasse}]")
        np.savez(OUT / f"{key}.npz", W=W, b=b)
        G2 = D["X"]["validation"] @ W.T + b
        rep[key + "_combinada"] = decode_metrics(ls(G2) + ls(sp), D, "validation")
        print(key, rep[key + "_combinada"]["grupo_hit@3"], f"({time.monotonic() - t0:.0f}s)", flush=True)
    elif a.etapa == "sub":
        W, b = train_flat(D["X"]["train"], D["ys"]["train"], a.lam, a.epochs)
        key = f"sub_lam{a.lam:g}_ep{a.epochs}"
        np.savez(OUT / f"{key}.npz", W=W, b=b)
        S2 = D["X"]["validation"] @ W.T + b
        order = np.argsort(-S2, axis=1)[:, :3]; hits = np.take_along_axis(D["ys"]["validation"], order, 1)
        _, S0 = base_scores(D, "validation")
        o0 = np.argsort(-S0, axis=1)[:, :3]; h0 = np.take_along_axis(D["ys"]["validation"], o0, 1)
        print(f"{key}: subclasse sozinha Hit@1={hits[:, 0].mean():.3f} Hit@3={hits.any(1).mean():.3f} "
              f"| base Hit@1={h0[:, 0].mean():.3f} Hit@3={h0.any(1).mean():.3f}", flush=True)
        rep.setdefault("_subclasse_sozinha", {})[key] = {"hit@1": float(hits[:, 0].mean()), "hit@3": float(hits.any(1).mean())}
    elif a.etapa == "mix":
        recipe = {"grupo": a.grupo, "subclasse": a.subclasse, "condicional": a.condicional or None, "beta": a.beta}
        key = f"mix_g[{a.grupo}]_s[{a.subclasse}]_c[{a.condicional or '-'}]_b{a.beta:g}"
        rep[key] = decode_metrics(component_scores(D, "validation", recipe), D, "validation")
        rep[key]["_receita"] = recipe
        print(key, f"{rep[key]['grupo_hit@3']:.3f}", flush=True)
    elif a.etapa == "teste":
        sel = {}
        for fam in ("v1_", "v1b_", "v2_", "ctrl_", "mix_"):
            cands = {k: v for k, v in rep.items() if k.startswith(fam)}
            if cands:
                sel[fam.rstrip("_")] = max(cands, key=lambda k: (cands[k]["grupo_hit@3"], cands[k]["grupo_hit@1"]))
        G, S = base_scores(D, "test"); sp = S[:, D["parent"]]; Xt = D["X"]["test"]
        test = {"base_independente_margem": decode_metrics(G, D, "test"),
                "base_combinada": decode_metrics(ls(G) + ls(sp), D, "test")}
        for fam, key in sel.items():
            if fam == "mix":
                score = component_scores(D, "test", rep[key]["_receita"])
            elif fam in ("v1", "v1b"):
                C = key.split("_C")[1]
                with np.load(OUT / f"v1_C{C}.npz") as d:
                    cond = ls(Xt @ d["W"].T + d["b"])
                score = ls(sp) + cond + (ls(G) if fam == "v1b" else 0)
            else:
                model_key = key.replace("_combinada", "")
                sp_used = sp
                if "_mae[" in model_key:  # v2 decodificado com a mesma subclasse usada como mãe na validação
                    mother = model_key.split("_mae[")[1].rstrip("]")
                    with np.load(OUT / f"{mother}.npz") as d:
                        sp_used = (Xt @ d["W"].T + d["b"])[:, D["parent"]]
                with np.load(OUT / f"{model_key}.npz") as d:
                    score = ls(Xt @ d["W"].T + d["b"]) + ls(sp_used)
            test[f"{fam} ({key})"] = decode_metrics(score, D, "test")
        rep_t = {"selecao_na_validacao": sel, "teste": test}
        (OUT / "teste.json").write_text(json.dumps(rep_t, ensure_ascii=False, indent=2))
        cols = ["grupo_hit@1", "grupo_hit@3", "grupo_hit@5", "subclasse_hit@1", "subclasse_hit@3", "recall@5_raros", "recall@5_frequentes"]
        print(f"{'teste (2.918 pedidos)':42s}" + "".join(f"{c:>20s}" for c in cols))
        for k, v in test.items():
            print(f"{k:42s}" + "".join(f"{v[c]:20.3f}" for c in cols))
        print("rótulos no teste: raros", test["base_combinada"]["rotulos_raros"], "| frequentes", test["base_combinada"]["rotulos_frequentes"])
        return
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        print(f"{k:32s} g@1={v['grupo_hit@1']:.3f} g@3={v['grupo_hit@3']:.3f} g@5={v['grupo_hit@5']:.3f} "
              f"s@1={v['subclasse_hit@1']:.3f} raros@5={v['recall@5_raros']:.3f} freq@5={v['recall@5_frequentes']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("etapa", choices=("base", "v1", "v2", "ctrl", "sub", "mix", "teste"))
    p.add_argument("--grupo", default="base", help="'base' ou arquivo (sem .npz) de um modelo de grupo, ex. ctrl_lam0_ep30")
    p.add_argument("--subclasse", default="base", help="'base' ou arquivo de um modelo de subclasse, ex. sub_lam0_ep30")
    p.add_argument("--condicional", default="", help="arquivo do modelo condicional, ex. v1_C64")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--Cs", nargs="+", type=float, default=[1.0, 4.0, 16.0])
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    main(p.parse_args())
