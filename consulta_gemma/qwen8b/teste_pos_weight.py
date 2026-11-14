#!/usr/bin/env python3
"""Teste do peso por frequência (pos_weight) na receita hierárquica — escolha na validação, teste uma vez.

Ideia vinda de github.com/rcgsheffield/ipc-classifier (BCE com peso inverso à frequência). Testado com o Gemma 2 TAC
na amostra 1 (13/11/2026): o esquema original (α = 1) piora muito; atenuado (α = 0,25) dá +0,9 pt no grupo Hit@1.
Famílias de peso:
    repo  : w = (n_treino / (n_rótulo + 1))^α, normalizado para média 1 (também rebaixa rótulos frequentes)
    raros : w = max(1, (10 / n_rótulo)^α)                  (só reforça rótulos com menos de 10 exemplos)

Uso:  python consulta_gemma/qwen8b/teste_pos_weight.py --ta PASTA_TA --c PASTA_C      (ou --vetores PASTA para uma só)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum import AQUI, bootstrap_pareado, carregar_notebook, media_normalizada


def treinar_com_peso(g, X, Y, w):
    """Mesma receita do notebook (Adam, 30 épocas, lotes de 1.024, semente 0), com BCE ponderada nos positivos."""
    X = torch.as_tensor(X); Y = torch.as_tensor(Y, dtype=torch.float32); w = torch.as_tensor(w, dtype=torch.float32)
    K, dim = Y.shape[1], X.shape[1]; W = torch.zeros((K, dim))
    prior = Y.mean(0).clamp(1e-4, 1 - 1e-4); c = torch.log(prior / (1 - prior))
    mW, vW, mc, vc = torch.zeros_like(W), torch.zeros_like(W), torch.zeros_like(c), torch.zeros_like(c)
    rng = np.random.default_rng(g["SEMENTE"]); n = len(X); t = 0
    for _ in range(g["EPOCAS"]):
        for idx in np.array_split(rng.permutation(n), max(1, n // g["LOTE"])):
            t += 1; ii = torch.as_tensor(idx); xb, yb = X[ii], Y[ii]
            s = torch.sigmoid(xb @ W.T + c)
            err = (s * (1 - yb) + (s - 1) * w * yb) / len(idx)       # derivada da BCE com pos_weight
            for p_, g_, m_, v_ in ((W, err.T @ xb, mW, vW), (c, err.sum(0), mc, vc)):
                m_.mul_(0.9).add_(0.1 * g_); v_.mul_(0.999).add_(0.001 * g_ * g_)
                p_.sub_(g["LR"] * (m_ / (1 - 0.9 ** t)) / (torch.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8))
    return W.numpy(), c.numpy()


def principal(a):
    g = carregar_notebook(saida=a.saida)
    X = g["carregar_vetores"](a.vetores)[0] if a.vetores else media_normalizada(g["carregar_vetores"](a.ta)[0], g["carregar_vetores"](a.c)[0])
    IDX, YG, YS = g["IDX"], g["YG"], g["YS"]; tr = IDX["train"]; ntr = len(tr)
    cg, cs = YG[tr].sum(0).astype(float), YS[tr].sum(0).astype(float)

    def pesos(cont, fam, alfa):
        if fam == "base": return np.ones_like(cont)
        if fam == "repo": w = (ntr / (cont + 1.0)) ** alfa; return w / w.mean()
        return np.maximum(1.0, (10.0 / np.maximum(cont, 1)) ** alfa)

    res = {}
    for fam, alfa in [("base", 0)] + [("repo", x) for x in (0.25, 0.5, 1.0)] + [("raros", x) for x in (0.5, 1.0, 1.5)]:
        t0 = time.time()
        Wg, bg = treinar_com_peso(g, X[tr], YG[tr], pesos(cg, fam, alfa)); Ws, bs = treinar_com_peso(g, X[tr], YS[tr], pesos(cs, fam, alfa))
        r = {}
        for parte in ("validation", "test"):
            og, os_ = g["decodificar"](X[IDX[parte]], Wg, bg, Ws, bs); r[parte], pp = g["metricas"](og, os_, IDX[parte])
            if parte == "test": r["pp"] = pp.tolist()
        res[f"{fam}_{alfa}"] = r; v, te = r["validation"], r["test"]
        print(f"{fam:5s} α={alfa:<4} VAL g@1={v['grupo_hit@1']:.3f} | TESTE g@1={te['grupo_hit@1']:.3f} g@3={te['grupo_hit@3']:.3f} "
              f"s@1={te['subclasse_hit@1']:.3f} s@3={te['subclasse_hit@3']:.3f} raros@5={te['recall@5_raros']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    melhor = max((k for k in res if k != "base_0"), key=lambda k: res[k]["validation"]["grupo_hit@1"])
    A, B = np.array(res[melhor]["pp"]), np.array(res["base_0"]["pp"])
    for j, col in enumerate(["grupo@1", "grupo@3", "subclasse@1", "subclasse@3"]):
        d, lo, hi = bootstrap_pareado(A[:, j], B[:, j]); print(f"{melhor} − sem peso | {col:12s} {d*100:+.1f} [{lo*100:+.1f}; {hi*100:+.1f}]")
    json.dump(res, open(Path(g["SAIDA"]) / "pos_weight.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vetores", type=Path); p.add_argument("--ta", type=Path); p.add_argument("--c", type=Path)
    p.add_argument("--saida", type=Path, default=AQUI / "saida_pos_weight")
    a = p.parse_args(); assert a.vetores or (a.ta and a.c), "informe --vetores ou --ta e --c"
    principal(a)
