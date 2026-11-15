#!/usr/bin/env python3
"""2ª rodada do classificador IPC: mesma receita do v1, variando só os dados de treino.

Receita fixa (igual a export_retrained.py): logística OvR, Adam, 30 épocas, λ=0, vetores TAC
(média de TA e Claims, Gemma 2, normalizados). Universo de rótulos fixo = o do v1 (2026):
575 subclasses e 2.585 grupos principais. Grupos antigos do v2 são atualizados pelas listas de
concordância da IPC quando a correspondência é 1:1 (ambíguos descartados); o que ficar fora do universo é descartado.
Decodificação hierárquica: nota do grupo = log σ(grupo) + log σ(subclasse-mãe).

Variantes de treino:
    v1     treino do v1 (13.565 pedidos de 2026)          -> linha de base, deve reproduzir 0,532 / 0,699
    v2     treino do v2 (XML 2014-17 + PDFs 2018+)
    v1v2   treino do v1 + treino do v2
Árbitro comum: validação e teste do v1 (2026). Secundário: teste do v2 por origem.

    python3 consulta_gemma/train_v2.py treino v1v2 grupo        # um modelo por chamada (pula se já existe)
    python3 consulta_gemma/train_v2.py avaliacao                 # mede todas as variantes treinadas
    (acrescente --escopo ta para usar só os vetores de título e resumo)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "consulta_gemma"))
from hierarquia_piloto import data as _data_tac, decode_metrics, ls, train_flat

V2 = ROOT / "consulta_gemma/v2"
V2_CSV = Path.home() / "Downloads/Recorte_imagens/claims_extractor/dataset_ipc_v2.csv"
EPOCHS, LAM = 30, 0.0
ESCOPO = "tac"  # "tac" (média TA + Claims) ou "ta" (só título e resumo); --escopo ta na linha de comando


def data():
    """Mesmo que hierarquia_piloto.data(), mas com os vetores do escopo escolhido."""
    D = _data_tac()
    if ESCOPO == "tac":
        return D
    from baseline import load_rows, split_indices
    rows, orig = load_rows(ROOT / "patentes_xml_embeddings_ipc.csv")
    splits = split_indices(rows, 42)
    X = np.load(ROOT / "results/embeddings/patentes_xml_vec_gemma2_ta.npy")[np.asarray(orig)]
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    D["X"] = {k: X[v] for k, v in splits.items()}
    return D


def sufixo():
    return "" if ESCOPO == "tac" else f"_{ESCOPO}"


def v2_csv_path():
    for p in (V2_CSV, ROOT.parent / "Recorte_imagens/claims_extractor/dataset_ipc_v2.csv",
              ROOT.parent / "claims_extractor/dataset_ipc_v2.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("dataset_ipc_v2.csv")


def load_v2(D):
    import pandas as pd
    df = pd.read_csv(v2_csv_path(), usecols=["numero_base", "origem", "split", "ipc"])
    ta = np.load(V2 / "v2_gemma2_ta.npy")
    X = ta if ESCOPO == "ta" else (ta + np.load(V2 / "v2_gemma2_claims.npy")) / 2
    assert len(X) == len(df)
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    gpos = {c: i for i, c in enumerate(D["gcls"])}; spos = {c: i for i, c in enumerate(D["scls"])}
    Yg = np.zeros((len(df), len(gpos)), bool); Ys = np.zeros((len(df), len(spos)), bool)
    for r, s in enumerate(df["ipc"]):
        for c in json.loads(s):
            c = c.strip().replace(" ", "")
            g, sub = atualizar_grupo(c.split("/")[0] + "/00", c[:4])
            if g in gpos: Yg[r, gpos[g]] = True
            if sub in spos: Ys[r, spos[sub]] = True
    return df, X, Yg, Ys


_CONC = None


def atualizar_grupo(g, sub):
    """Leva um grupo principal antigo à IPC 2026.01 pelas listas de concordância (2018.01 a 2026.01, sem 2022.01).

    Correspondência 1:1 -> grupo novo. 1:N -> grupo descartado (ambíguo); a subclasse fica se todos os
    destinos forem da mesma subclasse. Grupos sem entrada nas listas ficam como estão."""
    global _CONC
    if _CONC is None:
        _CONC = json.loads((V2 / "concordancia_ipc_grupos.json").read_text(encoding="utf-8"))
    cur = {g}
    for v in sorted(_CONC):
        nxt = set()
        for x in cur:
            nxt |= set(_CONC[v].get(x, [x]))
        cur = nxt
    if cur == {g}:
        return g, sub
    subs = {x[:4] for x in cur}
    return (next(iter(cur)) if len(cur) == 1 else None), (next(iter(subs)) if len(subs) == 1 else None)


def consultas_benchmark():
    """Pedidos usados como consulta no benchmark de busca (qrels e pasta benchmark/): ficam fora do treino,
    para que a avaliação do reranking não use um classificador que já viu a IPC da própria consulta."""
    ids = {l.split("\t")[0].strip() for l in (ROOT / "qrels/qrels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()}
    ids |= {d.name.removeprefix("BR") for d in (ROOT / "benchmark").iterdir() if d.name.startswith("BR")}
    return ids


def train_set(D, V, variante, nivel):
    Y1 = D["yg" if nivel == "grupo" else "ys"]["train"]; X1 = D["X"]["train"]
    if variante == "v1":
        return X1, Y1
    df, X2, Yg2, Ys2 = V
    Y2 = Yg2 if nivel == "grupo" else Ys2
    # pedidos do v2 sem rótulo no universo saem, e também as consultas do benchmark de busca
    m2 = (df["split"] == "train").to_numpy() & Y2.any(1) & ~df["numero_base"].astype(str).isin(consultas_benchmark()).to_numpy()
    if variante == "v1":
        return X1, Y1
    if variante == "v2":
        return X2[m2], Y2[m2]
    return np.vstack([X1, X2[m2]]), np.vstack([Y1, Y2[m2]])


def treino(variante, nivel):
    path = V2 / f"modelo_{variante}_{nivel}{sufixo()}.npz"
    if path.exists():
        print("já existe:", path.name); return
    D = data(); V = None if variante == "v1" else load_v2(D)
    X, Y = train_set(D, V, variante, nivel)
    t0 = time.monotonic()
    W, b = train_flat(X, Y, LAM, EPOCHS)
    np.savez(path, W=W, b=b, n=len(X))
    print(f"[{variante} {nivel}] {len(X)} pedidos, {Y.shape[1]} rótulos, {time.monotonic() - t0:.0f}s -> {path.name}")


def score(X, variante, parent):
    with np.load(V2 / f"modelo_{variante}_grupo{sufixo()}.npz") as g, np.load(V2 / f"modelo_{variante}_subclasse{sufixo()}.npz") as s:
        G = X @ g["W"].T + g["b"]; S = X @ s["W"].T + s["b"]
    return ls(G) + ls(S[:, parent])


def avaliacao():
    D = data(); V = load_v2(D); df, X2, Yg2, Ys2 = V
    # D2: teste do v2 por origem, com o suporte de treino do v1 definindo "raros" (comparável)
    parts = {}
    for origem in ("xml_rpi_2014_2017", "pdf_bucket", "pdf_2019"):
        m = ((df["split"] == "test") & (df["origem"] == origem)).to_numpy() & Yg2.any(1)
        parts[f"v2_teste_{origem}"] = m
    D2 = {"X": {k: X2[m] for k, m in parts.items()}, "yg": {"train": D["yg"]["train"], **{k: Yg2[m] for k, m in parts.items()}},
          "ys": {k: Ys2[m] for k, m in parts.items()}, "gcls": D["gcls"], "scls": D["scls"], "parent": D["parent"]}
    rel = {"receita": f"logística OvR, Adam, {EPOCHS} épocas, λ={LAM}", "escopo": ESCOPO, "variantes": {}}
    for variante in ("v1", "v2", "v1v2"):
        if not all((V2 / f"modelo_{variante}_{nv}{sufixo()}.npz").exists() for nv in ("grupo", "subclasse")):
            continue
        r = {}
        for part in ("validation", "test"):
            r[f"v1_{part}"] = decode_metrics(score(D["X"][part], variante, D["parent"]), D, part)
        for part in parts:
            r[part] = decode_metrics(score(D2["X"][part], variante, D["parent"]), D2, part)
            r[part]["n"] = int(parts[part].sum())
        rel["variantes"][variante] = r
        t = r["v1_test"]
        print(f"{variante:5s} teste v1: grupo Hit@1={t['grupo_hit@1']:.3f} Hit@3={t['grupo_hit@3']:.3f} "
              f"subclasse Hit@1={t['subclasse_hit@1']:.3f} Hit@3={t['subclasse_hit@3']:.3f} "
              f"recall@5 raros={t['recall@5_raros']:.3f}")
        for part in parts:
            t = r[part]
            print(f"      {part:30s} n={t['n']:5d} grupo Hit@1={t['grupo_hit@1']:.3f} Hit@3={t['grupo_hit@3']:.3f} "
                  f"subclasse Hit@1={t['subclasse_hit@1']:.3f}")
    (V2 / f"resultados_v2{sufixo()}.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    print("relatório:", V2 / f"resultados_v2{sufixo()}.json")


if __name__ == "__main__":
    if "--escopo" in sys.argv:
        i = sys.argv.index("--escopo"); ESCOPO = sys.argv[i + 1]; del sys.argv[i:i + 2]
    if sys.argv[1] == "treino":
        treino(sys.argv[2], sys.argv[3])
    else:
        avaliacao()
