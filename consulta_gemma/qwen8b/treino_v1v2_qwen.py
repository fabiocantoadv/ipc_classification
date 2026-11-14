#!/usr/bin/env python3
"""Treina o classificador IPC hierárquico com as DUAS amostras (v1 + v2) sobre a média TA + C de um modelo Qwen3.

É o procedimento que gerou os resultados v1 + v2 do Qwen 0.6B, 4B e 8B (resultados/resultados_qwen*_media_v1v2.json)
e os pesos em modelos/. Mesma receita do notebook classificador_ipc_qwen8b.ipynb:
  - logística um-contra-todos, Adam (lr 0,02), lotes de 1.024, 30 épocas, sem penalidade;
  - um modelo de grupo principal (2.585 rótulos) e um de subclasse (575), universo fixo da amostra 1;
  - decodificação hierárquica: nota do grupo = log σ(grupo) + log σ(subclasse-mãe);
  - treino = treino da amostra 1 (13.565) + treino da amostra 2 (≈18,6 mil no grupo, ≈20,0 mil na subclasse;
    sem pedidos sem rótulo no universo e sem as consultas do benchmark de busca);
  - avaliação no teste da amostra 1 (2.918 pedidos de 2026) e comparação pareada com o Gemma 2.

Uso (pastas no formato do notebook de embeddings: embeddings_*.npy + pedidos_*.csv, em blocos ou arquivo único):
    python consulta_gemma/qwen8b/treino_v1v2_qwen.py --nome qwen8b \
        --ta1 Embeddings/Amostra_1/Qwen_8B_a1/8B_TA --c1 Embeddings/Amostra_1/Qwen_8B_a1/8B_C \
        --ta2 Embeddings/Amostra_2/Qwen_8B_a2/TA   --c2 Embeddings/Amostra_2/Qwen_8B_a2/C
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum import AQUI, bootstrap_pareado, carregar_notebook, ler_vetores, media_normalizada, rotulos_amostra2


def principal(a):
    g = carregar_notebook(saida=a.saida)
    IDS, IDX, YG, YS = g["IDS"], g["IDX"], g["YG"], g["YS"]
    t0 = time.time()
    # amostra 1: alinhada à ordem de poc1_rotulos.csv.gz pelo carregar_vetores do notebook
    X1 = media_normalizada(g["carregar_vetores"](a.ta1)[0], g["carregar_vetores"](a.c1)[0])
    # amostra 2: alinhada à ordem do dataset_ipc_v2.csv (a mesma do .npz de rótulos)
    r = rotulos_amostra2()
    (ta2, ids_ta), (c2, ids_c) = ler_vetores(a.ta2), ler_vetores(a.c2)
    assert [i.lstrip("0") for i in ids_ta] == r["numero_base"] == [i.lstrip("0") for i in ids_c], "ordem da amostra 2"
    X2 = media_normalizada(ta2, c2); del ta2, c2
    tr, mg, ms = IDX["train"], r["treino_grupo"], r["treino_sub"]
    print(f"[{a.nome}] treino grupo {len(tr)} + {int(mg.sum())} | subclasse {len(tr)} + {int(ms.sum())}", flush=True)
    Wg, bg = g["treinar"](np.vstack([X1[tr], X2[mg]]), np.vstack([YG[tr], r["Yg2"][mg]]))
    print(f"  grupo ({time.time() - t0:.0f}s)", flush=True)
    Ws, bs = g["treinar"](np.vstack([X1[tr], X2[ms]]), np.vstack([YS[tr], r["Ys2"][ms]]))
    print(f"  subclasse ({time.time() - t0:.0f}s)", flush=True)
    saida = Path(g["SAIDA"])
    np.savez_compressed(saida / f"modelo_{a.nome}_media_v1v2.npz", grupo_W=Wg, grupo_b=bg, subclasse_W=Ws, subclasse_b=bs,
                        grupos=np.array(g["GRUPOS"]), subclasses=np.array(g["SUBCLASSES"]))
    res = {}
    for parte in ("validation", "test"):
        og, os_ = g["decodificar"](X1[IDX[parte]], Wg, bg, Ws, bs)
        res[parte], pp = g["metricas"](og, os_, IDX[parte])
        if parte == "test":
            longo = g["TOKENS"][IDX["test"]] > 4096
            for rot, m in (("teste_curtos", ~longo), ("teste_longos", longo)):
                res[rot], _ = g["metricas"](og[m], os_[m], IDX["test"][m])
    t = res["test"]
    print(f"[{a.nome}] teste:", {k: round(v, 3) for k, v in t.items() if k != "n"}, flush=True)
    ids_teste = [IDS[i] for i in IDX["test"]]
    refs = {"Gemma v1 (1ª rodada)": np.array([g["universo"]["referencia_gemma2"]["por_pedido_teste"][n] for n in ids_teste]),
            "Gemma v1+v2 (produção)": r["ref_prod_pp"][[r["ref_prod_ids"].index(n) for n in ids_teste]]}
    res["comparacoes"] = {}
    for nome, B in refs.items():
        c = {}
        for j, col in enumerate(["grupo@1", "grupo@3", "subclasse@1", "subclasse@3"]):
            dif, lo, hi = bootstrap_pareado(pp[:, j], B[:, j]); c[col] = [round(dif, 4), round(lo, 4), round(hi, 4)]
            print(f"  {a.nome} v1+v2 − {nome} | {col:12s} {dif*100:+.1f} [{lo*100:+.1f}; {hi*100:+.1f}]")
        res["comparacoes"][nome] = c
    res["pp_teste"] = pp.tolist()
    json.dump(res, open(saida / f"resultados_{a.nome}_media_v1v2.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nome", required=True, help="rótulo do modelo nos arquivos de saída (ex.: qwen8b)")
    p.add_argument("--ta1", type=Path, required=True); p.add_argument("--c1", type=Path, required=True)
    p.add_argument("--ta2", type=Path, required=True); p.add_argument("--c2", type=Path, required=True)
    p.add_argument("--saida", type=Path, default=AQUI / "saida_v1v2")
    principal(p.parse_args())
