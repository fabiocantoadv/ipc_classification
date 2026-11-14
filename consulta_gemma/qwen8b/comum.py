"""Funções compartilhadas pelos scripts desta pasta.

Os scripts reaproveitam as células 1 a 4 do notebook classificador_ipc_qwen8b.ipynb (configuração, rótulos e divisão
da amostra 1, leitura de vetores, treino, decodificação hierárquica e métricas), para que notebook e scripts usem
exatamente o mesmo código.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent


def carregar_notebook(saida: str | Path | None = None, pacote: Path = AQUI) -> dict:
    """Executa as células de código 1-4 do notebook e devolve o espaço de nomes (IDS, IDX, YG, YS, treinar, ...)."""
    nb = json.load(open(AQUI / "classificador_ipc_qwen8b.ipynb", encoding="utf-8"))
    g: dict = {}
    for i in (1, 2, 3, 4):
        exec(nb["cells"][i]["source"], g)
        if i == 1:
            g["PACOTE"] = Path(pacote)
            if saida is not None:
                g["SAIDA"] = Path(saida)
            g["SAIDA"].mkdir(parents=True, exist_ok=True)
    return g


def ler_vetores(pasta: Path) -> tuple[np.ndarray, list[str]]:
    """Lê embeddings_*.npy + pedidos*.csv de uma pasta (arquivo único ou blocos), na ordem dos arquivos."""
    pasta = Path(pasta)
    embs = sorted(pasta.glob("embeddings*.npy"))
    assert embs, f"nenhum embeddings*.npy em {pasta}"
    vetores, ids = [], []
    for e in embs:
        p = pasta / e.name.replace("embeddings", "pedidos").replace(".npy", ".csv")
        if not p.exists():
            cand = sorted(pasta.glob("pedidos*.csv"))
            assert len(embs) == 1 and len(cand) == 1, f"sem pedidos correspondente a {e.name}"
            p = cand[0]
        v = np.load(e)
        lidos = [r["numero_pedido"].strip() for r in csv.DictReader(open(p, encoding="utf-8-sig"))]
        assert len(lidos) == len(v), (e.name, len(lidos), len(v))
        vetores.append(v); ids += lidos
    return np.vstack(vetores), ids


def media_normalizada(*matrizes: np.ndarray) -> np.ndarray:
    """Média de vetores (ex.: TA e C) seguida de normalização L2 — o "TAC" usado em todo o projeto."""
    x = np.mean([m.astype(np.float32) for m in matrizes], axis=0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rotulos_amostra2(pacote: Path = AQUI) -> dict:
    """Rótulos da amostra 2 já no universo da amostra 1 (com concordância da CIB), máscaras de treino e referência
    do Gemma v1 + v2 de produção (acertos por pedido no teste). Gerado por consulta_gemma/train_v2.py (load_v2)."""
    r = np.load(Path(pacote) / "vetores/v2_rotulos_e_ref.npz", allow_pickle=True)
    return {
        "Yg2": np.unpackbits(r["Yg2"], axis=1)[:, :int(r["ng"])].astype(bool),
        "Ys2": np.unpackbits(r["Ys2"], axis=1)[:, :int(r["ns"])].astype(bool),
        "treino_grupo": r["treino_grupo"], "treino_sub": r["treino_sub"],
        # o .npz guardou numero_base sem zeros à esquerda (leitura como inteiro); a ordem é a do dataset_ipc_v2.csv
        "numero_base": [str(n).lstrip("0") for n in r["numero_base"]],
        "ref_prod_ids": list(r["ref_prod_ids"]), "ref_prod_pp": np.array(r["ref_prod_pp"]),
    }


def bootstrap_pareado(a: np.ndarray, b: np.ndarray, n: int = 2000, semente: int = 0) -> tuple[float, float, float]:
    """Diferença média a − b (acertos 0/1 por pedido) e intervalo de 95% por reamostragem pareada dos pedidos."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(semente)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
