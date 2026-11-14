#!/usr/bin/env python3
"""Gera vetores Gemma 2 de CONSULTA (prompt SearchQuery) para os 20.697 pedidos, por escopo.

Rodar no Mac com o ambiente da API semântica, que já tem o EmbeddingGemma 2 no cache:

    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/encode_query_vectors.py

O vetor é gerado exatamente como _encode_semantic_query() da API (mesma revisão, SearchQuery,
normalização). O texto imita o que um usuário digitaria no campo livre de cada escopo:
  tr   -> título + resumo
  r    -> primeira reivindicação
  tr_r -> título + resumo + primeira reivindicação, num único texto (um único vetor)
Saída: consulta_gemma/vetores/query_<escopo>.npy (float32, 20.697 x 768, ordem do CSV) e um JSON
com os parâmetros. O progresso é salvo em blocos; rodar de novo retoma de onde parou."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baseline import split_claims  # mesma segmentação de reivindicações do projeto

GEMMA_ID = "google/embeddinggemma-2"
GEMMA_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"  # igual a _get_gemma_text_model()
SCOPES = ("tr", "r", "tr_r")


def query_text(row: dict[str, str], scope: str, max_chars: int) -> str:
    title, abstract = row["titulo"].strip(), row["resumo"].strip()
    claims = split_claims(row["reivindicacoes"] or "")
    first = claims[0] if claims else (row["reivindicacoes"] or "").strip()
    if scope == "tr":
        text = f"{title}\n{abstract}"
    elif scope == "r":
        text = first
    else:
        text = f"{title}\n{abstract}\n{first}"
    return text.strip()[:max_chars]


def load_model(device: str | None):
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(GEMMA_ID, revision=GEMMA_REVISION, device=device, local_files_only=True,
                                config_kwargs={"vision_config": None, "audio_config": None})
    model.max_seq_length = 8192
    return model, device


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(texts, prompt_name="SearchQuery", convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=batch_size)
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def main(a):
    csv.field_size_limit(sys.maxsize)
    with a.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    a.output.mkdir(parents=True, exist_ok=True)
    model, device = load_model(a.device)
    print(f"{len(rows)} pedidos; dispositivo {device}", flush=True)
    for scope in a.scopes:
        final = a.output / f"query_{scope}.npy"
        if final.exists():
            print(f"[{scope}] já existe: {final}", flush=True)
            continue
        texts = [query_text(r, scope, a.max_chars) for r in rows]
        parts = a.output / f"partes_{scope}"; parts.mkdir(exist_ok=True)
        order = np.argsort([len(t) for t in texts])  # lotes de tamanho parecido aceleram o MPS
        start = time.monotonic()
        for c, s in enumerate(range(0, len(order), a.chunk)):
            part = parts / f"{c:04d}.npy"
            if part.exists():
                continue
            idx = order[s:s + a.chunk]
            np.save(part, encode(model, [texts[i] or " " for i in idx], a.batch_size))
            done = min(s + a.chunk, len(order))
            print(f"[{scope}] {done}/{len(order)}  {time.monotonic() - start:.0f}s", flush=True)
        X = np.zeros((len(rows), 768), dtype=np.float32)
        for c, s in enumerate(range(0, len(order), a.chunk)):
            X[order[s:s + a.chunk]] = np.load(parts / f"{c:04d}.npy")
        if not np.isfinite(X).all() or X.shape[1] != 768:
            raise ValueError(f"Matriz inválida em {scope}")
        np.save(final, X)
        meta = {"scope": scope, "model": GEMMA_ID, "revision": GEMMA_REVISION, "prompt_name": "SearchQuery",
                "normalize": True, "max_seq_length": 8192, "max_chars": a.max_chars, "rows": len(rows),
                "input_csv": a.input.name, "text_rule": {"tr": "titulo\\nresumo", "r": "primeira reivindicação",
                                                        "tr_r": "titulo\\nresumo\\nprimeira reivindicação"}[scope],
                "sha256_vectors": hashlib.sha256(X.tobytes()).hexdigest(),
                "sample_texts": [texts[i][:300] for i in range(3)]}
        (a.output / f"query_{scope}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{scope}] salvo {final} em {time.monotonic() - start:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "patentes_xml_embeddings_ipc.csv")
    p.add_argument("--output", type=Path, default=ROOT / "consulta_gemma/vetores")
    p.add_argument("--scopes", nargs="+", choices=SCOPES, default=list(SCOPES))
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--device")
    main(p.parse_args())
