#!/usr/bin/env python3
"""Codifica o dataset IPC v2 com o EmbeddingGemma 2 no formato exato dos índices do v1.

Rodar no Mac com o ambiente da API semântica (o modelo está no cache dela):
    cd ~/Downloads/predicao_ipc
    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/encode_v2.py

Formatos (conferidos em conferencia_formato_documento.json, cosseno ~1,0 com o índice):
    TA     -> "title: {titulo} | text: {resumo}"
    Claims -> "title: none | text: {reivindicações}"  (reivindicações separadas por linha em branco, como no v1)
prompt="" (sem prompt do modelo), normalizado, max_seq_length 8192, mesma revisão da API.

Grava em blocos de --chunk pedidos em consulta_gemma/v2/blocos/ e pode ser interrompido (Ctrl+C) e
retomado: blocos prontos são pulados. No fim junta tudo em consulta_gemma/v2/v2_gemma2_{ta,claims}.npy,
na ordem das linhas do CSV, e grava v2_ids.csv (numero_base) para conferência.
A cada --pausa-cada blocos (padrão 5) faz uma pausa de --pausa-seg segundos (padrão 180) para o
Mac resfriar. Os textos são ordenados por tamanho dentro de cada bloco para aproveitar melhor os lotes.
"""
from __future__ import annotations
import argparse, csv, gc, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "consulta_gemma"))
from encode_query_vectors import load_model  # mesma revisão e configuração da API

DEFAULT_CSV = Path.home() / "Downloads/Recorte_imagens/claims_extractor/dataset_ipc_v2.csv"
OUT = ROOT / "consulta_gemma/v2"


def claims_text(s: str) -> str:
    linhas = [l.strip() for l in (s or "").splitlines() if l.strip()]
    return "\n\n".join(linhas)


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def encode(model, texts, batch_size):
    order = np.argsort([-len(t) for t in texts])
    out = np.zeros((len(texts), 768), np.float32)
    for i in range(0, len(texts), batch_size):
        sel = order[i:i + batch_size]
        bs = 1 if len(texts[sel[0]]) > 12000 else batch_size  # textos longos um por vez (memória do MPS)
        v = model.encode([texts[j] for j in sel], prompt="", convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False, batch_size=bs)
        out[sel] = np.asarray(v, np.float32)
        if len(texts[sel[0]]) > 12000:
            free_memory()
    free_memory()
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main(a):
    csv.field_size_limit(sys.maxsize)
    with a.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    blocos = OUT / "blocos"; blocos.mkdir(parents=True, exist_ok=True)
    model, device = load_model(a.device)
    print(f"{n} pedidos; dispositivo {device}; blocos de {a.chunk}", flush=True)
    t0 = time.monotonic(); feitos = 0; blocos_feitos = 0
    for scope in a.scopes:
        for start in range(0, n, a.chunk):
            path = blocos / f"{scope}_{start:06d}.npy"
            if path.exists():
                continue
            part = rows[start:start + a.chunk]
            if scope == "ta":
                texts = [f"title: {r['titulo'].strip()} | text: {r['resumo'].strip()}" for r in part]
            else:
                texts = [f"title: none | text: {claims_text(r['reivindicacoes'])}" for r in part]
            np.save(path, encode(model, texts, a.batch_size))
            feitos += len(part)
            el = time.monotonic() - t0
            print(f"[{scope}] {min(start + a.chunk, n)}/{n}  ({feitos / el:.1f} pedidos/s, {el / 60:.0f} min)", flush=True)
            blocos_feitos += 1
            if a.pausa_cada and blocos_feitos % a.pausa_cada == 0 and start + a.chunk < n:
                print(f"  pausa de {a.pausa_seg} s para resfriar...", flush=True)
                free_memory(); time.sleep(a.pausa_seg)
    for scope in a.scopes:
        partes = [np.load(blocos / f"{scope}_{s:06d}.npy") for s in range(0, n, a.chunk)]
        X = np.vstack(partes); assert X.shape == (n, 768), X.shape
        np.save(OUT / f"v2_gemma2_{scope}.npy", X)
        print(f"gravado v2_gemma2_{scope}.npy {X.shape}")
    with (OUT / "v2_ids.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["linha", "numero_base"])
        for i, r in enumerate(rows):
            w.writerow([i, r["numero_base"]])
    print("pronto.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--scopes", nargs="+", choices=("ta", "claims"), default=["ta", "claims"])
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pausa-cada", type=int, default=5, help="pausa a cada N blocos codificados (0 = sem pausa)")
    p.add_argument("--pausa-seg", type=int, default=180, help="duração da pausa em segundos")
    p.add_argument("--device", help="cpu, mps ou cuda (padrão: o mesmo da API)")
    main(p.parse_args())
