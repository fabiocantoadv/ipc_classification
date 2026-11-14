#!/usr/bin/env python3
"""Executa o classificador_ipc_qwen8b.ipynb fora do Jupyter, com outra configuração (célula 1).

Exemplo — comparar TA, C e a média TA + C do Qwen 4B na amostra 1:
    python consulta_gemma/qwen8b/executar_notebook.py "VETORES = {'q4_TA': Path('Embeddings/Amostra_1/Qwen_4B_a1/4B_TA'), \
        'q4_C': Path('Embeddings/Amostra_1/Qwen_4B_a1/4B_C'), \
        'q4_media': (Path('Embeddings/Amostra_1/Qwen_4B_a1/4B_TA'), Path('Embeddings/Amostra_1/Qwen_4B_a1/4B_C'))}; \
        SAIDA = Path('saida_4b')"
"""
import json, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
nb = json.load(open(AQUI / "classificador_ipc_qwen8b.ipynb", encoding="utf-8"))
config = sys.argv[1] if len(sys.argv) > 1 else ""
g = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    exec(compile(c["source"], f"celula_{i}", "exec"), g)
    if i == 1:
        g["PACOTE"] = AQUI
        exec(config, g)
