#!/usr/bin/env python3
"""Extrai das listas de concordância da IPC da OMPI (PDF "antigo → novo" de cada revisão) o mapa
grupo principal antigo → grupos principais novos, usado para atualizar os rótulos da amostra 2 (2014-2019) para a
IPC 2026.01 (consulta_gemma/train_v2.py, função atualizar_grupo).

Entrada: uma pasta com os PDFs convertidos em texto (pdftotext -layout arquivo.pdf arquivo.txt), um por revisão.
Saída: <pasta>/concordancia.json  {versão: {grupo_antigo: [grupos_novos]}}  (cópia em consulta_gemma/v2/
concordancia_ipc_grupos.json). Revisões usadas: 2018.01, 2019.01, 2020.01, 2021.01, 2023.01, 2024.01, 2025.01, 2026.01
(faltaram 2015.01, 2016.01, 2017.01 e 2022.01). Intervalos "X - Y" entram pelas duas pontas (aproximação: grupos
intermediários de um intervalo que atravessa grupos principais não são listados).

Uso:  python preparacao_dados/extrair_concordancia_ipc.py PASTA_COM_TXT
"""
import re, json, glob, collections, sys
CODE = re.compile(r'([A-H]\d\d[A-Z])\s+(\d+)/(\d+)(?:\s*-\s*(?:([A-H]\d\d[A-Z])\s+)?(\d+)/(\d+))?')
def parse(path):
    txt = open(path).read()
    ver = re.search(r'(\d{4}\.\d\d)', txt.splitlines()[0]).group(1)
    rows = []  # (old_code, [new main groups])
    cur = None
    for line in txt.splitlines():
        if not CODE.search(line) or 'INDEX' in line: continue
        m0 = re.match(r'^(\s*)([A-H]\d\d[A-Z]\s+\d+/\d+)', line)
        if m0 and len(m0.group(1)) < 20:
            old = m0.group(2).replace(' ', '')
            rest = line[m0.end():]
            cur = [old, set()]; rows.append(cur)
        else:
            rest = line
            if cur is None: continue
        for m in CODE.finditer(rest):
            sc = m.group(1); cur[1].add(f"{sc}{m.group(2)}/00")
            if m.group(5):
                sc2 = m.group(4) or sc
                for g in range(int(m.group(2)), int(m.group(5)) + 1):
                    if sc2 == sc: pass
                cur[1].add(f"{sc2}{m.group(5)}/00")
    # agrega por grupo principal antigo
    mp = collections.defaultdict(set)
    for old, new in rows:
        mp[old.split('/')[0] + '/00'] |= new
    return ver, mp, len(rows)
out = {}
for f in sorted(glob.glob(sys.argv[1] + '/*.txt')):
    ver, mp, n = parse(f)
    out[ver] = {k: sorted(v) for k, v in mp.items()}
    removed = [k for k, v in mp.items() if k not in v]
    print(ver, 'linhas', n, 'grupos antigos', len(mp), 'grupos que deixam de existir', len(removed), removed[:6])
json.dump(out, open(sys.argv[1] + '/concordancia.json', 'w'), indent=1)
