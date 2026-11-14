#!/usr/bin/env python3
"""
Monta o dataset de treino do classificador IPC (uma linha por pedido), unindo:
  - dataset_tai_reivindicacoes.csv  (extração de PDFs + título/resumo/IPC do Manticore)
  - patentes-2.csv                  (XML ST.36, RPIs 2014-2017)

Tratamentos:
  - ID normalizado em numero_base (12 dígitos; 7 nos PI antigos) e deduplicação por ele
  - reivindicações do XML: fragmentos (<25 caracteres ou iniciados em minúscula) juntados
    à anterior e numeração "1. ", "2. "... acrescentada
  - rótulos: ipc (lista), ipc_principal (1º código), subclasses (únicas, na ordem),
    subclasse_principal
  - quase-duplicatas (mesmo texto de reivindicações normalizado) agrupadas para não
    vazarem entre treino e teste
  - split 80/10/10 estratificado pela subclasse principal, por grupo; subclasses com menos
    de 3 grupos vão só para treino
"""
import csv, hashlib, json, random, re, sys, unicodedata
from collections import Counter, defaultdict
csv.field_size_limit(sys.maxsize)
SEED = 42
RE_IPC = re.compile(r"([A-H]\d{2}[A-Z])\s*(\d{1,4}/\d{1,6})")

def ipcs(txt):
    out = []
    for sc, g in RE_IPC.findall(txt or ""):
        c = f"{sc} {g}"
        if c not in out: out.append(c)
    return out

def norm_txt(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"(^|\n)\s*\d{1,3}\s*[.\-)]\s*", r"\1", t)
    return re.sub(r"\W+", " ", t).strip()

rows = {}
# 1) nossa extração (PDF)
ids_2019 = {r["pedido"] for r in csv.DictReader(open("reivindicacoes_2019.csv", encoding="utf-8"))}
for r in csv.DictReader(open("dataset_tai_reivindicacoes.csv", encoding="utf-8")):
    rows[r["numero_base"]] = {
        "numero_base": r["numero_base"], "pedido": r["pedido"],
        "origem": "pdf_2019" if r["pedido"] in ids_2019 else "pdf_bucket",
        "titulo": r["title"].strip(), "resumo": r["abstract"].strip(),
        "reivindicacoes": r["reivindicacoes"], "ipc": ipcs(r["ipc_txt"])}

# 2) XML (Douglas)
frag = dup = 0
for r in csv.DictReader(open("patentes-2.csv", encoding="utf-8-sig")):
    d = r["numero_pedido"][2:]
    base = d[:7] if len(d) == 8 else d
    if base in rows:
        dup += 1; continue
    cl = []
    for t in json.loads(r["reivindicacoes"]):
        t = " ".join(t.split())
        if not t: continue
        if cl and (len(t) < 25 or t[0].islower()):
            cl[-1] = cl[-1] + " " + t; frag += 1
        else:
            cl.append(t)
    rows[base] = {
        "numero_base": base, "pedido": r["numero_pedido"], "origem": "xml_rpi_2014_2017",
        "titulo": r["titulo"].strip(), "resumo": r["resumo"].strip(),
        "reivindicacoes": "\n".join(f"{i}. {t}" for i, t in enumerate(cl, 1)), "ipc": ipcs(r["ipc"])}

# 3) rótulos, filtros e grupos de quase-duplicatas
final = []
for x in rows.values():
    if not (x["titulo"] and x["resumo"] and x["reivindicacoes"] and x["ipc"]): continue
    x["n_reivindicacoes"] = x["reivindicacoes"].count("\n") + 1
    x["ipc_principal"] = x["ipc"][0]
    subs = []
    for c in x["ipc"]:
        if c[:4] not in subs: subs.append(c[:4])
    x["subclasses"] = subs; x["subclasse_principal"] = subs[0]
    x["grupo_dup"] = hashlib.md5(norm_txt(x["reivindicacoes"]).encode()).hexdigest()[:12]
    final.append(x)

grupos = defaultdict(list)
for x in final: grupos[x["grupo_dup"]].append(x)
# subclasse do grupo = a mais comum entre os membros
por_sub = defaultdict(list)
for g, xs in grupos.items():
    por_sub[Counter(x["subclasse_principal"] for x in xs).most_common(1)[0][0]].append(g)
rnd = random.Random(SEED); split_g = {}
for sub, gs in sorted(por_sub.items()):
    gs = sorted(gs); rnd.shuffle(gs)
    if len(gs) < 3:
        for g in gs: split_g[g] = "train"
        continue
    n = len(gs); nt = max(1, round(n * .1)); nv = max(1, round(n * .1))
    for i, g in enumerate(gs):
        split_g[g] = "test" if i < nt else ("val" if i < nt + nv else "train")
for x in final: x["split"] = split_g[x["grupo_dup"]]

campos = ["numero_base", "pedido", "origem", "split", "titulo", "resumo", "reivindicacoes",
          "n_reivindicacoes", "ipc", "ipc_principal", "subclasses", "subclasse_principal", "grupo_dup"]
final.sort(key=lambda x: (x["origem"], x["numero_base"]))
with open("dataset_ipc_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
    for x in final:
        y = dict(x); y["ipc"] = json.dumps(x["ipc"], ensure_ascii=False)
        y["subclasses"] = json.dumps(x["subclasses"], ensure_ascii=False)
        w.writerow({k: y[k] for k in campos})

# resumo
print(f"pedidos: {len(final)} | números repetidos descartados: {dup} | fragmentos juntados: {frag}")
print("por origem:", dict(Counter(x["origem"] for x in final)))
print("por split:", dict(Counter(x["split"] for x in final)))
print("origem x split:", {o: dict(Counter(x["split"] for x in final if x["origem"] == o)) for o in sorted({x["origem"] for x in final})})
gd = [g for g, xs in grupos.items() if len(xs) > 1]
print(f"grupos de quase-duplicatas: {len(gd)} ({sum(len(grupos[g]) for g in gd)} pedidos)")
sc = Counter(x["subclasse_principal"] for x in final)
print(f"subclasses principais: {len(sc)} | com >=50 docs: {sum(v>=50 for v in sc.values())} | <10 docs: {sum(v<10 for v in sc.values())} | <3: {sum(v<3 for v in sc.values())}")
print("top 10:", sc.most_common(10))
print("seções:", sorted(Counter(x['subclasse_principal'][0] for x in final).items()))
