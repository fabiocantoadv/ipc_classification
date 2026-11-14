#!/usr/bin/env python3
"""TF-IDF TAC com a receita ORIGINAL (LinearSVC), mais subclasse e decodificação hierárquica.

Representação e classificador idênticos ao baseline.py (linha de base inicial):
  TfidfVectorizer(strip_accents="unicode", lowercase, ngram (1, 2), min_df=3, max_df=0.98, max_features=80000,
                  sublinear_tf) sobre título + resumo + todas as reivindicações (make_text "all_claims");
  OneVsRestClassifier(LinearSVC(C=0.5, class_weight="balanced", max_iter=2000)).
Acréscimos para comparar com os embeddings nas mesmas condições:
  - modelo de subclasse (575) além do de grupo (2.585), mesmo universo de rótulos;
  - calibração de Platt na validação (p = σ(a·margem + b), um par a, b por nível), para somar os níveis;
  - decodificação hierárquica igual à da API: nota do grupo = log p(grupo) + log p(subclasse-mãe);
  - mesmas métricas e mesmo teste (2.918 pedidos de 2026) do classificador_ipc_qwen8b.ipynb;
  - comparação pareada (bootstrap) com o Gemma v1 e o Gemma v1 + v2 de produção.
Também registra a variante sem calibração (σ das margens brutas).

Rodar no Mac, a partir de ~/Downloads/predicao_ipc:
    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/qwen8b/tfidf/treino_tfidf_svc.py v1
    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/qwen8b/tfidf/treino_tfidf_svc.py v1v2
"""
import json, sys, time, gzip, csv, importlib.util, warnings
from pathlib import Path as _P
import numpy as np, scipy.sparse as sp, joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

VAR = sys.argv[1]   # "v1" ou "v1v2"
assert VAR in ("v1", "v1v2")
ROOT = _P(__file__).resolve().parents[3]                 # ~/Downloads/predicao_ipc
Q = ROOT / 'consulta_gemma/qwen8b'; DIR_SAIDA = Q / 'tfidf'
t0 = time.time()

# funções e dados do notebook de avaliação (divisão, universo de rótulos, métricas)
nb = json.load(open(Q / 'classificador_ipc_qwen8b.ipynb')); g = {}
for i in (1, 2, 3, 4):
    exec(nb['cells'][i]['source'], g)
    if i == 1: exec(f"PACOTE=Path('{Q}')", g)
IDS, IDX, YG, YS = g['IDS'], g['IDX'], g['YG'], g['YS']
spec = importlib.util.spec_from_file_location('bl', ROOT / 'baseline.py'); bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
csv.field_size_limit(sys.maxsize)

v1 = {r['numero_pedido'].strip(): r for r in csv.DictReader(open(ROOT / 'patentes_xml_embeddings_ipc.csv', encoding='utf-8-sig'))}
T1 = [bl.make_text(v1[n], 'all_claims') for n in IDS]
tr = IDX['train']; tr_txt = [T1[i] for i in tr]
r = np.load(Q / 'vetores/v2_rotulos_e_ref.npz', allow_pickle=True)
if VAR == 'v1v2':
    Yg2 = np.unpackbits(r['Yg2'], axis=1)[:, :int(r['ng'])].astype(bool); Ys2 = np.unpackbits(r['Ys2'], axis=1)[:, :int(r['ns'])].astype(bool)
    mg, ms = r['treino_grupo'], r['treino_sub']
    v2 = list(csv.DictReader(gzip.open(Q / 'colab_v2_4b/dataset_ipc_v2_textos.csv.gz', 'rt', encoding='utf-8-sig')))
    assert [x['numero_base'].lstrip('0') for x in v2] == [str(n).lstrip('0') for n in r['numero_base']]  # npz perdeu zeros à esquerda; ordem idêntica
    T2 = [bl.make_text(x, 'all_claims') for x in v2]
    tr_txt = tr_txt + [T2[i] for i in np.where(mg | ms)[0]]

vec = bl.TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.98,
                         max_features=80000, sublinear_tf=True, dtype=np.float32)
vec.fit(tr_txt); del tr_txt
X1 = vec.transform(T1).tocsr()
if VAR == 'v1v2':
    X2 = vec.transform(T2).tocsr(); del T2
print(f'[{VAR}] vocabulário {len(vec.vocabulary_)} ({time.time()-t0:.0f}s)', flush=True)

if VAR == 'v1':
    Xg, Yg_, Xs, Ys_ = X1[tr], YG[tr], X1[tr], YS[tr]
else:
    Xg, Yg_ = sp.vstack([X1[tr], X2[mg]]).tocsr(), np.vstack([YG[tr], Yg2[mg]])
    Xs, Ys_ = sp.vstack([X1[tr], X2[ms]]).tocsr(), np.vstack([YS[tr], Ys2[ms]])
print(f'treino grupo {Xg.shape[0]} | subclasse {Xs.shape[0]}', flush=True)

def treinar_ou_carregar(nome, X, Y):
    caminho = DIR_SAIDA / f'svc_{VAR}_{nome}.joblib'
    if caminho.exists():
        print(f'{nome}: modelo carregado de {caminho.name}', flush=True); return joblib.load(caminho)
    m = OneVsRestClassifier(LinearSVC(C=0.5, dual="auto", class_weight="balanced", max_iter=2000), n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, Y)
    joblib.dump(m, caminho); print(f'{nome} ok ({time.time()-t0:.0f}s, modelo gravado)', flush=True); return m

ms_ = treinar_ou_carregar('subclasse', Xs, Ys_)
mg_ = treinar_ou_carregar('grupo', Xg, Yg_)

def margens(m, X): return np.asarray(m.decision_function(X), dtype=np.float32)
M = {p: (margens(mg_, X1[IDX[p]]), margens(ms_, X1[IDX[p]])) for p in ('validation', 'test')}

def platt(s, y):
    """a, b de p = σ(a·s + b), ajustados na validação (todas as células documento × rótulo)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y.reshape(-1).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])
ag, bg = platt(M['validation'][0], YG[IDX['validation']]); as_, bs = platt(M['validation'][1], YS[IDX['validation']])
print(f'calibração: grupo a={ag:.3f} b={bg:.3f} | subclasse a={as_:.3f} b={bs:.3f}', flush=True)

ls = g['log_sigmoide']
def decodificar(Sg, Ss, cal=True):
    if cal: Sg, Ss = ag * Sg + bg, as_ * Ss + bs
    nota = ls(Sg) + ls(Ss)[:, g['PAI']]
    melhor = np.full((len(nota), len(g['SUBCLASSES'])), -np.inf)
    for k, f in enumerate(g['FILHOS']):
        if len(f): melhor[:, k] = nota[:, f].max(1)
    return np.argsort(-nota, 1)[:, :5], np.argsort(-melhor, 1)[:, :3]

res = {'receita': 'TF-IDF (baseline.py) + OvR LinearSVC C=0,5 balanced; Platt na validação; decodificação hierárquica',
       'variante': VAR, 'calibracao': {'grupo': [ag, bg], 'subclasse': [as_, bs]}}
for cal in (True, False):
    chave = 'calibrado' if cal else 'sem_calibracao'; res[chave] = {}
    for p in ('validation', 'test'):
        og, os_ = decodificar(*M[p], cal=cal); res[chave][p], pp = g['metricas'](og, os_, IDX[p])
        if p == 'test': res[chave]['pp_teste'] = pp.tolist()
    t = res[chave]['test']
    print(f'[{VAR}] {chave:15s} TESTE grupo@1={t["grupo_hit@1"]:.3f} grupo@3={t["grupo_hit@3"]:.3f} subcl@1={t["subclasse_hit@1"]:.3f} '
          f'subcl@3={t["subclasse_hit@3"]:.3f} raros@5={t["recall@5_raros"]:.3f}', flush=True)

# comparação pareada com o Gemma (mesmo teste, pedido a pedido)
A = np.array(res['calibrado']['pp_teste']); ids = [IDS[i] for i in IDX['test']]
refs = {'Gemma v1': np.array([g['universo']['referencia_gemma2']['por_pedido_teste'][n] for n in ids]),
        'Gemma v1+v2 (produção)': np.array(r['ref_prod_pp'])[[list(r['ref_prod_ids']).index(n) for n in ids]]}
rng = np.random.default_rng(0); n = len(A); S = [rng.integers(0, n, n) for _ in range(2000)]; res['comparacoes'] = {}
for nome, B in refs.items():
    c = {}
    for j, col in enumerate(['grupo@1', 'grupo@3', 'subclasse@1', 'subclasse@3']):
        d = A[:, j].astype(float) - B[:, j]; b = np.array([d[s].mean() for s in S]); lo, hi = np.percentile(b, [2.5, 97.5])
        c[col] = [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]
    res['comparacoes'][f'TF-IDF {VAR} − {nome}'] = c
    print(f'TF-IDF {VAR} − {nome}: ' + ' | '.join(f'{k} {v[0]*100:+.1f} [{v[1]*100:+.1f}; {v[2]*100:+.1f}]' for k, v in c.items()), flush=True)
json.dump(res, open(DIR_SAIDA / f'resultado_svc_{VAR}.json', 'w'), ensure_ascii=False)
print(f'gravado {DIR_SAIDA / f"resultado_svc_{VAR}.json"} ({time.time()-t0:.0f}s)')
