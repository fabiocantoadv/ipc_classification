#!/usr/bin/env python3
"""Gera e grava, na pasta TF-IDF, os vetorizadores e as matrizes TF-IDF das duas amostras.

Os vetorizadores são reajustados exatamente como no treino dos modelos (treino_tfidf_svc.py): mesmos textos de treino
(título + resumo + todas as reivindicações), mesmos parâmetros. O ajuste é determinístico, então reproduz o vocabulário
e os IDF usados pelos modelos em modelos_svc/. No fim, confere: aplica o vetorizador v1v2 + os modelos svc_v1v2 ao
teste e compara com os acertos gravados em resultados/resultado_svc_v1v2.json (devem ser idênticos).

Rodar no Mac, a partir de ~/Downloads/predicao_ipc (≈3-5 min):
    patenteia_search/semantic_api/.venv/bin/python TF-IDF/scripts/gerar_vetores_tfidf.py
"""
import json, sys, time, gzip, csv, importlib.util
from pathlib import Path
import numpy as np, scipy.sparse as sp, joblib, sklearn

ROOT = Path(__file__).resolve().parents[2]; TF = ROOT / 'TF-IDF'; Q = ROOT / 'consulta_gemma/qwen8b'
t0 = time.time()
nb = json.load(open(Q / 'classificador_ipc_qwen8b.ipynb')); g = {}
for i in (1, 2, 3, 4):
    exec(nb['cells'][i]['source'], g)
    if i == 1: exec(f"PACOTE=Path('{Q}')", g)
IDS, IDX = g['IDS'], g['IDX']
spec = importlib.util.spec_from_file_location('bl', ROOT / 'baseline.py'); bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
csv.field_size_limit(sys.maxsize)

# textos: Amostra 1 na ordem do CSV completo (20.697), Amostra 2 na ordem do dataset_ipc_v2 (25.563)
linhas1 = list(csv.DictReader(open(ROOT / 'patentes_xml_embeddings_ipc.csv', encoding='utf-8-sig')))
ids1 = [r['numero_pedido'].strip() for r in linhas1]; T1 = [bl.make_text(r, 'all_claims') for r in linhas1]
pos1 = {n: i for i, n in enumerate(ids1)}
v2 = list(csv.DictReader(gzip.open(Q / 'colab_v2_4b/dataset_ipc_v2_textos.csv.gz', 'rt', encoding='utf-8-sig')))
ids2 = [x['numero_base'] for x in v2]; T2 = [bl.make_text(x, 'all_claims') for x in v2]
r = np.load(Q / 'vetores/v2_rotulos_e_ref.npz', allow_pickle=True)
assert [n.lstrip('0') for n in ids2] == [str(n).lstrip('0') for n in r['numero_base']]
mg, ms = r['treino_grupo'], r['treino_sub']
treino1 = [T1[pos1[IDS[i]]] for i in IDX['train']]
print(f'textos prontos ({time.time()-t0:.0f}s)', flush=True)

def vetorizador():
    return bl.TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.98,
                              max_features=80000, sublinear_tf=True, dtype=np.float32)

def gravar_vetorizador(vec, nome):
    vec.stop_words_ = None                       # conjunto de termos descartados: grande e desnecessário para transformar
    d = TF / 'vetorizadores'; d.mkdir(exist_ok=True)
    joblib.dump(vec, d / f'vetorizador_{nome}.joblib')
    # cópia portátil (não depende da versão do scikit-learn): vocabulário, IDF e parâmetros
    voc = {t: int(i) for t, i in vec.vocabulary_.items()}
    json.dump({'parametros': {k: (v if isinstance(v, (int, float, str, bool, type(None), list, tuple)) else str(v))
                              for k, v in vec.get_params().items()},
               'scikit_learn': sklearn.__version__, 'vocabulario': voc},
              open(d / f'vetorizador_{nome}_vocabulario.json', 'w'), ensure_ascii=False)
    np.save(d / f'vetorizador_{nome}_idf.npy', vec.idf_.astype(np.float32))

def gravar_matriz(X, ids, pasta, origem):
    pasta.mkdir(parents=True, exist_ok=True)
    sp.save_npz(pasta / 'matriz.npz', X.tocsr().astype(np.float32), compressed=True)
    (pasta / 'pedidos.csv').write_text('numero_pedido\n' + '\n'.join(ids) + '\n', encoding='utf-8')
    json.dump({'linhas': X.shape[0], 'dimensao': X.shape[1], 'nao_zeros': int(X.nnz), 'origem': origem,
               'normalizacao': 'L2 por linha (padrão do TfidfVectorizer)'}, open(pasta / 'metadata.json', 'w'), ensure_ascii=False, indent=1)
    print(f'  {pasta.relative_to(TF)}: {X.shape}, {X.nnz:,} não zeros ({time.time()-t0:.0f}s)', flush=True)

# vetorizador v1 (usado pelos modelos svc_v1_*): só o treino da Amostra 1
vec1 = vetorizador().fit(treino1); gravar_vetorizador(vec1, 'v1')
gravar_matriz(vec1.transform(T1), ids1, TF / 'vetores/vetorizador_v1/amostra1', 'patentes_xml_embeddings_ipc.csv (todas as linhas)')
del vec1
# vetorizador v1v2 (usado pelos modelos svc_v1v2_*): treino da Amostra 1 + treino da Amostra 2
vec = vetorizador().fit(treino1 + [T2[i] for i in np.where(mg | ms)[0]]); gravar_vetorizador(vec, 'v1v2')
X1 = vec.transform(T1); X2 = vec.transform(T2)
gravar_matriz(X1, ids1, TF / 'vetores/vetorizador_v1v2/amostra1', 'patentes_xml_embeddings_ipc.csv (todas as linhas)')
gravar_matriz(X2, ids2, TF / 'vetores/vetorizador_v1v2/amostra2', 'dataset_ipc_v2.csv (todas as linhas)')

# conferência: vetorizador regerado + modelos gravados reproduzem os acertos do teste
res = json.load(open(TF / 'resultados/resultado_svc_v1v2.json'))
(ag, bg), (as_, bs) = res['calibracao']['grupo'], res['calibracao']['subclasse']
mgr = joblib.load(TF / 'modelos_svc/svc_v1v2_grupo.joblib'); msu = joblib.load(TF / 'modelos_svc/svc_v1v2_subclasse.joblib')
Xt = X1[[pos1[IDS[i]] for i in IDX['test']]]
Sg = ag * np.asarray(mgr.decision_function(Xt)) + bg; Ss = as_ * np.asarray(msu.decision_function(Xt)) + bs
ls = g['log_sigmoide']; nota = ls(Sg) + ls(Ss)[:, g['PAI']]
melhor = np.full((len(nota), len(g['SUBCLASSES'])), -np.inf)
for k, f in enumerate(g['FILHOS']):
    if len(f): melhor[:, k] = nota[:, f].max(1)
_, pp = g['metricas'](np.argsort(-nota, 1)[:, :5], np.argsort(-melhor, 1)[:, :3], IDX['test'])
iguais = np.array_equal(pp, np.array(res['calibrado']['pp_teste']))
print(f'conferência com resultado_svc_v1v2.json: {"IDÊNTICO" if iguais else "DIFERENTE"} '
      f'(grupo@1 {pp[:, 0].mean():.3f}) ({time.time()-t0:.0f}s)')
