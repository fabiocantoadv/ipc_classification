#!/usr/bin/env python3
"""TF-IDF TAC com a MESMA receita dos classificadores de embeddings (para comparação justa).

Representação: o vetorizador do baseline.py original (palavras + bigramas, sem acentos, min_df=3, max_df=0,98,
sublinear_tf, até 80 mil termos; texto = título + resumo + todas as reivindicações), ajustado só no treino.
Receita: logística um-contra-todos, Adam (lr 0,02), lotes de 1.024, 30 épocas, sem penalidade; modelos de grupo
e de subclasse; decodificação hierárquica; mesma divisão, universo de rótulos e métricas do
classificador_ipc_qwen8b.ipynb. v1v2 = treino da Amostra 1 + treino da Amostra 2 (mesmas máscaras do Qwen/Gemma).

Rodar no Mac, a partir de ~/Downloads/predicao_ipc:
    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/qwen8b/tfidf/treino_tfidf.py v1
    patenteia_search/semantic_api/.venv/bin/python consulta_gemma/qwen8b/tfidf/treino_tfidf.py v1v2
"""
import json, sys, time, gzip, csv, importlib.util
from pathlib import Path as _P
import numpy as np, torch, scipy.sparse as sp
VAR = sys.argv[1]   # "v1" ou "v1v2"
ROOT = _P(__file__).resolve().parents[3]                 # ~/Downloads/predicao_ipc
Q = ROOT / 'consulta_gemma/qwen8b'
BASE = str(Q); U = None
DIR_SAIDA = Q / 'tfidf'
nb = json.load(open(Q / 'classificador_ipc_qwen8b.ipynb')); g = {}
for i in (1, 2, 3, 4):
    exec(nb['cells'][i]['source'], g)
    if i == 1: exec(f"PACOTE=Path('{Q}')", g)
print('threads de CPU:', torch.get_num_threads(), flush=True)
spec = importlib.util.spec_from_file_location('bl', ROOT / 'baseline.py'); bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)
csv.field_size_limit(sys.maxsize)
IDS, IDX, YG, YS = g['IDS'], g['IDX'], g['YG'], g['YS']
v1 = {r['numero_pedido'].strip(): r for r in csv.DictReader(open(ROOT / 'patentes_xml_embeddings_ipc.csv', encoding='utf-8-sig'))}
T1 = [bl.make_text(v1[n], 'all_claims') for n in IDS]
tr_txt = [T1[i] for i in IDX['train']]
if VAR == 'v1v2':
    r = np.load(allow_pickle=True, file=Q / 'vetores/v2_rotulos_e_ref.npz')
    Yg2 = np.unpackbits(r['Yg2'], axis=1)[:, :int(r['ng'])].astype(bool); Ys2 = np.unpackbits(r['Ys2'], axis=1)[:, :int(r['ns'])].astype(bool)
    mg, ms = r['treino_grupo'], r['treino_sub']
    v2 = list(csv.DictReader(gzip.open(Q / 'colab_v2_4b/dataset_ipc_v2_textos.csv.gz', 'rt', encoding='utf-8-sig')))
    assert [x['numero_base'].lstrip('0') for x in v2] == [str(n).lstrip('0') for n in r['numero_base']]   # npz perdeu zeros à esquerda; ordem idêntica
    T2 = [bl.make_text(x, 'all_claims') for x in v2]
    tr_txt = tr_txt + [T2[i] for i in np.where(mg | ms)[0]]
t0 = time.time()
vec = bl.TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.98,
                         max_features=80000, sublinear_tf=True, dtype=np.float32)
vec.fit(tr_txt); del tr_txt
X1 = vec.transform(T1).tocsr().astype(np.float32)
print(f'[{VAR}] vocabulário {len(vec.vocabulary_)} ({time.time()-t0:.0f}s)', flush=True)
if VAR == 'v1v2':
    X2 = vec.transform(T2).tocsr().astype(np.float32); del T2

def treinar_esparso(X, Y, EPOCAS=30, LOTE=1024, LR=0.02, SEMENTE=0):
    Y = torch.as_tensor(Y, dtype=torch.float32); K, dim = Y.shape[1], X.shape[1]
    W = torch.zeros((K, dim)); prior = Y.mean(0).clamp(1e-4, 1 - 1e-4); c = torch.log(prior / (1 - prior))
    mW, vW, mc, vc = torch.zeros_like(W), torch.zeros_like(W), torch.zeros_like(c), torch.zeros_like(c)
    rng = np.random.default_rng(SEMENTE); n = X.shape[0]; t = 0
    for ep in range(EPOCAS):
        for idx in np.array_split(rng.permutation(n), max(1, n // LOTE)):
            t += 1
            xb = X[idx].tocoo(); xt = torch.sparse_coo_tensor(np.vstack([xb.row, xb.col]), xb.data, xb.shape).coalesce()
            err = (torch.sigmoid(torch.sparse.mm(xt, W.T) + c) - Y[idx]) / len(idx)
            gW = torch.sparse.mm(xt.t(), err).T            # (K, dim), denso
            gc = err.sum(0); b1, b2 = 1 - 0.9 ** t, 1 - 0.999 ** t
            mW.mul_(0.9).add_(gW, alpha=0.1); vW.mul_(0.999).addcmul_(gW, gW, value=0.001); del gW
            W.addcdiv_(mW, (vW / b2).sqrt_().add_(1e-8), value=-LR / b1)
            mc.mul_(0.9).add_(gc, alpha=0.1); vc.mul_(0.999).addcmul_(gc, gc, value=0.001)
            c.addcdiv_(mc, (vc / b2).sqrt_().add_(1e-8), value=-LR / b1)
        print(f'  época {ep+1}/{EPOCAS} ({time.time()-t0:.0f}s)', flush=True)
    return W.numpy(), c.numpy()

tr = IDX['train']
if VAR == 'v1':
    Xg, Yg_ = X1[tr], YG[tr]; Xs, Ys_ = X1[tr], YS[tr]
else:
    Xg, Yg_ = sp.vstack([X1[tr], X2[mg]]).tocsr(), np.vstack([YG[tr], Yg2[mg]])
    Xs, Ys_ = sp.vstack([X1[tr], X2[ms]]).tocsr(), np.vstack([YS[tr], Ys2[ms]])
print(f'treino grupo {Xg.shape[0]} | subclasse {Xs.shape[0]}', flush=True)
SAIDA_PESOS = {'sub': f'pesos_tfidf_{VAR}_subclasse.npz', 'grupo': f'pesos_tfidf_{VAR}_grupo.npz'}
def treinar_ou_carregar(nome, X, Y):
    caminho = DIR_SAIDA / SAIDA_PESOS[nome]
    if caminho.exists():                       # retomada: pesos já treinados numa execução anterior
        with np.load(caminho) as z: print(f'{nome}: pesos carregados de {caminho.name}', flush=True); return z['W'], z['b']
    W, b = treinar_esparso(X, Y); np.savez(caminho, W=W, b=b); print(f'{nome} ok (pesos gravados)', flush=True); return W, b
Ws, bs = treinar_ou_carregar('sub', Xs, Ys_)
Wg, bg = treinar_ou_carregar('grupo', Xg, Yg_)

def decodificar_esparso(X, Wg, bg, Ws, bs):
    """Igual à decodificar() do notebook, aceitando matriz esparsa."""
    ls = g['log_sigmoide']
    nota = ls(np.asarray(X @ Wg.T) + bg) + ls(np.asarray(X @ Ws.T) + bs)[:, g['PAI']]
    melhor = np.full((X.shape[0], len(g['SUBCLASSES'])), -np.inf)
    for k, f in enumerate(g['FILHOS']):
        if len(f): melhor[:, k] = nota[:, f].max(1)
    return np.argsort(-nota, 1)[:, :5], np.argsort(-melhor, 1)[:, :3]
res = {}
for parte in ('validation', 'test'):
    og, os_ = decodificar_esparso(X1[IDX[parte]], Wg, bg, Ws, bs)
    res[parte], pp = g['metricas'](og, os_, IDX[parte])
res['pp_teste'] = pp.tolist()
json.dump(res, open(DIR_SAIDA / f'resultado_{VAR}.json', 'w'))
t = res['test']; print(f'[{VAR}] TESTE', {k: round(v, 3) for k, v in t.items() if k != 'n'}, f'({time.time()-t0:.0f}s)', flush=True)
