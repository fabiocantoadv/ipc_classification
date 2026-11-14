# TF-IDF — classificador IPC (linha de base)

Representação: TF-IDF de palavras e bigramas sobre título + resumo + todas as reivindicações
(baseline.py, make_text "all_claims"): strip_accents="unicode", lowercase, ngram (1, 2), min_df=3, max_df=0,98,
sublinear_tf, até 80.000 termos, L2 por linha. Vetores esparsos (80 mil dimensões), não embeddings densos.

## Pastas
- `vetorizadores/` — vetorizador ajustado em cada versão de treino: `vetorizador_v1.joblib` e `vetorizador_v1v2.joblib`
  (scikit-learn 1.9.1, ambiente patenteia_search/semantic_api/.venv). Cópia portátil, independente de versão:
  `*_vocabulario.json` (termo → coluna, parâmetros) e `*_idf.npy`.
- `vetores/vetorizador_v1/epoca1/` e `vetores/vetorizador_v1v2/{epoca1,epoca2}/` — matrizes esparsas
  (`matriz.npz`, abrir com `scipy.sparse.load_npz`) + `pedidos.csv` (mesma ordem das linhas) + `metadata.json`.
  1ª época = todas as 20.697 linhas de patentes_xml_embeddings_ipc.csv; 2ª época = 25.563 linhas de dataset_ipc_v2.csv.
  Use a matriz do mesmo vetorizador do modelo.
- `modelos_svc/` — OneVsRest LinearSVC(C=0,5, class_weight="balanced"), grupo (2.585) e subclasse (575);
  `svc_v1_*` (treino 1ª época) e `svc_v1v2_*` (treino 1ª + 2ª). Calibração de Platt (a, b por nível) em
  `resultados/resultado_svc_*.json` → `calibracao`.
- `resultados/` — métricas no teste de 2026 (2.918 pedidos) e acertos por pedido (`pp_teste`).
  `resultado_svc_*`: melhor receita (LinearSVC + hierarquia). `resultado_v1*.json`: receita dos embeddings (Adam), inferior.
  `tfidf_linha_de_base_original_por_pedido.json`: linha de base original (só grupo, sem hierarquia).
- `receita_adam/` — pesos da tentativa com a receita dos embeddings (logística Adam, λ=0); inferior, pode ser apagada.
- `scripts/` — treino_tfidf_svc.py (melhor receita), treino_tfidf.py (receita Adam), gerar_vetores_tfidf.py.

## Resultados (teste de 2026, grupo Hit@1 / Hit@3 / subclasse Hit@1 / Hit@3)
- LinearSVC hierárquico, só v1: 0,495 / 0,663 / 0,659 / 0,811
- LinearSVC hierárquico, v1 + v2: 0,514 / 0,693 / 0,675 / 0,836
- Linha de base original (só grupo): 0,487 / 0,655 / —
- Comparação: Gemma v1 + v2 0,551 / 0,721 / 0,718 / 0,878; Qwen 8B v1 + v2 0,595 / 0,756 / 0,742 / 0,892

## Classificar um texto novo
    vec = joblib.load('vetorizadores/vetorizador_v1v2.joblib'); X = vec.transform([texto])   # texto = make_text(...)
    Sg = a_g * svc_grupo.decision_function(X) + b_g;  Ss = a_s * svc_sub.decision_function(X) + b_s
    nota_grupo = log σ(Sg) + log σ(Ss)[:, subclasse-mãe]
