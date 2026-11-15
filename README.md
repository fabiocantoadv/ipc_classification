# Classificador IPC de pedidos de patente (PatenteIA — IBICT / INPI)

Classificador que sugere, para um pedido de patente brasileiro, os **grupos principais** e as **subclasses** da
Classificação Internacional de Patentes (IPC) mais prováveis, a partir de **embeddings** do título, do resumo e das
reivindicações. Este repositório reúne os dados de rótulos, os scripts, os notebooks e os resultados da comparação entre
**Gemma 2 (EmbeddingGemma)**, **Qwen3-Embedding 0.6B / 4B / 8B**, **BGE-M3** e uma linha de base **TF-IDF**, todos com o mesmo
**classificador hierárquico**.

> Resultado principal (teste com 2.918 pedidos de 2026): o **Qwen3-Embedding-8B**, com a média dos vetores TA e C e treino
> nas duas amostras, acerta o **grupo principal em 59,5%** dos pedidos na 1ª sugestão (75,6% entre as 3 primeiras) e a
> **subclasse em 74,2%** (89,2% entre as 3 primeiras) — **+4,4 pontos** sobre o Gemma 2 em produção no grupo Hit@1
> (IC 95%: +3,0 a +5,9).
>
> Na **prova independente** (7.888 pedidos inéditos de 2024–2025, só título e resumo), os Qwen 8B e 4B acertam o grupo
> principal em 53% dos pedidos na 1ª sugestão e em 79% entre as 5 primeiras, cerca de 4,4 pontos acima do Gemma 2.

---

## Sumário
1. [O problema](#1-o-problema)
2. [Dados](#2-dados)
3. [Universo de rótulos e concordância da IPC](#3-universo-de-rótulos-e-concordância-da-ipc)
4. [Divisão treino / validação / teste](#4-divisão-treino--validação--teste)
5. [Textos e embeddings](#5-textos-e-embeddings)
6. [O classificador hierárquico](#6-o-classificador-hierárquico)
7. [Métricas e comparação estatística](#7-métricas-e-comparação-estatística)
8. [Resultados](#8-resultados)
9. [Linha de base TF-IDF](#9-linha-de-base-tf-idf)
10. [Experimentos complementares](#10-experimentos-complementares)
11. [Decisões, cuidados e limitações](#11-decisões-cuidados-e-limitações)
12. [Embeddings no Google Drive](#12-embeddings-no-google-drive)
13. [Mapa do repositório](#13-mapa-do-repositório)
14. [Como reproduzir](#14-como-reproduzir)

---

## 1. O problema

Cada pedido recebe do examinador uma ou mais classificações IPC (por exemplo `A61K 31/496`). A hierarquia é:

| Nível | Exemplo | Quantos no nosso universo |
|---|---|---|
| Seção | `A` | 8 |
| Classe | `A61` | — |
| **Subclasse** | `A61K` | **575** |
| **Grupo principal** | `A61K 31/00` | **2.585** |
| Subgrupo | `A61K 31/496` | (não previsto) |

O classificador prevê **dois níveis**: subclasse e grupo principal. O problema é **multirrótulo** — um pedido pode ter
várias IPC, e uma sugestão conta como acerto se coincidir com **qualquer** IPC verdadeira do pedido.

O classificador é usado na busca semântica do PatenteIA (API `patenteia_search`, ambiente `pi-homol`) para sugerir
classificações e para reordenar resultados.

## 2. Dados

Os textos completos dos pedidos **não estão no repositório** (são extraídos da base do INPI); aqui estão os rótulos, os
identificadores, os índices de divisão e os resultados. Os embeddings estão no Google Drive (seção 12).

### Amostra 1 (PoC 1, pedidos de 2026)
- Origem: XML dos pedidos publicados em 2026 na base do INPI (amostra da PoC 1).
- **20.697 pedidos**, dos quais **19.411 têm IPC** utilizável no universo de rótulos.
- Campos: título, resumo, reivindicações e IPC.
- Rótulos e divisão: `consulta_gemma/qwen8b/poc1_rotulos.csv.gz` (número do pedido, IPC, partição, número de tokens).
- Observação: nas reivindicações há marcações de imagem do tipo `[Imagem: imgf0001.tif]` (fórmulas químicas e tabelas
  convertidas em figura) — 1.883 pedidos, 51% deles da classe C07. Ver seção 11.

### Amostra 2 (`dataset_ipc_v2`)
- Montada por `preparacao_dados/montar_dataset_ipc.py` a partir de pedidos com IPC de outros anos de depósito.
- **25.563 pedidos**, identificados por `numero_base`.
- Classificações atualizadas para a IPC 2026.01 pela concordância da OMPI (seção 3).
- Rótulos compactados, máscaras de treino e referência por pedido do Gemma em produção:
  `consulta_gemma/qwen8b/vetores/v2_rotulos_e_ref.npz` (gerado pelo `consulta_gemma/train_v2.py`).
- Não há sobreposição com a amostra 1 (a amostra 1 é só de 2026).

## 3. Universo de rótulos e concordância da IPC

- O **universo é fixado pela amostra 1**: grupos principais com suporte ≥ 2 no conjunto (**2.585**) e as subclasses
  correspondentes (**575**). Está em `consulta_gemma/qwen8b/poc1_universo.json` (também com a referência por pedido do
  Gemma v1 no teste).
- Rótulos da amostra 2 fora do universo são descartados; pedidos que ficam sem nenhum rótulo saem do treino.
- **Concordância**: as IPC antigas da amostra 2 são atualizadas pelas listas de concordância "antigo → novo" da OMPI
  (revisões 2018.01 a 2026.01; faltaram 2015.01, 2016.01, 2017.01 e 2022.01). O extrator é
  `preparacao_dados/extrair_concordancia_ipc.py` e o mapa resultante, `consulta_gemma/v2/concordancia_ipc_grupos.json`.
  **Só correspondências 1:1 são aplicadas**; quando um grupo antigo se divide em vários, o rótulo original é mantido
  (evita inventar rótulos).

## 4. Divisão treino / validação / teste

| Conjunto | Treino | Validação | Teste | Como |
|---|---|---|---|---|
| Amostra 1 | 13.565 | 2.928 | **2.918** | `GroupShuffleSplit` por hash de título+resumo (duplicados ficam juntos), semente 42 |
| Amostra 2 | 80% | 10% | 10% | estratificado pela subclasse principal, agrupando duplicados |

Terminologia: **Amostra 1 / Amostra 2** são os conjuntos de dados; **1ª rodada (v1)** e **2ª rodada (v1+v2)** são os
treinos; **época** é usada só no sentido de aprendizado de máquina (uma passada completa do otimizador pelo treino — a
receita usa 30); **conjunto de prova** é a avaliação independente (pedidos que nenhum modelo viu).

- **Treino v1**: só o treino da amostra 1.
- **Treino v1+v2**: treino da amostra 1 + treino da amostra 2 → 13.565 + **18.629** (modelo de grupo) e 13.565 +
  **20.035** (modelo de subclasse), após o filtro de universo e a **exclusão das 15 consultas do benchmark de busca**
  (para não contaminar a avaliação da busca).
- **Validação e teste são sempre os da amostra 1** (2026), para que todas as linhas da tabela sejam comparáveis.
- Papéis: o **treino** ajusta os pesos; a **validação** serve para escolher configurações (ex.: o peso α da seção 10);
  o **teste** é usado uma vez, para o número final.

## 5. Textos e embeddings

Três formas de texto por pedido:

| Sigla | Conteúdo |
|---|---|
| **TA** | título + resumo |
| **C** | reivindicações |
| **TAC** | no classificador: **média dos vetores TA e C, renormalizada (L2)** |

O "TAC" de referência do projeto é a média de dois vetores (assim foi feito no Gemma 2). Também testamos o TAC como
**texto concatenado** num vetor só (Qwen 8B antigo, corte em 4.096 tokens): é pior, sobretudo nos pedidos longos.

### Gemma 2 — `google/embeddinggemma-2` (revisão `f9972ac22e696a105c482202646979e33e4b5a1e`, 768 dimensões)
- TA: `title: {titulo} | text: {resumo}`; C: `title: none | text: {reivindicações}`; `prompt=""`; até 8.192 tokens.
- Scripts: `consulta_gemma/encode_v2.py`, `consulta_gemma/encode_query_vectors.py`.

### Qwen3-Embedding — 0.6B (1.024 dim.), 4B (2.560 dim.) e 8B (4.096 dim.)
- Notebook único (Colab A100 ou H200): `consulta_gemma/qwen8b/embeddings_qwen3.ipynb`.
- TA: `Título: {titulo}\nResumo: {resumo}`; C: `Reivindicações: {reivindicacoes}`.
- Pooling pelo **último token**, padding à esquerda, bf16, vetores L2-normalizados, até **8.192 tokens**.
- **Descoberta importante — o token final**: o Qwen3 acrescenta `<|endoftext|>` (id 151643) e usa o vetor desse último
  token. O corte ingênuo `tokens[:MAX]` removia esse token **justamente nos pedidos truncados**, que passavam a ter
  vetores quase aleatórios: no TAC antigo do 8B (corte em 4.096), o grupo Hit@1 dos truncados caía para **0,358**
  (contra 0,554 dos demais). O notebook final preserva o token no corte.
- Lotes por número de tokens, ordenados por tamanho, com divisão automática do lote se faltar memória e retomada por
  blocos.

### BGE-M3 — `BAAI/bge-m3` (revisão `5617a9f61b028005a4858fdac845db406aefb181`, 1.024 dimensões)
- Gerado no Mac (FlagEmbedding, modo híbrido), para a busca: TA = título + resumo com rótulos; C = reivindicações sem
  rótulo; até **4.096 tokens**; vetores densos L2-normalizados + pesos esparsos (léxicos) por pedido.
- Na busca híbrida, nota = 0,7 · cosseno denso + 0,3 · produto escalar esparso. **No classificador entra só o vetor
  denso.**

## 6. O classificador hierárquico

### O modelo: regressão logística um-contra-todos sobre embeddings congelados

O classificador é uma **regressão logística multirrótulo no esquema um-contra-todos** (*one-vs-rest*), treinada sobre
os embeddings. Os modelos de embeddings (Qwen3, Gemma 2, BGE-M3) **não são ajustados**: funcionam só como extratores
de vetores, e quem aprende é uma camada linear por cima deles (o que a literatura chama de *linear probe*).

- **Um contra todos:** cada rótulo (cada grupo principal, cada subclasse) tem o seu próprio classificador binário,
  "é este rótulo / não é", com uma sigmoide própria. As probabilidades são independentes, e por isso o mesmo pedido
  pode receber várias IPC (problema multirrótulo). É o mesmo princípio do `OneVsRestClassifier` do scikit-learn
  (documentação: [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
  e [guia de classificação multiclasse e multirrótulo](https://scikit-learn.org/stable/modules/multiclass.html)); aqui,
  todos os rótulos de um nível são treinados juntos como uma única camada linear, em PyTorch.
- **Dois classificadores independentes**, um por nível da IPC:
  - **grupo principal:** 2.585 saídas;
  - **subclasse:** 575 saídas.
- **Forma:** para um vetor `x` (L2-normalizado), a nota de cada rótulo é `z = W·x + b` e a probabilidade é `σ(z)`.
  Cada classificador é só uma matriz de pesos `W` (rótulos × dimensão do embedding) e um vetor de viés `b`:

  | Embedding | Dimensão | Parâmetros (grupo + subclasse) |
  |---|---|---|
  | Gemma 2 | 768 | 2,4 milhões |
  | Qwen3 0.6B, BGE-M3 | 1.024 | 3,2 milhões |
  | Qwen3 4B | 2.560 | 8,1 milhões |
  | Qwen3 8B | 4.096 | 12,9 milhões |

### Treino

Mesma receita para todos os embeddings (notebook `consulta_gemma/qwen8b/classificador_ipc_qwen8b.ipynb` e
`treino_v1v2_qwen.py`):

1. **Entrada:** embedding L2-normalizado — no modelo final, a média dos vetores TA e Claims, renormalizada.
2. **Perda:** entropia cruzada binária por rótulo (BCE; equivalente a
   [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) do PyTorch).
3. **Otimização:** [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) (lr 0,02; β 0,9 / 0,999),
   **lotes de 1.024**, **30 épocas** (passadas completas pelo treino), **sem regularização (λ = 0)**, semente 0.
4. **Viés inicial** pelo log-odds da frequência de cada rótulo no treino (os rótulos raros começam com probabilidade
   baixa, o que estabiliza o início do treino).
5. Os hiperparâmetros foram fixados na 1ª rodada e mantidos em todos os modelos, para que a comparação entre
   embeddings seja justa.

### Decodificação hierárquica

Os dois níveis são combinados na hora de sugerir as classificações:
- nota do grupo = `log σ(grupo) + log σ(subclasse-mãe)` — um grupo só fica bem colocado se a sua subclasse também for
  provável;
- as subclasses são ordenadas pelo **melhor grupo** que contêm.

Assim, a lista de grupos e a de subclasses são **100% coerentes** (o grupo sugerido sempre pertence à subclasse
sugerida). Em relação à decodificação plana, o efeito na precisão é pequeno e não significativo (+0,3 a +0,6 ponto
no grupo, −0,5 a −0,7 na subclasse); o ganho é a coerência.

### Classificar um pedido novo

```python
import numpy as np
m = np.load("consulta_gemma/qwen8b/modelos/modelo_qwen8b_media_v1v2.npz")
x = (v_ta + v_claims) / 2; x /= np.linalg.norm(x)            # vetores do mesmo modelo de embeddings
zg = m["grupo_W"] @ x + m["grupo_b"]; zs = m["subclasse_W"] @ x + m["subclasse_b"]
log_sig = lambda z: -np.logaddexp(0, -z)
pai = [list(m["subclasses"]).index(g[:4]) for g in m["grupos"]]   # subclasse-mãe de cada grupo
nota = log_sig(zg) + log_sig(zs)[pai]
top5 = [m["grupos"][i] for i in np.argsort(-nota)[:5]]
```

Os arquivos de pesos `modelo_{qwen06b,qwen4b,qwen8b,bgem3}_media_v1v2.npz` trazem `grupo_W`, `grupo_b`,
`subclasse_W`, `subclasse_b`, `grupos` e `subclasses` (os códigos na ordem das linhas de `W`). O Gemma 2 de
produção é exportado para a API por `consulta_gemma/export_v2.py`, no formato `coef` / `intercept` / `classes`.

### O classificador do TF-IDF (linha de base)

Para o TF-IDF a receita é outra, mais adequada a vetores esparsos de 80 mil dimensões:
- [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) (máquina de vetores de
  suporte linear; C = 0,5, `class_weight="balanced"`) dentro de um
  [`OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html),
  um por nível (≈ 253 milhões de parâmetros no total);
- o SVM devolve margens, não probabilidades: elas são convertidas por **calibração de Platt** — `p = σ(a·margem + b)`,
  um par `a, b` por nível, ajustado na validação (ver o
  [guia de calibração de probabilidades](https://scikit-learn.org/stable/modules/calibration.html));
- depois, a mesma decodificação hierárquica.

A receita logística dos embeddings aplicada ao TF-IDF fica bem abaixo (seção 9).

## 7. Métricas e comparação estatística

- **Grupo Hit@k / Subclasse Hit@k**: proporção de pedidos em que ao menos uma das *k* primeiras sugestões é uma IPC
  verdadeira do pedido. Leitura: "subclasse Hit@1 = 0,742 e Hit@3 = 0,892" significa que o modelo acerta a subclasse
  na 1ª sugestão em cerca de 74% dos pedidos e tem a subclasse certa entre as 3 sugeridas em cerca de 89%.
- **Recall@5 raros / frequentes**: recall dos rótulos verdadeiros nas 5 primeiras sugestões de grupo, separando grupos
  raros (menos de 10 exemplos no treino) e frequentes.
- **Curtos / longos**: longos = pedidos com título+resumo+reivindicações acima de 4.096 tokens (330 no teste).
- **Bootstrap pareado**: diferença entre dois modelos nos **mesmos pedidos**, com 2.000 reamostras; reportamos a média e
  o intervalo de 95%. Se o intervalo não inclui zero, a diferença é estatisticamente consistente.

## 8. Resultados

Teste da amostra 1 (2.918 pedidos de 2026). Todos com o classificador hierárquico da seção 6; "média" = média TA+C.

| Embedding | Texto | Treino | Grupo@1 | Grupo@3 | Grupo@5 | Sub@1 | Sub@3 | Raros@5 | Longos g@1 |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC | TAC (texto) | v1 | 0,495 | 0,663 | 0,711 | 0,659 | 0,811 | 0,252 | — |
| TF-IDF + LinearSVC | TAC (texto) | v1+v2 | 0,514 | 0,693 | 0,742 | 0,675 | 0,836 | 0,314 | — |
| Gemma 2 (300M) | média | v1 | 0,532 | 0,699 | 0,760 | 0,701 | 0,863 | — | — |
| **Gemma 2 (produção)** | média | v1+v2 | 0,551 | 0,721 | — | 0,718 | 0,878 | — | — |
| BGE-M3 | TA | v1 | 0,472 | 0,626 | — | 0,640 | 0,807 | 0,225 | 0,612 |
| BGE-M3 | C | v1 | 0,459 | 0,611 | — | 0,620 | 0,787 | 0,208 | 0,594 |
| BGE-M3 | média | v1 | 0,495 | 0,645 | — | 0,653 | 0,813 | 0,241 | 0,624 |
| BGE-M3 | média | v1+v2 | 0,510 | 0,670 | 0,734 | 0,673 | 0,830 | 0,316 | — |
| Qwen3 0.6B | TA | v1 | 0,503 | 0,668 | 0,729 | 0,664 | 0,837 | 0,267 | 0,636 |
| Qwen3 0.6B | C | v1 | 0,507 | 0,672 | 0,730 | 0,670 | 0,828 | 0,269 | 0,667 |
| Qwen3 0.6B | média | v1 | 0,526 | 0,694 | 0,745 | 0,687 | 0,849 | 0,290 | 0,682 |
| Qwen3 0.6B | média | v1+v2 | 0,542 | 0,706 | 0,764 | 0,700 | 0,861 | 0,361 | 0,715 |
| Qwen3 4B | TA | v1 | 0,542 | 0,725 | 0,775 | 0,706 | 0,871 | 0,337 | 0,639 |
| Qwen3 4B | C | v1 | 0,552 | 0,719 | 0,775 | 0,705 | 0,862 | 0,323 | 0,715 |
| Qwen3 4B | média | v1 | 0,558 | 0,731 | 0,791 | 0,718 | 0,875 | 0,361 | 0,667 |
| Qwen3 4B | média | v1+v2 | 0,573 | 0,746 | 0,800 | 0,726 | 0,887 | 0,414 | 0,712 |
| Qwen3 8B | TAC concatenado, 4.096 (antigo) | v1 | 0,532 | 0,702 | 0,759 | 0,687 | 0,850 | 0,331 | — |
| Qwen3 8B | TA | v1 | 0,556 | 0,725 | 0,785 | 0,717 | 0,873 | 0,341 | 0,694 |
| Qwen3 8B | C | v1 | 0,571 | 0,735 | 0,798 | 0,723 | 0,883 | 0,357 | 0,712 |
| Qwen3 8B | média | v1 | 0,576 | 0,744 | 0,802 | 0,735 | 0,885 | 0,370 | 0,721 |
| **Qwen3 8B** | **média** | **v1+v2** | **0,595** | **0,756** | **0,812** | **0,742** | **0,892** | **0,429** | **0,730** |

Comparações pareadas (bootstrap, grupo Hit@1, em pontos percentuais):

| Comparação | Diferença | IC 95% |
|---|---|---|
| Qwen 8B v1+v2 − Gemma 2 produção | +4,4 | [+3,0; +5,9] |
| Qwen 8B v1+v2 − Gemma 2 v1 | +6,4 | [+4,8; +7,9] |
| Qwen 8B v1+v2 − Qwen 4B v1+v2 | +2,3 | [+1,1; +3,4] |
| Qwen 8B v1+v2 − TF-IDF SVC v1+v2 | +8,1 | [+6,5; +9,8] |
| BGE-M3 v1+v2 − Gemma 2 produção | −4,1 | [−5,5; −2,7] |
| BGE-M3 v1+v2 − TF-IDF SVC v1+v2 | −0,4 | [−2,0; +1,2] |

Leituras:
- **Modelo maior ajuda**: 0.6B < Gemma 2 ≈ 0.6B v1+v2 < 4B < 8B.
- **O BGE-M3 empata com o TF-IDF** (−0,4 ponto no grupo@1, não significativo) e fica 4 pontos abaixo do Gemma 2. Ao
  contrário dos outros modelos, nele C sozinho é pior que TA — possivelmente pelo corte em 4.096 tokens.
- **A média TA+C supera TA e C isolados** em todos os modelos; C sozinho supera TA nos maiores.
- **Juntar a amostra 2 no treino ajuda sempre** (+1,5 a +2 pontos no grupo@1; ganhos maiores nos rótulos raros).
- **Pedidos longos são mais fáceis** (reivindicações extensas e técnicas), desde que o token final seja preservado.
- Custo: Qwen 8B ≈ 16 GB de GPU para gerar vetores; 4B ≈ 8 GB; Gemma 2 (300M) roda em CPU/Mac.

### Prova independente — pedidos inéditos de 2024 e 2025

Os modelos treinados nas Amostras 1 + 2 foram avaliados uma única vez num **conjunto de prova** que nenhum modelo viu
em nenhuma etapa (treino, validação ou teste): **7.888 pedidos inéditos** de três amostras aleatórias da base de
patentes — publicação 2024–2025 (4.835), depósito 2024 (2.424) e depósito 2025 (629). Foram excluídos todos os pedidos
das Amostras 1 e 2 (pelo número do pedido ou pelo texto: mesmo título e resumo, ou mesmo resumo) e as repetições
entre as próprias amostras; 7.479 pedidos têm ao menos um grupo no universo do classificador.

- **Notebook (Colab A100):** <https://colab.research.google.com/drive/13i6TGSs-HripSv-7xQ3_zL_6IoKYOLHz?usp=sharing>
- **Resultados completos:** `consulta_gemma/qwen8b/resultados/resultados_prova_v4.json`
- **Entrada só TA:** a prova tem apenas título e resumo, que entram com o vetor TA em classificadores treinados com a
  média TA + Claims (na parte da Amostra 1 repetida na prova, isso custou cerca de 2,5 pontos no grupo@1 do Qwen 8B).
  Os números são, portanto, conservadores.
- **Precisão@5** = fração das 5 sugestões de grupo que é verdadeira (os pedidos têm em média 3 IPC, então o máximo
  fica bem abaixo de 1); **Revocação@5** = fração dos grupos verdadeiros do pedido que aparece entre as 5.

Total da prova (7.479 pedidos no grupo; 7.877 na subclasse):

| Modelo | Grupo@1 | Grupo@3 | Grupo@5 | Precisão@5 | Revocação@5 | Sub@1 | Sub@3 |
|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | **0,532** | 0,721 | **0,791** | **0,247** | **0,626** | **0,688** | **0,862** |
| Qwen3-4B | 0,531 | **0,723** | **0,791** | **0,247** | **0,626** | 0,685 | 0,859 |
| Gemma 2 (produção) | 0,488 | 0,686 | 0,755 | 0,231 | 0,587 | 0,656 | 0,837 |
| Qwen3-0.6B | 0,479 | 0,670 | 0,734 | 0,224 | 0,567 | 0,632 | 0,812 |
| TF-IDF + LinearSVC | 0,455 | 0,636 | 0,698 | 0,214 | 0,535 | 0,602 | 0,788 |
| BGE-M3 | 0,448 | 0,635 | 0,709 | 0,212 | 0,535 | 0,607 | 0,790 |

Por amostra (grupo@1 / grupo@5; entre parênteses, pedidos com grupo no universo):

| Modelo | Publicação 2024-25 (4.667) | Depósito 2024 (2.236) | Depósito 2025 (576) |
|---|---|---|---|
| Qwen3-8B | 0,571 / 0,821 | 0,468 / 0,746 | 0,460 / 0,729 |
| Qwen3-4B | 0,576 / 0,822 | 0,458 / 0,741 | 0,460 / 0,733 |
| Gemma 2 (produção) | 0,525 / 0,789 | 0,432 / 0,697 | 0,411 / 0,707 |
| Qwen3-0.6B | 0,522 / 0,775 | 0,412 / 0,667 | 0,398 / 0,663 |
| TF-IDF + LinearSVC | 0,507 / 0,742 | 0,369 / 0,627 | 0,366 / 0,622 |
| BGE-M3 | 0,496 / 0,751 | 0,371 / 0,643 | 0,356 / 0,628 |

Leituras da prova:
- **A ordem dos modelos se mantém** em relação ao teste de 2026: Qwen 4B/8B > Gemma 2 > Qwen 0.6B > TF-IDF ≈ BGE-M3.
  Os dois Qwen maiores ficam cerca de **4,4 pontos acima do Gemma 2 de produção** no grupo@1.
- **Com entrada só TA, o Qwen 8B empata com o 4B** (diferença de 0,1 ponto), ao contrário do teste de 2026 com TA + C
  (+2,3 pontos para o 8B). A vantagem do 8B parece vir das reivindicações.
- **As amostras por depósito são mais difíceis** que a de publicação (cerca de 10 pontos a menos no grupo@1, em todos
  os modelos); o motivo ainda não foi investigado.
- Os números absolutos ficam abaixo do teste de 2026 (Qwen 8B: 0,532 contra 0,595 no grupo@1), coerente com a
  entrada só TA e com pedidos de outra distribuição.

## 9. Linha de base TF-IDF

Pasta `TF-IDF/` (detalhes em `TF-IDF/LEIAME.md`):
- Texto: título + resumo + todas as reivindicações (`baseline.py`, `make_text`).
- Vetorizador: palavras e bigramas, `strip_accents="unicode"`, `min_df=3`, `max_df=0,98`, `sublinear_tf`, até 80.000
  termos, L2 por linha.
- **Melhor receita** (`TF-IDF/scripts/treino_tfidf_svc.py`): `LinearSVC(C=0,5, class_weight="balanced")` um-contra-todos
  + calibração de Platt + a mesma decodificação hierárquica.
- **Mesma receita dos embeddings** (`TF-IDF/scripts/treino_tfidf.py`, Adam, 30 épocas): fica bem abaixo
  (0,397 / 0,429 no grupo@1, v1 / v1+v2) — a receita logística foi pensada para vetores densos normalizados.
- Linha de base original do início do projeto: 0,487 / 0,655 (grupo@1 / grupo@3), em
  `TF-IDF/resultados/tfidf_linha_de_base_original_por_pedido.json`.
- Os vetorizadores e as matrizes TF-IDF estão no Drive (`Embeddings/TF-IDF/`).

## 10. Experimentos complementares

- **Peso por frequência (pos_weight)** — inspirado em `github.com/rcgsheffield/ipc-classifier`
  (`consulta_gemma/qwen8b/teste_pos_weight.py`). Com o Gemma 2 v1: o peso inverso à frequência original (α = 1) piora;
  atenuado (α = 0,25, escolhido na validação) dá **+0,9 ponto no grupo@1 [+0,3; +1,5]**. Ainda não aplicado aos Qwen.
- **Decodificação hierárquica × plana**: ver seção 6.
- **Simulação de reranking** da busca com o classificador (no repositório da API).

## 11. Decisões, cuidados e limitações

- **Benchmark de busca fora do treino**: 5 das 15 consultas do benchmark apareciam no treino da amostra 2; foram
  retiradas e os modelos v1+v2 retreinados. (O exemplo-semente do benchmark novo do Edson está no treino da amostra 1 —
  atenção ao ampliar o benchmark.)
- **Marcações de imagem nas reivindicações** (`[Imagem: imgf….tif]`, só na amostra 1): são ruído para o embedding,
  concentradas em química (C07). A limpeza está preparada, mas ainda não foi aplicada aos vetores.
- **Até 5 IPC por pedido no XML** da amostra 1: pedidos com mais classificações podem ter rótulos incompletos, o que
  penaliza um pouco as métricas.
- **O teste foi consultado várias vezes** ao longo da comparação (sempre o mesmo conjunto de 2.918 pedidos). As
  diferenças grandes são robustas; o número sem viés vem da **prova independente** com pedidos inéditos de 2024–2025
  (seção 8), feita uma única vez.
- **Concordância só 1:1**: rótulos antigos que se dividiram em vários grupos novos permanecem como estavam.
- O "TAC" de referência é sempre a **média de vetores** TA e C; o TAC concatenado aparece só como comparação.

## 12. Embeddings no Google Drive

Pasta: <https://drive.google.com/drive/folders/1Kcd-IbHtDNbuP2B8m8cNt2wH3Ob-fKLy?usp=sharing>

```
Embeddings/
├── Amostra_1/                          # Amostra 1 — PoC 1, pedidos de 2026, 20.697 linhas (numero_pedido)
│   ├── BGE_M3_hybrid_a1/
│   │   ├── C/                          # BGE_M3_hybrid_a1_C.tar.zst + config.json + resultado.json
│   │   ├── TA/                         # BGE_M3_hybrid_a1_TA.tar.zst + config.json + resultado.json
│   │   └── BGE_M3_hybrid_a1_manifest.json
│   ├── EmbeddingGemma_2_a1/
│   │   ├── Claims/                     # EmbeddingGemma_2_a1_Claims.tar.zst + config.json + resultado.json
│   │   ├── TA/                         # EmbeddingGemma_2_a1_TA.tar.zst + config.json + resultado.json
│   │   └── EmbeddingGemma_2_a1_manifest.json
│   ├── Qwen_0.6B_a1/{06_C, 06_TA, 06_TAC}
│   ├── Qwen_4B_a1/{4B_C, 4B_TA, TAC}
│   ├── Qwen_8B_a1/
│   │   ├── 8B_C/, 8B_TA/
│   │   └── 8B_TAC/                     # embeddings_TAC.npy + pedidos.csv + metadata.json (arquivo único)
│   └── metadados_TAC_amostra_1.csv     # metadados dos pedidos da amostra
├── Amostra_2/                          # Amostra 2 — dataset_ipc_v2, 25.563 linhas (numero_base)
│   ├── BGE_M3_hybrid_a2/
│   │   ├── C/                          # BGE_M3_hybrid_a2_C_blocos_00000-00099.tar.zst e _00100-00199.tar.zst
│   │   ├── TA/                         # BGE_M3_hybrid_a2_TA_blocos_00000-00099.tar.zst e _00100-00199.tar.zst
│   │   └── BGE_M3_hybrid_a2_manifest.json
│   ├── EmbeddingGemma_2_a2/
│   │   ├── Claims/                     # v2_gemma2_claims.npy + ids.csv + config.json + resultado.json
│   │   ├── TA/                         # v2_gemma2_ta.npy + ids.csv + config.json + resultado.json
│   │   └── EmbeddingGemma_2_a2_manifest.json
│   ├── Qwen_0.6B_a2/{0.6B_C, 0.6B_TA}
│   ├── Qwen_4B_a2/{4B_C, 4B_TA}
│   ├── Qwen_8B_a2/{C, TA}
│   └── metadados_TAC_amostra_2.csv
└── TF-IDF/
    ├── modelos_svc/                    # 4 modelos LinearSVC (.joblib): grupo e subclasse, v1 e v1v2
    ├── receita_adam/                   # 4 arquivos de pesos da receita logística/Adam aplicada ao TF-IDF
    ├── resultados/                     # 5 arquivos .json
    ├── scripts/                        # 3 scripts .py
    ├── vetores/
    │   ├── vetorizador_v1/amostra1/    # matriz.npz + metadata.json + pedidos.csv
    │   └── vetorizador_v1v2/{amostra1, amostra2}/
    ├── vetorizadores/                  # vetorizador_v1 e v1v2: .joblib + vocabulário .json + idf .npy
    └── LEIAME.md
```

**Formatos**
- **Qwen3** (saída do `embeddings_qwen3.ipynb`), em blocos: `embeddings_00000.npy` (L2-normalizado),
  `pedidos_00000.csv` (identificador, mesma ordem), `bloco_00000.json`, `config.json` (modelo, versão, limite de tokens,
  SHA-256 do CSV de entrada) e `resultado.json`. As pastas `*TAC` (texto concatenado, corte antigo em 4.096 tokens)
  são só referência: o TAC do classificador é a média dos vetores TA e C.
- **BGE-M3** e **Gemma 2 da Amostra 1**: os blocos estão compactados em `.tar.zst` (o `manifest.json` traz o SHA-256,
  o número de arquivos, de linhas e de blocos). Para extrair:
  ```bash
  tar --zstd -xf BGE_M3_hybrid_a1_TA.tar.zst        # ou: zstd -dc ARQUIVO.tar.zst | tar -x
  ```
  Os blocos do BGE-M3 trazem `embeddings_*.npy` (denso, 1.024 dim.), `sparse_*.npz` (pesos léxicos) e `ids_*.csv`.
- **Gemma 2 da Amostra 2**: arquivo único por escopo (`v2_gemma2_ta.npy`, `v2_gemma2_claims.npy`, 768 dim.) com
  `ids.csv` na mesma ordem do `dataset_ipc_v2.csv`.
- No Gemma, o escopo das reivindicações se chama `Claims`; nos demais modelos, `C`.

## 13. Mapa do repositório

```
baseline.py                      leitura do CSV, rótulos, divisão da amostra 1, montagem do texto, TF-IDF de referência
embedding_experiment.py          preparação dos alvos (rótulos multirrótulo) usada pelos scripts do Gemma
preparacao_dados/
  montar_dataset_ipc.py          montagem da amostra 2 (dataset_ipc_v2)
  extrair_concordancia_ipc.py    extrator das listas de concordância da OMPI (PDF → texto → JSON)
consulta_gemma/                  pipeline Gemma 2 (produção da API)
  encode_query_vectors.py        geração de vetores Gemma 2 (carregamento do modelo)
  encode_v2.py                   vetores Gemma 2 TA/C da amostra 2
  hierarquia_piloto.py           primeiro piloto da decodificação hierárquica
  train_query_classifier.py      classificador plano original (referência)
  train_v2.py                    treino v1 / v2 / v1+v2, concordância, exclusão do benchmark
  export_v2.py, export_retrained.py   exportação dos pesos para a API
  v2/concordancia_ipc_grupos.json
  qwen8b/
    embeddings_qwen3.ipynb       notebook FINAL de embeddings (Qwen3 0.6B/4B/8B; TA, C, TAC; amostras 1 e 2)
    classificador_ipc_qwen8b.ipynb   treino e avaliação hierárquica (amostra 1) + bootstrap contra o Gemma
    comum.py                     funções compartilhadas pelos scripts
    treino_v1v2_qwen.py          treino v1+v2 sobre a média TA+C de um Qwen (gera modelos/ e resultados/)
    executar_notebook.py         executa o notebook de classificação fora do Jupyter, com outra configuração
    teste_pos_weight.py          experimento de peso por frequência
    poc1_rotulos.csv.gz, poc1_universo.json, vetores/v2_rotulos_e_ref.npz
    modelos/                     pesos v1+v2 (média TA+C): Qwen 0.6B, 4B, 8B e BGE-M3
    resultados/                  métricas e acertos por pedido de cada experimento; resultados_prova_v4.json = prova independente
TF-IDF/                          linha de base TF-IDF (scripts, resultados, LEIAME.md)
```

## 14. Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Embeddings Qwen** — abrir `consulta_gemma/qwen8b/embeddings_qwen3.ipynb` no Colab (A100) ou numa H200, apontar
   `CSV_ENTRADA` para o CSV da amostra, escolher `MODELOS`, `VERSOES` (`TA+C` gera os dois), `COLUNA_ID` e
   `LINHAS_ESPERADAS` (seção 2) e uma `PASTA_SAIDA` por amostra. Ou baixar os vetores do Drive.
2. **Classificador na amostra 1** — `classificador_ipc_qwen8b.ipynb`: na célula 1, apontar `VETORES` para as pastas
   (uma pasta = um vetor; uma tupla `(TA, C)` = média). Ou, pela linha de comando:
   ```bash
   python consulta_gemma/qwen8b/executar_notebook.py \
     "VETORES = {'q8_media': (Path('Embeddings/Amostra_1/Qwen_8B_a1/8B_TA'), Path('Embeddings/Amostra_1/Qwen_8B_a1/8B_C'))}; SAIDA = Path('saida_8b')"
   ```
3. **Treino v1+v2** (o modelo final de cada Qwen):
   ```bash
   python consulta_gemma/qwen8b/treino_v1v2_qwen.py --nome qwen8b \
     --ta1 Embeddings/Amostra_1/Qwen_8B_a1/8B_TA --c1 Embeddings/Amostra_1/Qwen_8B_a1/8B_C \
     --ta2 Embeddings/Amostra_2/Qwen_8B_a2/TA   --c2 Embeddings/Amostra_2/Qwen_8B_a2/C
   ```
   O treino usa PyTorch; com GPU é bem mais rápido, mas roda em CPU.
4. **TF-IDF** — ver `TF-IDF/LEIAME.md` (precisa do CSV de texto da amostra 1 e, para v1+v2, do `dataset_ipc_v2.csv`).
5. **Gemma 2 / API** — `consulta_gemma/train_v2.py` e `consulta_gemma/export_v2.py` (precisam dos CSVs e dos vetores
   Gemma).

