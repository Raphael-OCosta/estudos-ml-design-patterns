TITULO = "Representação dos dados: hash, embeddings, cruzamentos e multimodal"
MISSAO = (
    "Capítulo 2, padrões 1 a 4. O id do restaurante tem 1.600 valores e cresce todo mês: o one-hot explode e quebra "
    "quando aparece um restaurante novo. Você vai aplicar:\n"
    "- **Hashed Feature**: um número fixo de baldes, que aceita valores novos;\n"
    "- **Embeddings**: o restaurante vira um vetor pequeno aprendido;\n"
    "- **Feature Cross**: hora × dia da semana, para o modelo linear entender o pico;\n"
    "- **Multimodal Input**: números + texto do comentário no mesmo modelo."
)
ENTREGA = "o módulo `PKG/representacao.py` (balde por hash estável, cruzamento e embeddings por SVD) e a tabela de MAE de cada representação."
LEITURA = [("Simple Data Representations", 43), ("Design Pattern 1: Hashed Feature", 55), ("Design Pattern 2: Embeddings", 63),
           ("Design Pattern 3: Feature Cross", 79), ("Design Pattern 4: Multimodal Input", 91)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Hash:** `balde = hash_estável(id) % N`. Valores novos caem em algum balde, e as colisões são o preço.\n"
     "- **Embedding:** cada categoria vira um vetor denso de poucas dimensões, aprendido para que categorias "
     "parecidas fiquem próximas.\n"
     "- **Cruzamento:** cria a combinação explícita (`sexta_20h`). Um modelo linear não descobre interações sozinho.\n"
     "- **Multimodal:** concatena representações de tipos diferentes (números, texto, imagem)."),

    ("arquivo",
     "**As representações.** Crie `representacao.py` com:\n"
     "- `balde_hash(valor, n)`: estável entre execuções (`hashlib.md5` em vez de `hash()`, que muda a cada "
     "processo);\n"
     "- `cruzar(df, a, b)`: uma coluna de texto `\"a_b\"`;\n"
     "- `embeddings_restaurante(df, k=6)`: monta a matriz restaurante × (hora, dia) com o tempo médio de entrega "
     "(célula vazia recebe a média do restaurante), aplica `TruncatedSVD(k)` e devolve um DataFrame `restaurante_id → emb_0..emb_k-1`.",
     "representacao.py",
     r'''"""Representações de dados: hash estável, cruzamento e embeddings por SVD."""
import hashlib

import pandas as pd
from sklearn.decomposition import TruncatedSVD


def balde_hash(valor, n):
    return int(hashlib.md5(str(valor).encode()).hexdigest(), 16) % n


def cruzar(df, a, b):
    return df[a].astype(str) + "_" + df[b].astype(str)


def embeddings_restaurante(df, k=6, semente=0):
    matriz = df.pivot_table(index="restaurante_id", columns=["hora", "dia_semana"], values="tempo_entrega_min", aggfunc="mean")
    matriz = matriz.T.fillna(matriz.mean(axis=1)).T  # célula vazia recebe a média do restaurante
    z = TruncatedSVD(k, random_state=semente).fit_transform(matriz.to_numpy())
    return pd.DataFrame(z, index=matriz.index, columns=[f"emb_{i}" for i in range(k)]).reset_index()'''),

    ("tarefa",
     "**Hash × one-hot × nada.** Prever o tempo com as numéricas + o restaurante como (a) nada, (b) one-hot e (c) "
     "hash com 100 baldes (one-hot do balde), usando `Ridge`. Compare o MAE e o número de colunas. Depois simule "
     "**restaurantes novos** no teste (troque o id de 10% dos pedidos por ids nunca vistos): o one-hot ignora, e o "
     "hash ainda dá alguma informação?",
     r'''from sklearn.linear_model import Ridge
from PKG.base import carregar, separar_no_tempo, avaliar, NUMERICAS
from PKG.representacao import balde_hash, cruzar, embeddings_restaurante
df = carregar(); tr, te = separar_no_tempo(df)
def matriz(d, modo, colunas=None):
    X = d[NUMERICAS].copy()
    if modo == "onehot":
        X = X.join(pd.get_dummies(d.restaurante_id.astype(str), prefix="r", dtype=float))
    if modo == "hash":
        X = X.join(pd.get_dummies(d.restaurante_id.map(lambda v: balde_hash(v, 100)).astype(str), prefix="h", dtype=float))
    return X.reindex(columns=colunas, fill_value=0) if colunas is not None else X
te_novos = te.copy(); te_novos.loc[te_novos.sample(frac=.1, random_state=1).index, "restaurante_id"] += 100000
for modo in ["nada", "onehot", "hash"]:
    Xtr = matriz(tr, modo); m = Ridge(1.0).fit(Xtr, tr.tempo_entrega_min)
    print(f"{modo:<7} colunas {Xtr.shape[1]:>5} | MAE {avaliar(te.tempo_entrega_min, m.predict(matriz(te, modo, Xtr.columns)))['MAE']}"
          f" | com 10% de restaurantes novos {avaliar(te.tempo_entrega_min, m.predict(matriz(te_novos, modo, Xtr.columns)))['MAE']}")'''),

    ("tarefa",
     "**Embeddings.** Calcule embeddings de 6 dimensões dos restaurantes **só com o treino** e acrescente-os às "
     "numéricas (restaurante sem embedding recebe zeros). Compare o MAE do `Ridge` com o one-hot: o modelo usa 6 "
     "colunas em vez de 1.600. Mostre também os 3 restaurantes mais parecidos com o mais popular (distância entre "
     "embeddings).",
     r'''emb = embeddings_restaurante(tr)
def com_emb(d):
    return d[NUMERICAS].join(d[["restaurante_id"]].merge(emb, on="restaurante_id", how="left").drop(columns="restaurante_id").fillna(0).set_index(d.index))
m = Ridge(1.0).fit(com_emb(tr), tr.tempo_entrega_min)
print("embeddings (6 colunas) MAE:", avaliar(te.tempo_entrega_min, m.predict(com_emb(te)))["MAE"])
top = tr.restaurante_id.value_counts().index[0]; E = emb.set_index("restaurante_id")
dist = ((E - E.loc[top]) ** 2).sum(axis=1).sort_values()
print("mais parecidos com", top, ":", dist.index[1:4].tolist())'''),

    ("tarefa",
     "**Feature cross.** O pico (sexta a domingo, 19h a 21h) é uma **interação**. Compare um `Ridge` com hora e dia "
     "como números, depois como one-hot separados e, por fim, com o cruzamento `hora × dia_semana` em one-hot. Só o "
     "cruzamento deixa o modelo linear capturar o pico.",
     r'''def com_cat(d, colunas_cat, colunas=None):
    X = d[NUMERICAS].drop(columns=["hora", "dia_semana"]).join(pd.get_dummies(colunas_cat(d).astype(str), dtype=float))
    return X.reindex(columns=colunas, fill_value=0) if colunas is not None else X
for nome, cat in [("numéricas", None), ("one-hot separados", lambda d: d[["hora", "dia_semana"]]), ("cruzamento", lambda d: cruzar(d, "hora", "dia_semana"))]:
    if cat is None:
        Xa, Xb = tr[NUMERICAS], te[NUMERICAS]
    else:
        Xa = com_cat(tr, cat); Xb = com_cat(te, cat, Xa.columns)
    print(f"{nome:<18}MAE {avaliar(te.tempo_entrega_min, Ridge(1.0).fit(Xa, tr.tempo_entrega_min).predict(Xb))['MAE']}")'''),

    ("tarefa",
     "**Multimodal: números + texto.** Agora o alvo é a **nota** do pedido (1 a 5), prevista com o tempo de entrega "
     "e o **texto** do comentário juntos: `ColumnTransformer` com `TfidfVectorizer` no texto e `StandardScaler` nos "
     "números, e uma `LogisticRegression`. Compare com só números e só texto (acurácia). Se o multimodal não ganhar do texto, pergunte-se: o comentário já carrega quase toda a informação da nota? A combinação só ajuda quando cada fonte traz algo que a outra não tem.",
     r'''from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
nums = ["tempo_entrega_min", "distancia_km", "itens"]
opcoes = {"só números": ColumnTransformer([("n", StandardScaler(), nums)]),
          "só texto": ColumnTransformer([("t", TfidfVectorizer(), "comentario")]),
          "multimodal": ColumnTransformer([("n", StandardScaler(), nums), ("t", TfidfVectorizer(), "comentario")])}
for nome, prep in opcoes.items():
    m = make_pipeline(prep, LogisticRegression(max_iter=2000)).fit(tr, tr.nota)
    print(f"{nome:<11} acurácia {m.score(te, te.nota):.3f}")'''),
]
