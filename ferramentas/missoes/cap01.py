TITULO = "Por que padrões de projeto em ML"
MISSAO = (
    "Você entrou no time de ML do **Rango**, um app de entrega de comida em Curitiba e Londrina. O produto precisa "
    "prever o **tempo de entrega** mostrado ao cliente. O livro reúne 30 padrões de projeto: soluções testadas para "
    "problemas que **todo** projeto de ML encontra. Antes deles, você vai ver os problemas acontecendo:\n"
    "- resultado que não se repete;\n"
    "- qualidade de dados;\n"
    "- **mudança no mundo** (*drift*);\n"
    "- a tentação de comparar modelos sem uma linha de base."
)
ENTREGA = "o módulo `PKG/base.py` (carga, separação temporal e avaliação padrão do projeto) e os quatro desafios do capítulo 1 medidos em números."
LEITURA = [("What Are Design Patterns?", 15), ("Machine Learning Terminology", 19), ("Common Challenges in Machine Learning", 29)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "Um **padrão de projeto** descreve um problema recorrente, a solução e as trocas que ela envolve (como no livro "
     "de arquitetura de Christopher Alexander). Os desafios típicos de ML que os padrões atacam:\n"
     "- **qualidade dos dados**;\n"
     "- **reprodutibilidade**;\n"
     "- **drift**;\n"
     "- **escala**;\n"
     "- **objetivos múltiplos** (o produto quer uma coisa, o modelo otimiza outra)."),

    ("arquivo",
     "**A base do projeto.** Crie `base.py` com:\n"
     "- `carregar()`: os pedidos de Curitiba;\n"
     "- `separar_no_tempo(df, data_corte)`: treino até a data e teste depois (nunca embaralhar dados com tempo);\n"
     "- `X_y(df, colunas)`;\n"
     "- `avaliar(y, p)`: um dicionário com o MAE, o RMSE e a fração das previsões dentro de ±5 minutos.",
     "base.py",
     r'''"""Funções comuns do projeto Rango."""
import numpy as np
import pandas as pd

from dados import pedidos

NUMERICAS = ["distancia_km", "itens", "chuva_mm", "hora", "dia_semana", "entregador_exp_meses"]


def carregar():
    return pedidos()


def separar_no_tempo(df, data_corte="2025-05-01"):
    return df[df.momento < data_corte].copy(), df[df.momento >= data_corte].copy()


def X_y(df, colunas=NUMERICAS, alvo="tempo_entrega_min"):
    return df[colunas], df[alvo]


def avaliar(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": round(float(np.mean(np.abs(y - p))), 2), "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
            "dentro_5min": round(float(np.mean(np.abs(y - p) <= 5)), 3)}'''),

    ("tarefa",
     "**Desafio 1: a linha de base.** Separe no tempo (treino até abril, teste em maio e junho). Antes de qualquer "
     "modelo, avalie dois \"chutes\": a média do treino e a mediana por bairro e hora. Depois treine uma "
     "`GradientBoostingRegressor` com as colunas numéricas. Quanto o modelo ganha?",
     r'''from sklearn.ensemble import GradientBoostingRegressor
from PKG.base import carregar, separar_no_tempo, X_y, avaliar, NUMERICAS
df = carregar(); tr, te = separar_no_tempo(df)
print("média:", avaliar(te.tempo_entrega_min, np.full(len(te), tr.tempo_entrega_min.mean())))
mediana = tr.groupby(["bairro", "hora"]).tempo_entrega_min.median()
p_med = te.set_index(["bairro", "hora"]).index.map(mediana).fillna(tr.tempo_entrega_min.median())
print("mediana bairro×hora:", avaliar(te.tempo_entrega_min, p_med))
gb = GradientBoostingRegressor(random_state=0).fit(*X_y(tr))
print("gradient boosting:", avaliar(te.tempo_entrega_min, gb.predict(X_y(te)[0])))'''),

    ("tarefa",
     "**Desafio 2: reprodutibilidade.** Treine o mesmo `RandomForestRegressor(50)` três vezes **sem** "
     "`random_state` e três vezes com `random_state=42`. Compare os MAEs: só a segunda série se repete.",
     r'''from sklearn.ensemble import RandomForestRegressor
Xtr, ytr = X_y(tr); Xte, yte = X_y(te)
mae4 = lambda m: round(float(np.mean(np.abs(yte - m.fit(Xtr, ytr).predict(Xte)))), 4)
sem = [mae4(RandomForestRegressor(50, n_jobs=-1)) for _ in range(3)]
com = [mae4(RandomForestRegressor(50, n_jobs=-1, random_state=42)) for _ in range(3)]
print("sem semente:", sem, "| com semente:", com)'''),

    ("tarefa",
     "**Desafio 3: qualidade dos dados.** Estrague 2% das distâncias do **treino** (multiplique por 1.000, como se "
     "alguém digitasse metros no lugar de km) e retreine uma `LinearRegression` (modelos lineares sentem muito mais os valores absurdos que as árvores). Quanto o MAE no teste piora? Depois limpe "
     "(as distâncias acima de 50 km viram `NaN` e são preenchidas com a mediana) e confira a recuperação.",
     r'''sujo = tr.copy(); r_ = np.random.default_rng(0)
idx = r_.choice(len(sujo), int(.02 * len(sujo)), replace=False)
sujo.iloc[idx, sujo.columns.get_loc("distancia_km")] *= 1000
from sklearn.linear_model import LinearRegression
print("linear com dados limpos:", avaliar(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))["MAE"])
mae_sujo = avaliar(yte, LinearRegression().fit(*X_y(sujo)).predict(Xte))["MAE"]
limpo = sujo.copy(); limpo.loc[limpo.distancia_km > 50, "distancia_km"] = np.nan
limpo["distancia_km"] = limpo.distancia_km.fillna(limpo.distancia_km.median())
mae_limpo = avaliar(yte, LinearRegression().fit(*X_y(limpo)).predict(Xte))["MAE"]
print(f"sujo {mae_sujo} | limpo {mae_limpo}")'''),

    ("tarefa",
     "**Desafio 4: drift.** Gere pedidos do 2º semestre com o trânsito pior (`pedidos(semente=8, "
     "inicio=\"2025-07-01\", drift=1.2)`) e avalie o **mesmo** modelo. O MAE sobe e o erro passa a ter **viés** (a "
     "média de y - p deixa de ser zero). É o problema que o padrão *Continued Model Evaluation* (cap. 5) resolve.",
     r'''futuro = pedidos(semente=8, inicio="2025-07-01", drift=1.2)
pf = gb.predict(X_y(futuro)[0])
print("teste original:", avaliar(yte, gb.predict(Xte)), "| viés", round(float((yte - gb.predict(Xte)).mean()), 2))
print("2º semestre:   ", avaliar(futuro.tempo_entrega_min, pf), "| viés", round(float((futuro.tempo_entrega_min - pf).mean()), 2))'''),
]
