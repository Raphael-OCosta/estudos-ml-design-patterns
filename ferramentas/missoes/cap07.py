TITULO = "IA responsável: referência heurística, explicações e justiça"
MISSAO = (
    "Capítulo 7, padrões 28 a 30. Antes de dizer que o modelo é bom, compare com uma **regra simples** que um "
    "humano entende (*Heuristic Benchmark*). Antes de confiar nele, **explique** as previsões, no global e caso a "
    "caso. E antes de lançar, olhe pela **lente da justiça**:\n"
    "- o erro é parecido para os bairros da periferia e do centro?\n"
    "- o app promete tempos piores para quem mora longe?"
)
ENTREGA = "o módulo `PKG/responsavel.py` (explicação por contribuição de features e métricas por grupo) e uma análise de justiça com correção de viés."
LEITURA = [("Design Pattern 28: Heuristic Benchmark", 367), ("Design Pattern 29: Explainable Predictions", 376), ("Design Pattern 30: Fairness Lens", 401)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Referência heurística:** uma regra que o negócio já usaria, como a \"mediana do bairro naquele horário\". "
     "Se o modelo não ganha dela com folga, não vale a complexidade.\n"
     "- **Explicação local:** o quanto cada feature empurrou **esta** previsão para cima ou para baixo.\n"
     "- **Justiça:** compare métricas **por grupo** (erro, viés). Um erro médio bom pode esconder um grupo mal "
     "atendido."),

    ("arquivo",
     "**Ferramentas responsáveis.** Crie `responsavel.py` com:\n"
     "- `contribuicoes(modelo, X, linha, fundo)`: uma explicação por substituição. Para cada feature, a mudança na "
     "previsão quando o valor da linha substitui o do fundo, na média sobre 200 linhas de fundo. É uma aproximação "
     "simples do SHAP;\n"
     "- `por_grupo(y, p, grupo)`: um DataFrame com n, MAE e viés (y - p) por grupo.",
     "responsavel.py",
     r'''"""IA responsável: explicações locais e métricas por grupo."""
import numpy as np
import pandas as pd


def contribuicoes(modelo, X, linha, fundo, n_fundo=200, semente=0):
    base = fundo.sample(min(n_fundo, len(fundo)), random_state=semente)
    p_base = modelo.predict(base).mean()
    efeitos = {}
    for col in X.columns:
        trocado = base.copy()
        trocado[col] = linha[col]
        efeitos[col] = float(modelo.predict(trocado).mean() - p_base)
    return float(p_base), pd.Series(efeitos).sort_values(key=abs, ascending=False)


def por_grupo(y, p, grupo):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(grupo)})
    return d.groupby("g").apply(lambda t: pd.Series({"n": len(t), "MAE": np.mean(np.abs(t.y - t.p)), "vies": np.mean(t.y - t.p)}), include_groups=False).round(2)'''),

    ("tarefa",
     "**Referência heurística.** A regra do negócio: \"mediana do tempo no bairro, naquela hora e dia útil ou fim de "
     "semana\". Compare essa regra com o `HistGradientBoostingRegressor` com as numéricas + o bairro "
     "(one-hot). Quanto o modelo ganha em MAE e na fração dentro de ±5 min?",
     r'''from sklearn.ensemble import HistGradientBoostingRegressor
from PKG.base import carregar, separar_no_tempo, avaliar, NUMERICAS
from PKG.responsavel import contribuicoes, por_grupo
df = carregar(); df["fim_semana"] = (df.dia_semana >= 4).astype(int); tr, te = separar_no_tempo(df)
heur = tr.groupby(["bairro", "hora", "fim_semana"]).tempo_entrega_min.median()
p_heur = te.set_index(["bairro", "hora", "fim_semana"]).index.map(heur).fillna(tr.tempo_entrega_min.median())
X = lambda d: d[NUMERICAS].join(pd.get_dummies(d.bairro, prefix="b", dtype=int))
modelo = HistGradientBoostingRegressor(random_state=0).fit(X(tr), tr.tempo_entrega_min)
print("heurística:", avaliar(te.tempo_entrega_min, p_heur)); print("modelo:    ", avaliar(te.tempo_entrega_min, modelo.predict(X(te))))'''),

    ("tarefa",
     "**Explicar previsões.** (a) **Global:** a `permutation_importance` do modelo no teste. (b) **Local:** para o "
     "pedido com a **maior** previsão do teste, use `contribuicoes` e mostre a frase que o atendimento leria (as 3 "
     "maiores causas, com os minutos).",
     r'''from sklearn.inspection import permutation_importance
Xte = X(te); pi = permutation_importance(modelo, Xte, te.tempo_entrega_min, n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
print(pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).head(5).round(2))
i = int(np.argmax(modelo.predict(Xte))); linha = Xte.iloc[i]
base, ef = contribuicoes(modelo, Xte, linha, X(tr))
print(f"previsão {modelo.predict(Xte.iloc[[i]])[0]:.0f} min (média {base:.0f}). Principais causas: " +
      "; ".join(f"{c} = {linha[c]} ({v:+.1f} min)" for c, v in ef.head(3).items()))'''),

    ("tarefa",
     "**Lente da justiça.** Compare o MAE e o **viés** do modelo por região (central × periferia) e por bairro. Com "
     "dados representativos, o viés deve ficar perto de zero. Agora o cenário real de muitos apps: no lançamento quase "
     "todos os pedidos eram do centro. Treine um modelo só com as numéricas num treino com apenas 5% da periferia e "
     "olhe de novo, com o viés por bairro desenhado: quem foi pouco visto recebe promessas piores.",
     r'''import matplotlib.pyplot as plt
p = modelo.predict(Xte)
print(por_grupo(te.tempo_entrega_min, p, te.regiao))
tr_centro = pd.concat([tr[tr.regiao == "central"], tr[tr.regiao == "periferia"].sample(frac=.05, random_state=0)])
enviesado = HistGradientBoostingRegressor(random_state=0).fit(tr_centro[NUMERICAS], tr_centro.tempo_entrega_min)
print(por_grupo(te.tempo_entrega_min, enviesado.predict(te[NUMERICAS]), te.regiao))
g = por_grupo(te.tempo_entrega_min, enviesado.predict(te[NUMERICAS]), te.bairro).sort_values("vies")
g.vies.plot.barh(color=["#c0392b" if v > 0 else "#2e86c1" for v in g.vies]); plt.axvline(0, c="k"); plt.title("Viés por bairro, treino concentrado no centro"); plt.show()'''),

    ("tarefa",
     "**Promessa justa.** O app mostra ao cliente o quantil 80% previsto (\"chega em até X min\"). Treine um "
     "boosting de quantil 0,8 **no treino concentrado no centro** e meça, por região, a fração de pedidos que **cumpriram** a promessa. Se uma região "
     "fica abaixo de 80%, ajuste um acréscimo por região (calibrado no treino) e mostre o antes e o depois.",
     r'''q80 = HistGradientBoostingRegressor(loss="quantile", quantile=.8, random_state=0).fit(tr_centro[NUMERICAS], tr_centro.tempo_entrega_min)
def cumpriu(d, extra=0):
    return pd.Series(d.tempo_entrega_min.to_numpy() <= q80.predict(d[NUMERICAS]) + extra, index=d.index).groupby(d.regiao).mean().round(3)
print("antes:", cumpriu(te).to_dict())
# calibra com o treino completo (faz o papel dos pedidos novos coletados depois do lançamento)
p_tr = q80.predict(tr[NUMERICAS]); extra = {}
for reg in ["central", "periferia"]:
    m = (tr.regiao == reg).to_numpy()
    extra[reg] = float(np.quantile(tr.tempo_entrega_min.to_numpy()[m] - p_tr[m], .8))
extra_te = te.regiao.map(extra).to_numpy()
print("acréscimo por região (min):", {k: round(v, 1) for k, v in extra.items()})
print("depois:", cumpriu(te, extra_te).to_dict())'''),
]
