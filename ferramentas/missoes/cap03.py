TITULO = "Representação do problema: reenquadrar, cascata e rebalancear"
MISSAO = (
    "Capítulo 3, padrões 5 a 10. Muitas vezes o que muda o resultado é **como se formula** o problema, não o "
    "algoritmo. Você vai:\n"
    "- **reenquadrar** a regressão do tempo como previsão de **intervalos** (quantis), porque o cliente prefere "
    "\"30 a 40 min\" a \"34,2 min\";\n"
    "- prever **várias etiquetas** de um comentário (*multilabel*);\n"
    "- **combinar modelos** (*ensembles*);\n"
    "- montar uma **cascata** para os pedidos grandes;\n"
    "- criar uma **classe neutra** para a nota 3;\n"
    "- **rebalancear** a previsão de cancelamento, que é raro (1%)."
)
ENTREGA = "o módulo `PKG/problema.py` (intervalos por quantis e cobertura, cascata e avaliação de classes raras) e as métricas de cada formulação."
LEITURA = [("Design Pattern 5: Reframing", 111), ("Design Pattern 6: Multilabel", 120), ("Design Pattern 7: Ensembles", 127),
           ("Design Pattern 8: Cascade", 134), ("Design Pattern 9: Neutral Class", 141), ("Design Pattern 10: Rebalancing", 145)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Reframing:** regressão ↔ classificação, ou prever a distribuição (quantis) em vez de um número.\n"
     "- **Cascade:** um primeiro modelo decide **qual** modelo especialista usar, útil quando há subgrupos com "
     "comportamento diferente.\n"
     "- **Neutral class:** quando os especialistas discordam ou o caso é ambíguo, uma terceira classe evita forçar a "
     "escolha.\n"
     "- **Rebalancing:** com classes raras, a acurácia mente. Use pesos, subamostragem, ou trate o problema como "
     "detecção de anomalia."),

    ("arquivo",
     "**Funções do capítulo.** Crie `problema.py` com:\n"
     "- `treinar_quantis(X, y, quantis=(.1, .5, .9))`: um `GradientBoostingRegressor(loss=\"quantile\", "
     "alpha=q)` por quantil;\n"
     "- `cobertura(y, baixo, alto)`: a fração dentro do intervalo;\n"
     "- `Cascata(roteador, especialistas)`: um `fit(X, y, grupo)` que treina o roteador para prever o grupo e um "
     "especialista por grupo, e um `predict`;\n"
     "- `relatorio_raro(y, prob, limiar)`: precisão, revocação e alertas.",
     "problema.py",
     r'''"""Formulações do problema: quantis, cascata e classes raras."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor


def treinar_quantis(X, y, quantis=(.1, .5, .9), semente=0):
    return {q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150, random_state=semente).fit(X, y) for q in quantis}


def cobertura(y, baixo, alto):
    y = np.asarray(y)
    return float(np.mean((y >= baixo) & (y <= alto)))


class Cascata:
    def __init__(self, roteador, especialista):
        self.roteador, self.base = roteador, especialista

    def fit(self, X, y, grupo):
        self.roteador.fit(X, grupo)
        self.especialistas = {g: clone(self.base).fit(X[grupo == g], y[grupo == g]) for g in np.unique(grupo)}
        return self

    def predict(self, X):
        g = self.roteador.predict(X)
        p = np.zeros(len(X))
        for k, m in self.especialistas.items():
            if np.any(g == k):
                p[g == k] = m.predict(X[g == k])
        return p


def relatorio_raro(y, prob, limiar):
    y, alerta = np.asarray(y), np.asarray(prob) >= limiar
    vp = int((alerta & (y == 1)).sum())
    return {"precisao": round(vp / max(alerta.sum(), 1), 3), "revocacao": round(vp / max(y.sum(), 1), 3), "alertas": int(alerta.sum())}'''),

    ("tarefa",
     "**Reenquadrar: um intervalo.** Treine os quantis 10%, 50% e 90% do tempo de entrega e mostre, para 5 pedidos "
     "do teste, a frase que o app exibiria (\"entre X e Y min\"). Meça a **cobertura** do intervalo 10-90% (deve "
     "ficar perto de 80%) e a largura média.",
     r'''from PKG.base import carregar, separar_no_tempo, X_y, avaliar, NUMERICAS
from PKG.problema import treinar_quantis, cobertura, Cascata, relatorio_raro
df = carregar(); tr, te = separar_no_tempo(df); Xtr, ytr = X_y(tr); Xte, yte = X_y(te)
qs = treinar_quantis(Xtr, ytr)
b, m, a = (qs[q].predict(Xte) for q in (.1, .5, .9))
for i in range(5):
    print(f"pedido {te.pedido_id.iloc[i]}: entre {b[i]:.0f} e {a[i]:.0f} min (real {yte.iloc[i]:.0f})")
print("cobertura 10-90%:", round(cobertura(yte, b, a), 3), "| largura média:", round(float(np.mean(a - b)), 1), "min")'''),

    ("tarefa",
     "**Multilabel.** Cada comentário pode ter várias etiquetas: `demora` (\"demorou\", \"frio\"), `erro_pedido` "
     "(\"faltou\", \"errou\", \"derramado\") e `elogio` (\"rápido\", \"ótima\", \"perfeito\", \"caprichada\"). Crie as "
     "etiquetas por regra, treine um `OneVsRestClassifier(LogisticRegression)` sobre TF-IDF **sem** usar as palavras "
     "da regra como entrada direta (use só os números do pedido) e veja quais etiquetas são previsíveis pelos "
     "números.",
     r'''from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
regras = {"demora": "demorou|frio", "erro_pedido": "faltou|errou|derramado", "elogio": "rápido|ótima|perfeito|caprichada"}
Y = lambda d: np.column_stack([d.comentario.str.contains(p).astype(int) for p in regras.values()])
ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(Xtr.assign(t=ytr), Y(tr))
P = ovr.predict(Xte.assign(t=yte))
for i, nome in enumerate(regras):
    print(f"{nome:<12} F1 {f1_score(Y(te)[:, i], P[:, i]):.3f} | frequência {Y(te)[:, i].mean():.2f}")'''),

    ("tarefa",
     "**Cascata.** Pedidos no horário de pico (19h a 21h) têm outra dinâmica (fila nos restaurantes e trânsito). Monte uma `Cascata` com um "
     "roteador (`DecisionTreeClassifier` que prevê \"pico\") e especialistas `Ridge`, e compare o MAE com um único "
     "`Ridge` e com um único `GradientBoostingRegressor`.",
     r'''from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
grupo = tr.hora.between(19, 21).astype(int).to_numpy()
cas = Cascata(DecisionTreeClassifier(max_depth=3), Ridge(1.0)).fit(Xtr, ytr.to_numpy(), grupo)
print("um Ridge:  ", avaliar(yte, Ridge(1.0).fit(Xtr, ytr).predict(Xte)))
print("cascata:   ", avaliar(yte, cas.predict(Xte)))
print("um boosting:", avaliar(yte, GradientBoostingRegressor(random_state=0).fit(Xtr, ytr).predict(Xte)))'''),

    ("tarefa",
     "**Classe neutra.** Para prever o \"humor\" do cliente pela nota, compare: (a) duas classes, positivo (≥ 4) × "
     "negativo (≤ 3); (b) três classes, com a nota 3 como **neutra**. Nos dois casos, use o tempo de entrega e a "
     "distância. Na formulação (b), quantos clientes \"negativos\" de (a) eram, na verdade, neutros? Para o time de retenção, que liga para quem deu nota 1 ou 2, compare a fração dos alertas \"negativo\" que são mesmo nota 1-2 em cada formulação.",
     r'''from sklearn.metrics import precision_score
cols = ["tempo_entrega_min", "distancia_km", "itens"]
duas = lambda d: np.where(d.nota >= 4, "positivo", "negativo")
tres = lambda d: np.where(d.nota >= 4, "positivo", np.where(d.nota == 3, "neutro", "negativo"))
for nome, f in [("2 classes", duas), ("3 classes", tres)]:
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[cols], f(tr)); p = m.predict(te[cols]) == "negativo"
    ruins = (te.nota <= 2).to_numpy()
    print(f"{nome}: {p.sum()} alertas 'negativo' | eram nota 1-2 de verdade: {ruins[p].mean():.3f} | pegou {ruins[p].sum() / ruins.sum():.3f} dos clientes nota 1-2")
print("dos 'negativos' em 2 classes, eram nota 3:", round(float((te.nota == 3).sum() / (te.nota <= 3).sum()), 3))'''),

    ("tarefa",
     "**Rebalancear o cancelamento (1%).** Compare três estratégias de `LogisticRegression` para prever "
     "`cancelado`:\n"
     "- (a) sem nada;\n"
     "- (b) com `class_weight=\"balanced\"`;\n"
     "- (c) com subamostragem dos não cancelados para 5×.\n\n"
     "Use as numéricas + o tempo de entrega. Mostre a AUC e o `relatorio_raro` com o limiar que dá 50% de "
     "revocação. Repare: a acurácia \"sem nada\" engana, e o rebalanceamento muda a **calibração** (onde fica o limiar), não a **ordenação** (a AUC). Para a ordenação melhorar, é preciso informação nova.",
     r'''from sklearn.metrics import roc_auc_score, precision_recall_curve
cols_c = NUMERICAS + ["tempo_entrega_min"]
r_ = np.random.default_rng(0)
pos = tr[tr.cancelado == 1]; neg = tr[tr.cancelado == 0].sample(5 * len(pos), random_state=0)
opcoes = {"sem nada": (tr, None), "class_weight": (tr, "balanced"), "subamostragem": (pd.concat([pos, neg]), None)}
for nome, (base, cw) in opcoes.items():
    m = LogisticRegression(max_iter=3000, class_weight=cw).fit(base[cols_c], base.cancelado)
    prob = m.predict_proba(te[cols_c])[:, 1]
    pr, rv, lim = precision_recall_curve(te.cancelado, prob); i = np.where(rv[:-1] >= .5)[0][-1]
    print(f"{nome:<14} AUC {roc_auc_score(te.cancelado, prob):.3f} | {relatorio_raro(te.cancelado, prob, lim[i])} | acurácia com 0,5: {np.mean((prob >= .5) == te.cancelado):.3f}")'''),
]
