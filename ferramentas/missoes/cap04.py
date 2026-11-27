TITULO = "Padrões de treino: checkpoints, transferência e ajuste"
MISSAO = (
    "Capítulo 4, padrões 11 a 15. Você vai usar:\n"
    "- **Useful Overfitting**: decorar de propósito uma função determinística, a tabela de preparo, porque não "
    "existe \"dado novo\" ali;\n"
    "- **Checkpoints**: salvar o modelo a cada época, retomar o treino e escolher a melhor época pela validação;\n"
    "- **Transfer Learning**: treinar em Curitiba (muitos dados) e ajustar para Londrina (poucos dados);\n"
    "- **Distribution Strategy**: o treino paralelo em dados com a média dos gradientes, simulado;\n"
    "- **Hyperparameter Tuning**: a busca por halving, mais barata que a grade."
)
ENTREGA = "o módulo `PKG/treino.py` (laço de treino com checkpoints e retomada, e SGD paralelo em dados) e os ganhos de cada padrão medidos."
LEITURA = [("Typical Training Loop", 159), ("Design Pattern 11: Useful Overfitting", 162), ("Design Pattern 12: Checkpoints", 173),
           ("Design Pattern 13: Transfer Learning", 189), ("Design Pattern 14: Distribution Strategy", 208), ("Design Pattern 15: Hyperparameter Tuning", 226)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Checkpoint:** o estado completo (pesos e otimizador) salvo periodicamente. Serve para sobreviver a quedas, "
     "voltar à melhor época (a parada antecipada \"virtual\") e continuar treinando com dados novos.\n"
     "- **Transferência:** reaproveita o que foi aprendido num domínio grande como ponto de partida num domínio "
     "pequeno.\n"
     "- **Paralelismo em dados:** cada trabalhador calcula o gradiente num pedaço dos dados, e a média dos "
     "gradientes é o gradiente do lote inteiro."),

    ("tarefa",
     "**Overfitting útil.** O tempo de preparo é uma **tabela determinística** (`TEMPO_PREPARO`, culinária × itens). "
     "Treine uma `DecisionTreeRegressor` **sem limite** em todas as combinações e confira que o erro é zero. Aqui não "
     "há teste a temer: o domínio inteiro está no treino. Depois mostre o que acontece com `max_depth=3`, a "
     "\"regularização\" errada para esse caso.",
     r'''from sklearn.tree import DecisionTreeRegressor
tab = pd.DataFrame([(c, i, t) for (c, i), t in TEMPO_PREPARO.items()], columns=["culinaria", "itens", "preparo"])
X = pd.get_dummies(tab[["culinaria"]], dtype=int).assign(itens=tab.itens)
for prof in [None, 3]:
    p = DecisionTreeRegressor(max_depth=prof, random_state=0).fit(X, tab.preparo).predict(X)
    print(f"max_depth={prof}: erro máximo {np.abs(p - tab.preparo).max():.2f} min")'''),

    ("arquivo",
     "**Laço de treino com checkpoints.** Crie `treino.py` com:\n"
     "- `treinar_com_checkpoints(modelo, X, y, X_val, y_val, epocas, pasta)`: a cada época, faz `partial_fit`, "
     "salva `ep_XX.joblib` e registra o MAE de validação; devolve o histórico;\n"
     "- `melhor_checkpoint(historico)`;\n"
     "- `retomar(pasta, epoca)`: carrega o checkpoint;\n"
     "- `gradiente_mse(w, X, y)` e `sgd_paralelo(X, y, trabalhadores, passos, taxa)`: divide cada lote em "
     "`trabalhadores` pedaços, calcula os gradientes separados e aplica a média.",
     "treino.py",
     r'''"""Padrões de treino: checkpoints e treino paralelo em dados."""
from pathlib import Path

import joblib
import numpy as np


def treinar_com_checkpoints(modelo, X, y, X_val, y_val, epocas, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    historico = []
    for ep in range(1, epocas + 1):
        modelo.partial_fit(X, y)
        joblib.dump(modelo, pasta / f"ep_{ep:02d}.joblib")
        historico.append((ep, float(np.mean(np.abs(modelo.predict(X_val) - y_val)))))
    return historico


def melhor_checkpoint(historico):
    return min(historico, key=lambda e: e[1])


def retomar(pasta, epoca):
    return joblib.load(Path(pasta) / f"ep_{epoca:02d}.joblib")


def gradiente_mse(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)


def sgd_paralelo(X, y, trabalhadores=1, passos=300, taxa=0.05, lote=512, semente=0):
    r = np.random.default_rng(semente)
    w = np.zeros(X.shape[1])
    for _ in range(passos):
        idx = r.choice(len(X), lote, replace=False)
        pedacos = np.array_split(idx, trabalhadores)
        g = np.mean([gradiente_mse(w, X[p], y[p]) for p in pedacos], axis=0)
        w -= taxa * g
    return w'''),

    ("tarefa",
     "**Checkpoints na prática.** Use um `MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=True)` padronizado (o "
     "`partial_fit` faz uma passada). Treine 25 épocas com checkpoints em `../saidas/ckpt`, desenhe o MAE de "
     "validação, ache a melhor época e retome a partir dela para avaliar no teste.",
     r'''import warnings, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from PKG.base import carregar, separar_no_tempo, X_y, avaliar, NUMERICAS
from PKG.treino import treinar_com_checkpoints, melhor_checkpoint, retomar, sgd_paralelo
warnings.filterwarnings("ignore")
df = carregar(); tr, te = separar_no_tempo(df); tr, va = separar_no_tempo(tr, "2025-04-01")
esc = StandardScaler().fit(X_y(tr)[0]); Z = lambda d: esc.transform(X_y(d)[0])
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, random_state=0)
hist = treinar_com_checkpoints(mlp, Z(tr), tr.tempo_entrega_min.to_numpy(), Z(va), va.tempo_entrega_min.to_numpy(), 25, "../saidas/ckpt")
plt.plot(*zip(*hist), "o-"); plt.xlabel("época"); plt.ylabel("MAE validação"); plt.show()
ep, mae = melhor_checkpoint(hist); melhor = retomar("../saidas/ckpt", ep)
print("melhor época", ep, "| MAE val", round(mae, 2), "| teste", avaliar(te.tempo_entrega_min, melhor.predict(Z(te))))'''),

    ("tarefa",
     "**Transferência Curitiba → Londrina.** Londrina tem só 300 pedidos de treino. Compare no teste de Londrina:\n"
     "- (a) o MLP treinado do zero com os 300;\n"
     "- (b) o melhor checkpoint de Curitiba **sem** ajuste;\n"
     "- (c) o checkpoint de Curitiba **ajustado** com mais 10 épocas nos 300 de Londrina (`partial_fit`).\n\nSe o ajuste não ajudar, pense: as duas cidades seguem a mesma física de entrega? Com domínios parecidos, o modelo grande já serve, e 10 épocas em 300 pedidos podem até atrapalhar.",
     r'''import copy
lon = pedidos(4000, "Londrina", semente=11); lon_tr, lon_te = lon.iloc[:300], lon.iloc[1000:]
y_l = lon_tr.tempo_entrega_min.to_numpy()
zero = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0).fit(Z(lon_tr), y_l)
ajustado = copy.deepcopy(melhor)
for _ in range(10):
    ajustado.partial_fit(Z(lon_tr), y_l)
for nome, m in [("do zero (300)", zero), ("Curitiba sem ajuste", melhor), ("Curitiba + ajuste", ajustado)]:
    print(f"{nome:<22}", avaliar(lon_te.tempo_entrega_min, m.predict(Z(lon_te))))'''),

    ("tarefa",
     "**Paralelismo em dados.** Com uma regressão linear nas numéricas padronizadas (+ intercepto), rode o "
     "`sgd_paralelo` com 1, 4 e 16 \"trabalhadores\". Os pesos finais são praticamente iguais: a média dos "
     "gradientes dos pedaços **é** o gradiente do lote. Compare com a solução de mínimos quadrados.",
     r'''A = np.column_stack([np.ones(len(tr)), Z(tr)]); yv = tr.tempo_entrega_min.to_numpy()
exato = np.linalg.lstsq(A, yv, rcond=None)[0]
for k in [1, 4, 16]:
    w = sgd_paralelo(A, yv, trabalhadores=k, passos=600, taxa=0.05)
    print(f"{k:>2} trabalhadores: diferença máxima para o exato {np.abs(w - exato).max():.3f}")'''),

    ("tarefa",
     "**Ajuste de hiperparâmetros.** Compare a `GridSearchCV` com a `HalvingRandomSearchCV` (`from "
     "sklearn.experimental import enable_halving_search_cv`) para um `HistGradientBoostingRegressor` "
     "(`learning_rate`, `max_leaf_nodes`, `l2_regularization`). Meça o tempo e o MAE no teste. (O `best_score_` do halving vem de rodadas com poucos dados e não se compara com o da grade.)",
     r'''from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
grade = {"learning_rate": [.03, .1, .3], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 1.0]}
Xtr2, ytr2 = X_y(pd.concat([tr, va]))
for nome, busca in [("grade", GridSearchCV(HistGradientBoostingRegressor(random_state=0), grade, cv=3, scoring="neg_mean_absolute_error")),
                    ("halving", HalvingRandomSearchCV(HistGradientBoostingRegressor(random_state=0), grade, cv=3, scoring="neg_mean_absolute_error", random_state=0))]:
    t = time.perf_counter(); busca.fit(Xtr2, ytr2)
    print(f"{nome:<8}{time.perf_counter() - t:5.1f}s | teste {avaliar(te.tempo_entrega_min, busca.predict(X_y(te)[0]))['MAE']} | {busca.best_params_}")'''),
]
