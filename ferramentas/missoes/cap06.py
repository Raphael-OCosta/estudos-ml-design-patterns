TITULO = "Reprodutibilidade: transform, divisão repetível, feature store e versões"
MISSAO = (
    "Capítulo 6, padrões 21 a 27. É o que separa um notebook de um sistema. Você vai:\n"
    "- provocar e corrigir o **desvio treino/serving** com o padrão *Transform* (a preparação salva junto com o "
    "modelo);\n"
    "- fazer uma **divisão repetível** por hash do cliente;\n"
    "- usar uma **ponte de esquema** para uma coluna nova;\n"
    "- calcular uma **inferência com janela** igual no treino e no serving;\n"
    "- montar um **pipeline de workflow** com cache;\n"
    "- usar uma **feature store** com junção no ponto do tempo;\n"
    "- **versionar** os modelos."
)
ENTREGA = "o módulo `PKG/reprodutivel.py` (divisão por hash, feature de janela, runner de workflow com cache, mini feature store e registro de versões)."
LEITURA = [("Design Pattern 21: Transform", 301), ("Design Pattern 22: Repeatable Splitting", 308), ("Design Pattern 23: Bridged Schema", 315),
           ("Design Pattern 24: Windowed Inference", 322), ("Design Pattern 25: Workflow Pipeline", 331), ("Design Pattern 26: Feature Store", 341),
           ("Design Pattern 27: Model Versioning", 357)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Desvio treino/serving:** é o bug mais caro de ML. O treino normaliza com a média do treino e o serving "
     "esquece, ou usa outra média.\n"
     "- **Transform:** guarda a preparação **dentro** do artefato do modelo.\n"
     "- **Divisão repetível:** `hash(chave) % 100 < 80` manda sempre o mesmo cliente para o mesmo lado, mesmo com "
     "dados novos.\n"
     "- **Feature store:** uma única definição de cada feature, servida *offline* (treino, com junção no ponto do "
     "tempo) e *online* (serving, o valor mais recente)."),

    ("arquivo",
     "**As peças.** Crie `reprodutivel.py` com:\n"
     "- `lado_por_hash(chave, pct_treino=80)`;\n"
     "- `pedidos_ultima_hora(df)`: para cada pedido, quantos pedidos o **mesmo restaurante** recebeu nos 60 minutos "
     "**anteriores** (`rolling('60min')` por restaurante, sem contar o próprio);\n"
     "- `Workflow`: `passo(nome, funcao, *dependencias)` registra e `rodar()` executa em ordem, guardando o "
     "resultado de cada passo em `../saidas/cache/<nome>.joblib`. Se o cache existe, pula;\n"
     "- `FeatureStore`: `offline(entidades_com_tempo)`, com `merge_asof` do valor válido **antes** de cada "
     "momento, e `online(id)`, com o último valor;\n"
     "- `Registro(pasta)`: `salvar(modelo, nome, metricas)` cria `v1`, `v2`... com o `metadados.json`, e "
     "`carregar(nome, versao=None)` pega a mais recente.",
     "reprodutivel.py",
     r'''"""Padrões de reprodutibilidade: hash, janela, workflow, feature store e registro de modelos."""
import hashlib
import json
import time
from pathlib import Path

import joblib
import pandas as pd


def lado_por_hash(chave, pct_treino=80):
    return "treino" if int(hashlib.md5(str(chave).encode()).hexdigest(), 16) % 100 < pct_treino else "teste"


def pedidos_ultima_hora(df):
    d = df.sort_values("momento").set_index("momento")
    cont = d.groupby("restaurante_id")["pedido_id"].transform(lambda s: s.rolling("60min").count() - 1)
    return pd.Series(cont.to_numpy(), index=d["pedido_id"].to_numpy()).reindex(df["pedido_id"]).to_numpy()


class Workflow:
    def __init__(self, pasta_cache):
        self.passos, self.cache = [], Path(pasta_cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def passo(self, nome, funcao, *dependencias):
        self.passos.append((nome, funcao, dependencias))

    def rodar(self):
        resultados, log = {}, []
        for nome, funcao, deps in self.passos:
            arq = self.cache / f"{nome}.joblib"
            if arq.exists():
                resultados[nome] = joblib.load(arq)
                log.append((nome, "cache"))
                continue
            t = time.perf_counter()
            resultados[nome] = funcao(*[resultados[d] for d in deps])
            joblib.dump(resultados[nome], arq)
            log.append((nome, f"rodou em {time.perf_counter() - t:.2f}s"))
        return resultados, log


class FeatureStore:
    def __init__(self, tabela, chave, tempo, colunas):
        self.t = tabela.sort_values(tempo)
        self.chave, self.tempo, self.colunas = chave, tempo, colunas

    def offline(self, entidades):
        return pd.merge_asof(entidades.sort_values(self.tempo), self.t[[self.chave, self.tempo] + self.colunas],
                             on=self.tempo, by=self.chave, direction="backward", allow_exact_matches=False)

    def online(self, chave):
        return self.t[self.t[self.chave] == chave].iloc[-1][self.colunas].to_dict()


class Registro:
    def __init__(self, pasta):
        self.pasta = Path(pasta)

    def salvar(self, modelo, nome, metricas):
        base = self.pasta / nome
        versoes = sorted(int(p.name[1:]) for p in base.glob("v*")) if base.exists() else []
        destino = base / f"v{(versoes[-1] + 1) if versoes else 1}"
        destino.mkdir(parents=True)
        joblib.dump(modelo, destino / "modelo.joblib")
        (destino / "metadados.json").write_text(json.dumps({"metricas": metricas, "criado": time.strftime("%Y-%m-%d %H:%M")}), encoding="utf-8")
        return destino.name

    def carregar(self, nome, versao=None):
        base = self.pasta / nome
        versao = versao or f"v{max(int(p.name[1:]) for p in base.glob('v*'))}"
        return joblib.load(base / versao / "modelo.joblib"), json.loads((base / versao / "metadados.json").read_text(encoding="utf-8"))'''),

    ("tarefa",
     "**Transform: o desvio treino/serving.** Treine uma `Ridge` com as numéricas **padronizadas à mão** (média e "
     "desvio do treino). No \"serving\", o engenheiro esquece de padronizar: compare o MAE. Depois empacote "
     "`StandardScaler` + `Ridge` num `make_pipeline`, salve com `joblib` e carregue: a preparação viaja junto, e o "
     "erro some.",
     r'''import joblib, shutil
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PKG.base import carregar, separar_no_tempo, X_y, avaliar, NUMERICAS
from PKG.reprodutivel import lado_por_hash, pedidos_ultima_hora, Workflow, FeatureStore, Registro
df = carregar(); tr, te = separar_no_tempo(df); Xtr, ytr = X_y(tr); Xte, yte = X_y(te)
mu, sd = Xtr.mean(), Xtr.std()
manual = Ridge().fit((Xtr - mu) / sd, ytr)
print("serving correto:", avaliar(yte, manual.predict((Xte - mu) / sd))["MAE"], "| esqueceu de padronizar:", avaliar(yte, manual.predict(Xte))["MAE"])
joblib.dump(make_pipeline(StandardScaler(), Ridge()).fit(Xtr, ytr), "../saidas/pipeline.joblib")
print("pipeline carregado:", avaliar(yte, joblib.load("../saidas/pipeline.joblib").predict(Xte))["MAE"])'''),

    ("tarefa",
     "**Divisão repetível.** Separe os clientes por `lado_por_hash(cliente_id)`. Mostre que nenhum cliente aparece "
     "nos dois lados (o que acontece com a divisão aleatória por **pedido**: o modelo \"conhece\" o cliente). Depois "
     "acrescente 10 mil pedidos novos e confira que os clientes antigos **não mudam de lado**.",
     r'''lado = df.cliente_id.map(lado_por_hash)
print("clientes nos dois lados (hash):", len(set(df[lado == "treino"].cliente_id) & set(df[lado == "teste"].cliente_id)))
aleat = np.random.default_rng(0).random(len(df)) < .8
print("clientes nos dois lados (aleatório por pedido):", len(set(df[aleat].cliente_id) & set(df[~aleat].cliente_id)))
novos = pedidos(10000, semente=99, inicio="2025-07-01", meses=1)
antes = dict(zip(df.cliente_id, lado)); depois = {c: lado_por_hash(c) for c in novos.cliente_id}
print("clientes que mudaram de lado:", sum(antes[c] != depois[c] for c in depois if c in antes))'''),

    ("tarefa",
     "**Ponte de esquema e janela.**\n"
     "(a) A partir de junho, o app passou a registrar a `gorjeta`. Para usar os dados antigos, imite a "
     "distribuição: preencha a gorjeta dos pedidos antigos sorteando dos valores recentes (*bridged schema*) e "
     "compare com zerar.\n"
     "(b) Calcule `pedidos_ultima_hora` (a carga do restaurante) e veja quanto essa feature de janela melhora o "
     "MAE de um `HistGradientBoostingRegressor`.",
     r'''from sklearn.ensemble import HistGradientBoostingRegressor
r_ = np.random.default_rng(0)
gorjeta_recente = np.round(r_.gamma(1.5, 2, (df.momento >= "2025-06-01").sum()), 1)
ponte = df.copy(); ponte["gorjeta"] = np.nan; ponte.loc[ponte.momento >= "2025-06-01", "gorjeta"] = gorjeta_recente
zerada = ponte.gorjeta.fillna(0); imitada = ponte.gorjeta.copy(); falta = imitada.isna()
imitada[falta] = r_.choice(gorjeta_recente, falta.sum())
print("média da gorjeta antiga: zerada", round(zerada[falta].mean(), 2), "| imitada", round(imitada[falta].mean(), 2), "| recente", round(gorjeta_recente.mean(), 2))
df["carga_restaurante"] = pedidos_ultima_hora(df)
tr, te = separar_no_tempo(df)
for cols in [NUMERICAS, NUMERICAS + ["carga_restaurante"]]:
    m = HistGradientBoostingRegressor(random_state=0).fit(tr[cols], tr.tempo_entrega_min)
    print(f"{len(cols)} colunas: MAE {avaliar(te.tempo_entrega_min, m.predict(te[cols]))['MAE']}")'''),

    ("tarefa",
     "**Workflow com cache e feature store.**\n"
     "(a) Monte um `Workflow` com três passos: `dados` → `features` → `modelo`. Rode duas vezes: na segunda, tudo "
     "vem do cache. Apague o cache do passo `modelo` e rode de novo: só ele reexecuta.\n"
     "(b) Crie uma feature store com a **nota média acumulada** de cada restaurante ao longo do tempo. Faça a "
     "junção *offline* correta no tempo (`merge_asof`) e compare com o vazamento de usar a média do ano inteiro.",
     r'''shutil.rmtree("../saidas/cache", ignore_errors=True)
wf = Workflow("../saidas/cache")
wf.passo("dados", lambda: separar_no_tempo(carregar()))
wf.passo("features", lambda d: (X_y(d[0]), X_y(d[1])), "dados")
wf.passo("modelo", lambda f: HistGradientBoostingRegressor(random_state=0).fit(*f[0]), "features")
print(wf.rodar()[1]); print(wf.rodar()[1])
(Path("../saidas/cache") / "modelo.joblib").unlink(); print(wf.rodar()[1])
hist = df.sort_values("momento")[["restaurante_id", "momento", "nota"]].copy()
hist["nota_media_rest"] = hist.groupby("restaurante_id").nota.transform(lambda s: s.expanding().mean())
fs = FeatureStore(hist, "restaurante_id", "momento", ["nota_media_rest"])
off = fs.offline(df[["pedido_id", "restaurante_id", "momento", "nota"]])
vaz = df.restaurante_id.map(df.groupby("restaurante_id").nota.mean())
print("correlação com a nota do próprio pedido: no ponto do tempo", round(off.nota_media_rest.corr(off.nota), 3), "| com vazamento", round(vaz.corr(df.nota), 3))
print("online (último valor do restaurante 1):", fs.online(df.restaurante_id.iloc[0]))'''),

    ("tarefa",
     "**Versionar modelos.** Com um `Registro(\"../saidas/registro\")` (limpe a pasta antes), salve o Ridge como "
     "`tempo_entrega` v1 e o boosting com a carga do restaurante como v2, cada um com o MAE nos metadados. Carregue "
     "a versão mais recente e a v1 explicitamente: são os dois modelos que um teste A/B serviria lado a lado.",
     r'''shutil.rmtree("../saidas/registro", ignore_errors=True)
reg = Registro("../saidas/registro")
v1 = make_pipeline(StandardScaler(), Ridge()).fit(tr[NUMERICAS], tr.tempo_entrega_min)
cols2 = NUMERICAS + ["carga_restaurante"]; v2 = HistGradientBoostingRegressor(random_state=0).fit(tr[cols2], tr.tempo_entrega_min)
print(reg.salvar(v1, "tempo_entrega", {"MAE": avaliar(te.tempo_entrega_min, v1.predict(te[NUMERICAS]))["MAE"], "colunas": NUMERICAS}))
print(reg.salvar(v2, "tempo_entrega", {"MAE": avaliar(te.tempo_entrega_min, v2.predict(te[cols2]))["MAE"], "colunas": cols2}))
for versao in [None, "v1"]:
    m, meta = reg.carregar("tempo_entrega", versao); print(versao or "mais recente", "->", type(m).__name__, meta["metricas"]["MAE"])'''),
]
