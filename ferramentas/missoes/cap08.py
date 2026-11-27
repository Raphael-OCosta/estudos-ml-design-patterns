TITULO = "Padrões conectados: o sistema completo"
MISSAO = (
    "O último capítulo mostra que os padrões funcionam **em conjunto**. Você vai montar, com os seus módulos, o "
    "sistema de tempo de entrega do Rango de ponta a ponta. A sequência:\n"
    "1. divisão repetível;\n"
    "2. features de janela;\n"
    "3. Transform (pipeline);\n"
    "4. ajuste de hiperparâmetros;\n"
    "5. comparação com a heurística;\n"
    "6. registro de versão;\n"
    "7. serving sem estado com chave;\n"
    "8. monitor contínuo que decide o retreino;\n"
    "9. lente de justiça antes de promover a versão nova."
)
ENTREGA = "o script `PKG/sistema.py` (`treinar_versao` e `promover_se_melhor`) e um ciclo completo de duas versões, decidido por métricas e por justiça."
LEITURA = [("Patterns Reference", 424), ("Pattern Interactions", 431), ("Patterns Within ML Projects", 435),
           ("Common Patterns by Use Case and Data Type", 451)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "No mundo real, um padrão puxa o outro:\n"
     "- *Repeatable Splitting* exige *Transform*;\n"
     "- *Continued Evaluation* dispara o retreino, que gera uma nova *Model Versioning*;\n"
     "- a *Fairness Lens* e o *Heuristic Benchmark* são os **portões** antes de promover a versão nova.\n\n"
     "O capítulo 8 do livro é um mapa dessas ligações."),

    ("arquivo",
     "**O sistema.** Crie `sistema.py` com:\n"
     "- `preparar(df)`: acrescenta a carga do restaurante (`pedidos_ultima_hora`) e `fim_semana`;\n"
     "- `COLUNAS`;\n"
     "- `treinar_versao(df, registro, nome=\"tempo_entrega\")`: separa por hash do cliente, ajusta um "
     "`HistGradientBoostingRegressor` com `HalvingRandomSearchCV` pequena, mede o MAE, a heurística e o viés por "
     "região no teste, e salva no registro com essas métricas;\n"
     "- `promover_se_melhor(registro, nome, max_vies_periferia=1.5)`: compara a versão nova com a anterior. Promove "
     "(grava `producao.txt`) só se o MAE melhorou, se ganhou da heurística e se o viés na periferia está abaixo do "
     "limite.",
     "sistema.py",
     r'''"""O sistema completo de tempo de entrega, ligando os padrões do livro."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from .base import NUMERICAS
from .reprodutivel import lado_por_hash, pedidos_ultima_hora

COLUNAS = NUMERICAS + ["carga_restaurante", "fim_semana"]


def preparar(df):
    df = df.copy()
    df["carga_restaurante"] = pedidos_ultima_hora(df)
    df["fim_semana"] = (df.dia_semana >= 4).astype(int)
    return df


def treinar_versao(df, registro, nome="tempo_entrega"):
    lado = df.cliente_id.map(lado_por_hash)
    tr, te = df[lado == "treino"], df[lado == "teste"]
    busca = HalvingRandomSearchCV(HistGradientBoostingRegressor(random_state=0),
                                  {"learning_rate": [.05, .1, .2], "max_leaf_nodes": [15, 31, 63]},
                                  cv=3, scoring="neg_mean_absolute_error", random_state=0).fit(tr[COLUNAS], tr.tempo_entrega_min)
    p = busca.predict(te[COLUNAS])
    heur = tr.groupby(["bairro", "hora"]).tempo_entrega_min.median()
    p_h = te.set_index(["bairro", "hora"]).index.map(heur).fillna(tr.tempo_entrega_min.median())
    erro = te.tempo_entrega_min.to_numpy() - p
    metricas = {"MAE": round(float(np.mean(np.abs(erro))), 2),
                "MAE_heuristica": round(float(np.mean(np.abs(te.tempo_entrega_min.to_numpy() - p_h))), 2),
                "vies_periferia": round(float(erro[(te.regiao == "periferia").to_numpy()].mean()), 2)}
    return registro.salvar(busca.best_estimator_, nome, metricas), metricas


def promover_se_melhor(registro, nome="tempo_entrega", max_vies_periferia=1.5):
    base = Path(registro.pasta) / nome
    versoes = sorted(base.glob("v*"), key=lambda p: int(p.name[1:]))
    nova = json.loads((versoes[-1] / "metadados.json").read_text(encoding="utf-8"))["metricas"]
    prod = base / "producao.txt"
    atual = None
    if prod.exists():
        atual = json.loads((base / prod.read_text().strip() / "metadados.json").read_text(encoding="utf-8"))["metricas"]
    portoes = {"melhor que a produção": atual is None or nova["MAE"] < atual["MAE"],
               "melhor que a heurística": nova["MAE"] < nova["MAE_heuristica"],
               "viés na periferia ok": abs(nova["vies_periferia"]) <= max_vies_periferia}
    if all(portoes.values()):
        prod.write_text(versoes[-1].name)
    return versoes[-1].name, portoes, prod.read_text().strip() if prod.exists() else None'''),

    ("tarefa",
     "**Versão 1.** Com um registro limpo em `../saidas/sistema`, prepare os pedidos de janeiro a junho, treine a "
     "versão 1 e tente promovê-la. Mostre as métricas e os portões.",
     r'''import shutil
from PKG.reprodutivel import Registro
from PKG.sistema import preparar, treinar_versao, promover_se_melhor, COLUNAS
shutil.rmtree("../saidas/sistema", ignore_errors=True); reg = Registro("../saidas/sistema")
dados = preparar(pedidos())
print(treinar_versao(dados, reg)); print(promover_se_melhor(reg))'''),

    ("tarefa",
     "**Serving e monitor.** Carregue a versão em produção e sirva com a API do cap. 5 (`criar_app`) para julho a "
     "setembro, quando o trânsito piora (`drift` de 0,6 a 1,2). Registre num `Monitor(5.5)`. O gatilho disparou?",
     r'''from PKG.servico import criar_app, Monitor
prod = (reg.pasta / "tempo_entrega" / "producao.txt").read_text().strip()
modelo, meta = reg.carregar("tempo_entrega", prod); cli = criar_app(modelo, COLUNAS).test_client()
mon = Monitor(5.5); novos = []
for k, (mes, drift) in enumerate(zip(["2025-07", "2025-08", "2025-09"], [.6, .9, 1.2])):
    d = preparar(pedidos(8000, semente=40 + k, inicio=f"{mes}-01", meses=1, drift=drift)); novos.append(d)
    resp = cli.post("/prever", json=d[["pedido_id"] + COLUNAS].to_dict("records")).get_json()
    p = d[["pedido_id"]].merge(pd.DataFrame(resp), on="pedido_id").minutos
    mon.registrar(mes, d.tempo_entrega_min, p)
print(pd.DataFrame(mon.historico)); print("retreinar?", mon.precisa_retreinar())'''),

    ("tarefa",
     "**Versão 2 e os portões.** Se o gatilho disparou, treine a versão 2 com os dados recentes (julho a setembro) e "
     "tente promovê-la. Explique num comentário por que a comparação de MAE entre versões treinadas em períodos "
     "diferentes precisa ser feita no **mesmo** conjunto, e faça essa comparação justa nos dados de setembro.",
     r'''if mon.precisa_retreinar():
    print(treinar_versao(pd.concat(novos), reg)); print(promover_se_melhor(reg))
setembro = novos[-1]
for v in ["v1", "v2"]:
    m, _ = reg.carregar("tempo_entrega", v)
    print(v, "MAE em setembro:", round(float(np.mean(np.abs(setembro.tempo_entrega_min - m.predict(setembro[COLUNAS])))), 2))
# Os MAE salvos vêm de testes de períodos diferentes (o de julho a setembro é mais difícil). A comparação justa usa o mesmo conjunto.'''),
]
