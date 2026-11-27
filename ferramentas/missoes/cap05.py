TITULO = "Serving resiliente: API sem estado, lote, avaliação contínua"
MISSAO = (
    "Capítulo 5, padrões 16 a 20: pôr o modelo para trabalhar e **mantê-lo** confiável. Você vai construir:\n"
    "- uma **função de serving sem estado** (API Flask com o pipeline inteiro);\n"
    "- o **serving em lote** para o painel diário;\n"
    "- a **avaliação contínua**, que mede o erro mês a mês e dispara o retreino quando o mundo muda;\n"
    "- a **previsão em duas fases**: um modelo pequeno no celular e um grande no servidor;\n"
    "- as **previsões com chave**, que voltam com o id e não dependem da ordem."
)
ENTREGA = "o módulo `PKG/servico.py` (API stateless com chave, lote em partes e monitor de drift com gatilho) e o retreino disparado sozinho quando o trânsito piora."
LEITURA = [("Design Pattern 16: Stateless Serving Function", 244), ("Design Pattern 17: Batch Serving", 258),
           ("Design Pattern 18: Continued Model Evaluation", 265), ("Design Pattern 19: Two-Phase Predictions", 280),
           ("Design Pattern 20: Keyed Predictions", 293)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Sem estado:** a função de serving recebe tudo o que precisa na requisição e não guarda nada entre "
     "chamadas. Por isso dá para ter 1 ou 100 cópias atrás de um balanceador.\n"
     "- **Com chave:** a resposta carrega o id de quem perguntou. Num lote processado em paralelo, a ordem se "
     "perde, e a chave não."),

    ("arquivo",
     "**O serviço.** Crie `servico.py` com:\n"
     "- `criar_app(modelo, colunas)`: um Flask com `POST /prever`, que recebe uma **lista** de pedidos (cada um com "
     "`pedido_id` e as colunas), devolve `[{\"pedido_id\": ..., \"minutos\": ...}]` e recusa com 400 quando falta "
     "coluna;\n"
     "- `prever_em_lote(modelo, df, colunas, tamanho=20000)`: processa em partes e devolve um DataFrame com a "
     "chave;\n"
     "- `Monitor(limite_mae)`: `registrar(mes, y, p)` guarda o MAE e o viés do mês; `precisa_retreinar()` diz se "
     "os dois últimos meses **desde o último retreino** passaram do limite; `retreinou()` marca o retreino.",
     "servico.py",
     r'''"""Serving: API sem estado com chave, lote em partes e monitor contínuo."""
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


def criar_app(modelo, colunas):
    app = Flask(__name__)

    @app.post("/prever")
    def prever():
        pedidos = request.get_json(silent=True)
        if not isinstance(pedidos, list) or not pedidos:
            return jsonify(erro="envie uma lista de pedidos"), 400
        faltando = sorted({c for p in pedidos for c in colunas + ["pedido_id"] if c not in p})
        if faltando:
            return jsonify(erro=f"faltam colunas: {faltando}"), 400
        df = pd.DataFrame(pedidos)
        minutos = modelo.predict(df[colunas])
        return jsonify([{"pedido_id": int(i), "minutos": round(float(m), 1)} for i, m in zip(df.pedido_id, minutos)])

    return app


def prever_em_lote(modelo, df, colunas, tamanho=20000):
    partes = [pd.DataFrame({"pedido_id": p.pedido_id.to_numpy(), "minutos": modelo.predict(p[colunas])})
              for _, p in df.groupby(np.arange(len(df)) // tamanho)]
    return pd.concat(partes, ignore_index=True)


class Monitor:
    def __init__(self, limite_mae):
        self.limite, self.historico, self.desde = limite_mae, [], 0

    def registrar(self, mes, y, p):
        y, p = np.asarray(y), np.asarray(p)
        self.historico.append({"mes": mes, "MAE": round(float(np.mean(np.abs(y - p))), 2), "vies": round(float(np.mean(y - p)), 2), "n": len(y)})

    def retreinou(self):
        self.historico[-1]["acao"] = "retreinou"
        self.desde = len(self.historico)

    def precisa_retreinar(self):
        janela = self.historico[self.desde:][-2:]
        return len(janela) == 2 and all(h["MAE"] > self.limite for h in janela)'''),

    ("tarefa",
     "**A API sem estado e com chave.** Treine um `HistGradientBoostingRegressor` nas numéricas, crie o app e, pelo "
     "`test_client`, mande 3 pedidos **fora de ordem**. Confira que cada resposta volta com o seu `pedido_id`. "
     "Mande também um pedido sem `distancia_km`: deve voltar 400.",
     r'''from sklearn.ensemble import HistGradientBoostingRegressor
from PKG.base import carregar, separar_no_tempo, X_y, avaliar, NUMERICAS
from PKG.servico import criar_app, prever_em_lote, Monitor
df = carregar(); tr, te = separar_no_tempo(df)
modelo = HistGradientBoostingRegressor(random_state=0).fit(*X_y(tr))
cli = criar_app(modelo, NUMERICAS).test_client()
lote = te.sample(3, random_state=1)[["pedido_id"] + NUMERICAS].to_dict("records")
print(cli.post("/prever", json=lote[::-1]).get_json())
ruim = {k: v for k, v in lote[0].items() if k != "distancia_km"}
r = cli.post("/prever", json=[ruim]); print(r.status_code, r.get_json())'''),

    ("tarefa",
     "**Serving em lote.** Pontue os 20 mil pedidos do teste **embaralhados** com o `prever_em_lote` (partes de 5 "
     "mil) e junte de volta pela chave. Confira que o MAE é o mesmo da previsão direta. Meça o tempo por pedido do "
     "lote e de 200 chamadas individuais à API.",
     r'''emb = te.sample(frac=1, random_state=2)
t = time.perf_counter(); res = prever_em_lote(modelo, emb, NUMERICAS, 5000); t_lote = (time.perf_counter() - t) / len(emb)
junto = te.merge(res, on="pedido_id")
print("MAE pela chave:", avaliar(junto.tempo_entrega_min, junto.minutos)["MAE"], "| direto:", avaliar(te.tempo_entrega_min, modelo.predict(X_y(te)[0]))["MAE"])
t = time.perf_counter()
for reg in te.head(200)[["pedido_id"] + NUMERICAS].to_dict("records"):
    cli.post("/prever", json=[reg])
print(f"por pedido: lote {t_lote * 1000:.4f} ms | API um a um {(time.perf_counter() - t) / 200 * 1000:.2f} ms")'''),

    ("tarefa",
     "**Avaliação contínua com gatilho.** Gere os meses de julho a dezembro com o trânsito piorando aos poucos "
     "(`drift` de 0 a 1,5). Registre o MAE e o viés de cada mês num `Monitor(limite_mae=5.5)`. Quando o gatilho "
     "disparar, **retreine** com os últimos 2 meses e siga monitorando com o modelo novo. Mostre a tabela.",
     r'''mon = Monitor(5.5); atual = modelo; eventos = []
for k, (mes, drift) in enumerate(zip(["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"], [0, .3, .6, .9, 1.2, 1.5])):
    d = pedidos(8000, semente=20 + k, inicio=f"{mes}-01", meses=1, drift=drift)
    mon.registrar(mes, d.tempo_entrega_min, atual.predict(X_y(d)[0])); eventos.append(d)
    if mon.precisa_retreinar():
        recente = pd.concat(eventos[-2:])
        atual = HistGradientBoostingRegressor(random_state=0).fit(*X_y(recente))
        mon.retreinou()
display(pd.DataFrame(mon.historico))'''),

    ("tarefa",
     "**Duas fases.** No celular cabe só um modelo minúsculo (`DecisionTreeRegressor(max_depth=3)`) que responde na "
     "hora. Se a previsão dele passar de um limite, a pergunta vai para o modelo grande no servidor. Varra o limite (25 a 45 min) e mostre a troca entre o MAE e a fração de pedidos que vão ao servidor.",
     r'''from sklearn.tree import DecisionTreeRegressor
pequeno = DecisionTreeRegressor(max_depth=3, random_state=0).fit(*X_y(tr))
Xte, yte = X_y(te); p1 = pequeno.predict(Xte); p2 = modelo.predict(Xte)
print("só pequeno:", avaliar(yte, p1)["MAE"], "| só grande:", avaliar(yte, p2)["MAE"])
for limite in [25, 30, 35, 40, 45]:
    vai = p1 > limite
    print(f"limite {limite}: MAE {avaliar(yte, np.where(vai, p2, p1))['MAE']} | foram ao servidor {vai.mean():.1%}")'''),
]
