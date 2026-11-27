"""Dados fictícios do Rango, um aplicativo de entrega de comida em duas cidades (gerados com semente fixa).

Use nos cadernos com `from dados import *` (a Preparação já faz isso).

  pedidos(n, cidade, semente, inicio, meses, drift)   um pedido por linha, com o tempo de entrega (alvo)
  restaurantes()                                      2.000 restaurantes (o id tem alta cardinalidade)
  TEMPO_PREPARO                                       tabela determinística de preparo por culinária e itens

Colunas de pedidos:
  pedido_id, momento, cidade, cliente_id, restaurante_id, culinaria, bairro, regiao (central/periferia),
  distancia_km, itens, valor, chuva_mm, hora, dia_semana, entregador_exp_meses,
  tempo_entrega_min (alvo), atrasou (>45 min), cancelado (raro), nota (1-5), comentario
"""
import numpy as np
import pandas as pd

CULINARIAS = ["pizza", "japonesa", "hamburguer", "brasileira", "árabe", "vegana", "doces", "chinesa"]
PREPARO_BASE = {"pizza": 18, "japonesa": 22, "hamburguer": 12, "brasileira": 15, "árabe": 14, "vegana": 16, "doces": 8, "chinesa": 17}
TEMPO_PREPARO = {(c, i): PREPARO_BASE[c] + 2.5 * (i - 1) for c in CULINARIAS for i in range(1, 9)}
BAIRROS = {"Curitiba": [("Centro", "central"), ("Batel", "central"), ("Água Verde", "central"), ("Cabral", "central"),
                        ("CIC", "periferia"), ("Sítio Cercado", "periferia"), ("Boqueirão", "periferia"), ("Tatuquara", "periferia")],
           "Londrina": [("Centro", "central"), ("Gleba Palhano", "central"), ("Cinco Conjuntos", "periferia"), ("Vivi Xavier", "periferia")]}
_COMENTARIOS = {"bom": ["chegou rápido e quente", "comida ótima, entregador educado", "perfeito como sempre", "embalagem caprichada"],
                "neutro": ["ok, nada demais", "chegou no tempo previsto", "comida razoável"],
                "ruim": ["demorou muito", "chegou frio", "faltou item no pedido", "entregador errou o endereço", "veio derramado"]}


def restaurantes(semente=1):
    r = np.random.default_rng(semente)
    return pd.DataFrame({"restaurante_id": np.arange(1, 2001), "culinaria": r.choice(CULINARIAS, 2000),
                         "qualidade": np.round(r.normal(0, 1, 2000), 3),        # oculto: afeta atraso e nota
                         "cidade": r.choice(["Curitiba", "Londrina"], 2000, p=[.8, .2])})


def pedidos(n=60000, cidade="Curitiba", semente=7, inicio="2025-01-01", meses=6, drift=0.0):
    """`drift` > 0 piora o trânsito (mais tempo por km) para simular mudança no mundo depois do treino."""
    r = np.random.default_rng(semente)
    rest = restaurantes()
    rest = rest[rest.cidade == cidade].reset_index(drop=True)
    pop = r.pareto(1.2, len(rest)) + 1
    ri = r.choice(len(rest), n, p=pop / pop.sum())
    bairros = BAIRROS[cidade]
    bi = r.integers(0, len(bairros), n)
    momento = pd.Timestamp(inicio) + pd.to_timedelta(r.integers(0, meses * 30 * 24 * 60, n), unit="min")
    hora = np.where(r.random(n) < .55, r.normal(20, 1.3, n), r.normal(12.5, 1.0, n)).clip(10, 23.9).astype(int)
    momento = momento.normalize() + pd.to_timedelta(hora, unit="h") + pd.to_timedelta(r.integers(0, 60, n), unit="min")
    dia = momento.dayofweek
    periferia = np.array([bairros[b][1] == "periferia" for b in bi])
    dist = np.round(np.clip(r.gamma(2.2, 1.4, n) + 2.0 * periferia, 0.3, 20), 2)
    itens = np.clip(r.poisson(1.8, n) + 1, 1, 8)
    culinaria = rest.culinaria.to_numpy()[ri]
    qual = rest.qualidade.to_numpy()[ri]
    chuva = np.where(r.random(n) < .25, np.round(r.gamma(1.5, 6, n), 1), 0.0)
    exp = r.integers(0, 60, n)
    pico = ((hora >= 19) & (hora <= 21)) & (dia >= 4)                         # sexta a domingo, 19h-21h
    preparo = np.array([TEMPO_PREPARO[(c, i)] for c, i in zip(culinaria, itens)])
    tempo = (preparo + dist * (3.2 + drift) + 6 * pico + 0.25 * chuva - 0.06 * exp - 2.0 * qual
             + 4 * periferia * pico + r.normal(0, 4, n))
    tempo = np.round(np.clip(tempo, 8, None), 1)
    valor = np.round(itens * r.normal(38, 8, n).clip(15) + 7, 2)
    z_cancel = -4.6 + 0.04 * (tempo - 35) + 0.8 * (chuva > 15) - 0.5 * qual
    cancelado = (r.random(n) < 1 / (1 + np.exp(-z_cancel))).astype(int)
    nota_cont = 4.3 - 0.06 * np.clip(tempo - 30, 0, None) + 0.4 * qual + r.normal(0, .5, n)
    nota = np.clip(np.round(nota_cont), 1, 5).astype(int)
    comentario = [r.choice(_COMENTARIOS["bom" if nt >= 4 else "neutro" if nt == 3 else "ruim"]) for nt in nota]
    df = pd.DataFrame({
        "pedido_id": np.arange(1, n + 1), "momento": momento, "cidade": cidade,
        "cliente_id": r.integers(1, 20001 if cidade == "Curitiba" else 5001, n),
        "restaurante_id": rest.restaurante_id.to_numpy()[ri], "culinaria": culinaria,
        "bairro": [bairros[b][0] for b in bi], "regiao": np.where(periferia, "periferia", "central"),
        "distancia_km": dist, "itens": itens, "valor": valor, "chuva_mm": chuva, "hora": hora, "dia_semana": dia,
        "entregador_exp_meses": exp, "tempo_entrega_min": tempo, "atrasou": (tempo > 45).astype(int),
        "cancelado": cancelado, "nota": nota, "comentario": comentario})
    df = df.sort_values("momento").reset_index(drop=True)
    df["pedido_id"] = np.arange(1, n + 1) + (0 if cidade == "Curitiba" else 5_000_000)
    return df
