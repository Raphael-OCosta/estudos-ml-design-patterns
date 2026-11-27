# Estudos — *Machine Learning Design Patterns* (Lakshmanan, Robinson e Munn) · trilha prática

> **A melhor forma de estudar um livro técnico é ler e praticar ao mesmo tempo.**
> Aqui a prática vem primeiro. Cada capítulo é uma **missão** curta: você lê só o essencial e aplica os padrões em código.

O livro reúne 30 padrões de projeto de ML: soluções testadas para problemas que todo projeto encontra (representar dados, formular o problema, treinar, servir, reproduzir e agir com responsabilidade). Os exemplos do livro usam Google Cloud. Aqui cada padrão roda **no seu computador** com scikit-learn, pandas e Flask, sobre 60 mil pedidos fictícios do **Rango**, um app de entrega de comida em Curitiba e Londrina.

## A trilha (42 passos)

| Cap. | Missão | Você constrói |
|---|---|---|
| 1 | Por que padrões | linha de base, sementes, dados sujos, **drift** medido |
| 2 | Representação dos dados | **Hashed Feature**, **Embeddings** (SVD), **Feature Cross**, **Multimodal** (números + TF-IDF) |
| 3 | Representação do problema | **Reframing** em quantis, **Multilabel**, **Cascade**, **Neutral Class**, **Rebalancing** |
| 4 | Treino | **Useful Overfitting**, **Checkpoints** e retomada, **Transfer Learning** entre cidades, SGD paralelo, halving |
| 5 | Serving | **API sem estado** com chave (Flask), **lote**, **avaliação contínua** com retreino, **duas fases** |
| 6 | Reprodutibilidade | **Transform**, **divisão por hash**, **ponte de esquema**, **janela**, **workflow** com cache, **feature store**, **versões** |
| 7 | IA responsável | **referência heurística**, explicações global e local, **lente da justiça** com correção de promessa |
| 8 | Padrões conectados | sistema completo com portões de promoção (MAE, heurística, viés na periferia) |

## Como usar

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-estudos.txt
.venv\Scripts\python.exe ferramentas\testar_cadernos.py   # opcional: roda o gabarito (0 erros)
```

Abra `praticar/cap01_*.ipynb` no VS Code e escolha o kernel `.venv`. Os dados são gerados na hora (sem download). O [Guia de Estudos (PDF)](Guia_de_Estudos.pdf) mostra o mapa e a rotina.

## Direitos autorais

O livro **não** está neste repositório. Use a sua própria cópia.
As missões, os cadernos, os dados fictícios, os scripts e o guia são autorais, sob licença [MIT](LICENSE).
