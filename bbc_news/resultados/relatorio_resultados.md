# Resultados — BBC News

2225 notícias, 99 duplicatas removidas → 2126. Treino/validação/teste = 1275/425/426. TF-IDF com 1000 termos + 2 atributos estruturados.

Alpha escolhido: {'LR sem regularizacao': 0.0, 'LR Elastic Net': 1e-05, 'LR Lasso': 1e-05, 'LR Ridge': 0.0001}. Profundidade AD escolhida: 10.

## Modelos selecionados (teste)
|Modelo|Precision macro|Recall macro|F1 micro|F1 macro|
|---|---:|---:|---:|---:|
|curva_lr|LR sem regularizacao|seed=42|fracao=1.0|prof=None|0.9701|0.9609|0.9648|0.9647|
|curva_lr|LR Elastic Net|seed=42|fracao=1.0|prof=None|0.9675|0.9582|0.9624|0.9621|
|curva_ad|AD propria|seed=42|fracao=1.0|prof=10|0.7755|0.7362|0.7512|0.7444|

Baseline: F1 micro=0.2371; F1 macro=0.0767

## Consistência
| familia    |   fracao_treino |   n_treino |   treino_media |   validacao_media |   validacao_desvio |   gap_medio |
|:-----------|----------------:|-----------:|---------------:|------------------:|-------------------:|------------:|
| AD propria |             0.3 |        382 |         0.7831 |            0.686  |             0.0193 |      0.0971 |
| AD propria |             0.6 |        765 |         0.7839 |            0.682  |             0.0224 |      0.1019 |
| AD propria |             0.9 |       1148 |         0.7926 |            0.7175 |             0.0155 |      0.0751 |
| LR Ridge   |             0.3 |        382 |         0.9886 |            0.9367 |             0.0097 |      0.052  |
| LR Ridge   |             0.6 |        765 |         0.9832 |            0.9572 |             0.0062 |      0.026  |
| LR Ridge   |             0.9 |       1148 |         0.981  |            0.962  |             0.0022 |      0.019  |

## Erros
- curva_lr|LR sem regularizacao|seed=42|fracao=1.0|prof=None: confusões [(4, 'tech', 'business'), (4, 'politics', 'business'), (3, 'entertainment', 'business')]; recall {'business': 0.9802, 'entertainment': 0.9459, 'politics': 0.9506, 'sport': 1.0, 'tech': 0.9275}
- curva_lr|LR Elastic Net|seed=42|fracao=1.0|prof=None: confusões [(4, 'tech', 'business'), (4, 'politics', 'business'), (3, 'entertainment', 'business')]; recall {'business': 0.9802, 'entertainment': 0.9324, 'politics': 0.9506, 'sport': 1.0, 'tech': 0.9275}
- curva_ad|AD propria|seed=42|fracao=1.0|prof=10: confusões [(16, 'entertainment', 'sport'), (13, 'tech', 'sport'), (13, 'politics', 'sport')]; recall {'business': 0.7129, 'entertainment': 0.6892, 'politics': 0.7531, 'sport': 0.9604, 'tech': 0.5652}

## Versões
{"python": "3.13.16", "numpy": "2.5.3", "pandas": "3.0.5", "scikit_learn": "1.9.1", "scipy": "1.18.1"}