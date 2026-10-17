# Projeto 2 — Classificação de notícias da BBC

Teste com a base **BBC News** (2.225 notícias, 5 categorias: business, entertainment, politics, sport, tech), com texto livre de verdade para o TF-IDF.

## Arquivos

- `data/bbc-text.csv`: base (colunas `category` e `text`).
- `BBCNewsClassificacao.ipynb`: notebook completo, já executado.
- `bbc_classificacao.py`: mesmo código em script; imprime as métricas no terminal.
- `resultados/`: figuras (EDA, busca de alpha, 5 curvas, 3 matrizes de confusão), CSVs e `relatorio_resultados.md`.

## Executar

```bash
pip install numpy pandas scipy scikit-learn matplotlib tabulate
python bbc_classificacao.py
```

Leva cerca de 2 a 3 minutos.

## Resumo do protocolo

- 99 notícias duplicadas removidas (evita vazamento entre treino e teste), restando 2.126.
- Divisão 60/20/20 estratificada: 1.275 / 425 / 426.
- TF-IDF (1.000 termos, uni e bigramas, sem stopwords) + 2 atributos estruturados (nº de palavras e tamanho médio das palavras).
- LR-GD (`SGDClassifier`, log-loss) sem penalidade, Ridge, Lasso e Elastic Net, com alpha escolhido na validação.
- Árvore de decisão própria (CART/Gini), com profundidade escolhida pela curva 1–12.
- Consistência com 30/60/90% do treino e 3 sementes; precision, recall e F1 micro/macro no teste para os 81 modelos.
