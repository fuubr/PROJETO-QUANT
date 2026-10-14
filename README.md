# Projeto QUANT — um modelo de IA para trading testado contra a própria vontade

![Testes](https://github.com/fuubr/projeto-quant/actions/workflows/tests.yml/badge.svg)

Pipeline completo em Python — dados reais, modelo de machine learning calibrado, gestão de
risco, paper trading automático diário — construído com uma regra: **todo resultado só vale
se sobreviver a um controle feito para derrubá-lo.**

**Resultado até agora: o modelo não mostrou informação.** Com ablação, baselines justos e
correção por múltiplos testes, nenhuma estratégia superou o buy-and-hold com evidência
estatística. O projeto existe para mostrar *como* se chega a essa conclusão de forma
confiável — e para testá-la ao vivo, com hipóteses pré-registradas.

---

## A pergunta

Um classificador (XGBoost + regressão logística, probabilidades calibradas) consegue prever a
direção de 20 dias de SPY, AAPL, PETR4 e VALE3 bem o bastante para que uma estratégia baseada
nele (dimensionamento por Kelly fracionário, stop-loss, circuit breaker) bata alternativas
simples?

## Os controles

| Controle | O que ele derruba |
|---|---|
| **Ativo sintético** (random walk, edge zero por construção) | Qualquer "acerto" aqui é bug ou sorte: o pipeline precisa *falhar* nele |
| **Walk-forward purgado** | Vazamento de futuro entre treino e teste (rótulos de 20 dias se sobrepõem) |
| **Holdout 2024+ intocado** | Ajuste fino no período de teste; só pode ser gasto uma vez |
| **Newey-West + Bonferroni** | Autocorrelação e múltiplas comparações inflando significância |
| **Ablação** | O mesmo Kelly e stops, mas com probabilidade *sem informação*: isola o valor do modelo |
| **Baselines justos** | Buy-and-hold puro, exposição igualada, vol-target — com caixa rendendo CDI/T-bill **reais** (BCB, FRED) |
| **Deflated Sharpe** | O melhor de muitas tentativas parece bom por acaso |
| **Pré-registro** ([`docs/CRITERIOS.md`](docs/CRITERIOS.md)) | Mudar a régua depois de ver o resultado |

## Resultados (2018–2023, fora da amostra)

| | SPY | AAPL | PETR4 | VALE3 |
|---|---|---|---|---|
| Modelo − ablação (retorno a.a.) | +1,8% (p=0,16) | +0,4% | −0,4% | +0,1% |
| Sharpe de excesso: estratégia do modelo | 0,49 | 0,68 | −0,25 | 0,00 |
| Sharpe de excesso: buy-and-hold | 0,53 | 0,93 | 0,67 | 0,41 |
| Drawdown máx.: vol-target (sem modelo) | −13% | −12% | −15% | −17% |
| Drawdown máx.: buy-and-hold | −34% | −39% | −63% | −44% |

- **O modelo não agrega informação**: trocar suas probabilidades por uma constante quase não muda nada.
- **O que a estratégia entrega é menos risco, não mais retorno por risco.** Nos ativos
  brasileiros, rendeu menos que deixar o dinheiro no CDI.
- **Nenhuma estratégia atinge Deflated Sharpe ≥ 0,95.**
- Uma comparação anterior ("modelo +18,9% contra buy-and-hold −2,8%") era **enganosa**: o
  buy-and-hold usado tinha circuit breaker e ficou fora da recuperação. Corrigida e documentada.

Detalhes: [`docs/leitura_baselines.md`](docs/leitura_baselines.md) ·
números completos: [`docs/analise_baselines.md`](docs/analise_baselines.md)

## O teste de verdade: paper trading ao vivo

Desde agosto/2026, um GitHub Action roda todo dia útil às 19h (BRT), sem nenhum computador
ligado: baixa preços, decide posições, grava o log e commita. Três hipóteses, pré-registradas
antes de qualquer dado ao vivo:

- **H1** — o modelo de IA tem informação (supera a previsão ingênua e a ablação).
- **H2** — vol-target: exposição = 10% ÷ volatilidade realizada, sem modelo.
- **H3** — vol-target + filtro de tendência de 12 meses.

Primeira avaliação: ~6 meses de dados (abril/2027), com critérios de sucesso **e de parada**
fixados antes. Um relatório semanal de drift diz "INSUFICIENTE" até lá — por construção.

## O que os testes pegaram

202 testes automatizados (CI em todo push). Alguns bugs reais que eles encontraram:

- **Decisão com um dia de atraso** no paper trading: a linha "de hoje" era descartada por
  falta de rótulo futuro. Achado pela checagem de consistência backtest × ao vivo.
- **Preços que mudavam no passado**: o yfinance reajusta o histórico por dividendos (até 2,7%).
  Corrigido com uma série canônica só de acréscimo, encadeada por razões de preço.
- **Vazamento de holdout** no walk-forward da etapa de risco, achado pela contagem de folds.
- **Intervalo de confiança de largura zero** com um único bloco de bootstrap — parecia preciso.
- **Gate estatístico ingênuo** reprovando o ativo sintético por sorte (seed 303): corrigido
  agregando seeds, não trocando de seed.

Histórico completo de decisões e correções: [`docs/HISTORICO.md`](docs/HISTORICO.md)

## Como rodar

```bash
pip install -r requirements.txt
python -m pytest                          # suíte completa
python run_baseline_ablation.py           # baselines, ablação, H2/H3, Deflated Sharpe
python run_stage6_paper_trading_daily.py  # um dia de paper trading
python run_stage6_report.py               # resumo + dashboard.html
python run_stage7_drift_check.py          # relatório de drift
```
`run_stage3_holdout_evaluation.py` **gasta o holdout** — só uma vez, com o pipeline final.

## Estrutura

```
data/           preços (yfinance), taxas livres de risco (BCB, FRED), ativo sintético
models/         features, rótulos, calibração, walk-forward purgado
backtest/       motor, estatística (Newey-West, Bonferroni, Deflated Sharpe, Memmel)
risk/           Kelly fracionário, stop-loss, circuit breaker
strategies/     regras sem modelo (H2, H3)
paper_trading/  estado, preços canônicos, consistência, drift, retreino challenger
.github/        paper trading diário, drift semanal, testes
```

**Stack:** Python · pandas · scikit-learn · XGBoost · SHAP · SciPy · GitHub Actions

## Limitações

Um único período, majoritariamente de alta. Quatro ativos que equivalem a ~2 amostras
independentes. Custos de transação estimados, não medidos. O período 2015–2023 foi consultado
muitas vezes — serve para descartar hipóteses, não para escolhê-las.

## Licenca

MIT -- ver [`LICENSE`](LICENSE). Nao e recomendacao de investimento.
