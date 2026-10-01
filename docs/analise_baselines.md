# Analise de baselines justos e ablacao

Fora da amostra (walk-forward purgado), periodo de desenvolvimento; holdout intocado. Gerado por `run_baseline_ablation.py`. `retorno`/`sharpe` tratam o caixa a 0%; `retorno_c/_caixa` e `sharpe_excesso_rf` usam a taxa livre de risco REAL (BCB CDI para .SA, FRED T-bill 3m para os demais; a fonte aparece em cada secao).

## SPY (2018-03-22 a 2023-09-22, 1386 dias)

Taxa livre de risco / caixa: **FRED DTB3**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|
| kelly_modelo (c/ stops)       |    0.1824 |    0.571 |     0.695 |  -0.0874 |             0.221 |          100   |             0.2815 |               0.523 |
| ablacao_sem_modelo (c/ stops) |    0.0629 |    0.344 |     0.409 |  -0.0717 |             0.177 |          100   |             0.1505 |               0.258 |
| exposicao_igualada (s/ stops) |    0.1607 |    0.614 |     0.753 |  -0.0821 |             0.221 |          100   |             0.2501 |               0.531 |
| vol_target (s/ stops)         |    0.4823 |    0.739 |     0.969 |  -0.1289 |             0.687 |          100   |             0.5247 |               0.622 |
| buy_and_hold_puro (s/ stops)  |    0.7896 |    0.614 |     0.753 |  -0.3372 |             1     |          100   |             0.7896 |               0.531 |
| buy_and_hold (c/ breaker)     |   -0.0279 |    0.011 |     0.007 |  -0.2671 |             0.358 |           35.8 |             0.0287 |              -0.052 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +2.04% a.a., t=1.55, p=0.122, significativo=False
- modelo - exposicao_igualada: +0.39% a.a., t=0.37, p=0.713, significativo=False

## AAPL (2018-03-22 a 2023-09-22, 1386 dias)

Taxa livre de risco / caixa: **FRED DTB3**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|
| kelly_modelo (c/ stops)       |    0.2209 |    0.65  |     0.83  |  -0.0716 |             0.152 |          100   |             0.3307 |               0.622 |
| ablacao_sem_modelo (c/ stops) |    0.2198 |    0.793 |     1.075 |  -0.0572 |             0.143 |          100   |             0.3246 |               0.743 |
| exposicao_igualada (s/ stops) |    0.298  |    0.986 |     1.367 |  -0.068  |             0.152 |          100   |             0.4072 |               0.933 |
| vol_target (s/ stops)         |    1.1985 |    1.334 |     2.051 |  -0.1238 |             0.4   |          100   |             1.3216 |               1.269 |
| buy_and_hold_puro (s/ stops)  |    3.3531 |    0.986 |     1.367 |  -0.3852 |             1     |          100   |             3.3531 |               0.933 |
| buy_and_hold (c/ breaker)     |    0.0585 |    0.156 |     0.071 |  -0.2351 |             0.122 |           12.2 |             0.1489 |               0.131 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.08% a.a., t=0.08, p=0.935, significativo=False
- modelo - exposicao_igualada: -1.07% a.a., t=-0.87, p=0.385, significativo=False

## PETR4.SA (2018-04-02 a 2023-10-30, 1386 dias)

Taxa livre de risco / caixa: **BCB CDI (SGS 12)**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|
| kelly_modelo (c/ stops)       |   -0.0455 |   -0.235 |    -0.041 |  -0.1255 |             0.008 |            2.8 |             0.4085 |              -0.25  |
| ablacao_sem_modelo (c/ stops) |   -0.0215 |   -0.315 |    -0.055 |  -0.05   |             0.003 |            2.8 |             0.4464 |              -0.33  |
| exposicao_igualada (s/ stops) |    0.0173 |    0.816 |     0.989 |  -0.0069 |             0.008 |          100   |             0.5005 |               0.666 |
| vol_target (s/ stops)         |    0.7076 |    0.922 |     1.25  |  -0.1451 |             0.286 |          100   |             1.2586 |               0.741 |
| buy_and_hold_puro (s/ stops)  |    3.4378 |    0.816 |     0.989 |  -0.6336 |             1     |          100   |             3.4378 |               0.666 |
| buy_and_hold (c/ breaker)     |   -0.0392 |   -0.029 |    -0.006 |  -0.2662 |             0.027 |            2.7 |             0.4087 |              -0.047 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: -0.40% a.a., t=-0.26, p=0.794, significativo=False
- modelo - exposicao_igualada: -1.10% a.a., t=-0.48, p=0.628, significativo=False

## VALE3.SA (2018-04-02 a 2023-10-30, 1386 dias)

Taxa livre de risco / caixa: **BCB CDI (SGS 12)**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|
| kelly_modelo (c/ stops)       |    0.0387 |    0.131 |     0.124 |  -0.1466 |             0.146 |          100   |             0.4577 |               0.001 |
| ablacao_sem_modelo (c/ stops) |    0.0442 |    0.235 |     0.296 |  -0.0753 |             0.091 |          100   |             0.4923 |               0.061 |
| exposicao_igualada (s/ stops) |    0.1972 |    0.594 |     0.838 |  -0.0722 |             0.146 |          100   |             0.6731 |               0.414 |
| vol_target (s/ stops)         |    0.3408 |    0.529 |     0.747 |  -0.1739 |             0.316 |          100   |             0.7495 |               0.326 |
| buy_and_hold_puro (s/ stops)  |    1.3656 |    0.594 |     0.838 |  -0.4372 |             1     |          100   |             1.3656 |               0.414 |
| buy_and_hold (c/ breaker)     |    0.182  |    0.312 |     0.165 |  -0.205  |             0.132 |           13.2 |             0.672  |               0.245 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.12% a.a., t=0.07, p=0.945, significativo=False
- modelo - exposicao_igualada: -2.47% a.a., t=-1.31, p=0.189, significativo=False
