# Analise de baselines justos e ablacao

Fora da amostra (walk-forward purgado), periodo de desenvolvimento; holdout intocado. Gerado por `run_baseline_ablation.py`. `retorno`/`sharpe` tratam o caixa a 0%; `retorno_c/_caixa` e `sharpe_excesso_rf` usam a taxa livre de risco REAL (BCB CDI para .SA, FRED T-bill 3m para os demais; a fonte aparece em cada secao).

## SPY (2018-03-22 a 2023-09-22, 1386 dias)

Taxa livre de risco / caixa: **FRED DTB3**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |   dsr |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|------:|
| kelly_modelo (c/ stops)       |    0.1687 |    0.533 |     0.647 |  -0.0921 |             0.221 |          100   |             0.2667 |               0.486 | 0.527 |
| ablacao_sem_modelo (c/ stops) |    0.0629 |    0.344 |     0.409 |  -0.0717 |             0.177 |          100   |             0.1505 |               0.258 | 0.354 |
| exposicao_igualada (s/ stops) |    0.1606 |    0.614 |     0.753 |  -0.082  |             0.221 |          100   |             0.25   |               0.531 | 0.6   |
| H2_vol_target (s/ stops)      |    0.4823 |    0.739 |     0.969 |  -0.1289 |             0.687 |          100   |             0.5247 |               0.622 | 0.706 |
| H3_tendencia_vol (s/ stops)   |    0.3485 |    0.632 |     0.702 |  -0.1198 |             0.587 |          100   |             0.4131 |               0.538 | 0.615 |
| buy_and_hold_puro (s/ stops)  |    0.7896 |    0.614 |     0.753 |  -0.3372 |             1     |          100   |             0.7896 |               0.531 | 0.6   |
| buy_and_hold (c/ breaker)     |   -0.0279 |    0.011 |     0.007 |  -0.2671 |             0.358 |           35.8 |             0.0287 |              -0.052 | 0.124 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +1.83% a.a., t=1.40, p=0.163, significativo=False
- modelo - exposicao_igualada: +0.18% a.a., t=0.18, p=0.857, significativo=False

Diferenca de Sharpe contra o buy-and-hold puro (Memmel; historico, so descritivo -- H2/H3 sao decididas no paper trading). DSR com N=28 tentativas declaradas (limite inferior):

- H2 vol-target vs buy-and-hold: z=+0.55, p=0.580
- H3 tendencia+vol vs buy-and-hold: z=+0.05, p=0.958

## AAPL (2018-03-22 a 2023-09-22, 1386 dias)

Taxa livre de risco / caixa: **FRED DTB3**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |   dsr |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|------:|
| kelly_modelo (c/ stops)       |    0.2415 |    0.704 |     0.908 |  -0.0723 |             0.153 |          100   |             0.3529 |               0.675 | 0.463 |
| ablacao_sem_modelo (c/ stops) |    0.2198 |    0.793 |     1.075 |  -0.0572 |             0.143 |          100   |             0.3246 |               0.743 | 0.546 |
| exposicao_igualada (s/ stops) |    0.3004 |    0.986 |     1.367 |  -0.0685 |             0.153 |          100   |             0.4096 |               0.933 | 0.715 |
| H2_vol_target (s/ stops)      |    1.1985 |    1.334 |     2.051 |  -0.1238 |             0.4   |          100   |             1.3216 |               1.269 | 0.918 |
| H3_tendencia_vol (s/ stops)   |    0.6978 |    0.966 |     1.366 |  -0.1134 |             0.353 |          100   |             0.8097 |               0.912 | 0.7   |
| buy_and_hold_puro (s/ stops)  |    3.3531 |    0.986 |     1.367 |  -0.3852 |             1     |          100   |             3.3531 |               0.933 | 0.715 |
| buy_and_hold (c/ breaker)     |    0.0585 |    0.156 |     0.071 |  -0.2351 |             0.122 |           12.2 |             0.1489 |               0.131 | 0.085 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.38% a.a., t=0.34, p=0.735, significativo=False
- modelo - exposicao_igualada: -0.80% a.a., t=-0.65, p=0.514, significativo=False

Diferenca de Sharpe contra o buy-and-hold puro (Memmel; historico, so descritivo -- H2/H3 sao decididas no paper trading). DSR com N=28 tentativas declaradas (limite inferior):

- H2 vol-target vs buy-and-hold: z=+1.86, p=0.063
- H3 tendencia+vol vs buy-and-hold: z=-0.08, p=0.935

## PETR4.SA (2018-04-02 a 2023-10-30, 1386 dias)

Taxa livre de risco / caixa: **BCB CDI (SGS 12)**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |   dsr |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|------:|
| kelly_modelo (c/ stops)       |   -0.0455 |   -0.235 |    -0.041 |  -0.1255 |             0.008 |            2.8 |             0.4085 |              -0.25  | 0     |
| ablacao_sem_modelo (c/ stops) |   -0.0215 |   -0.315 |    -0.055 |  -0.05   |             0.003 |            2.8 |             0.4464 |              -0.33  | 0     |
| exposicao_igualada (s/ stops) |    0.0173 |    0.816 |     0.989 |  -0.0069 |             0.008 |          100   |             0.5005 |               0.666 | 0.189 |
| H2_vol_target (s/ stops)      |    0.7076 |    0.922 |     1.25  |  -0.1451 |             0.286 |          100   |             1.2586 |               0.741 | 0.257 |
| H3_tendencia_vol (s/ stops)   |    0.6919 |    1.009 |     1.323 |  -0.0933 |             0.246 |          100   |             1.2484 |               0.814 | 0.324 |
| buy_and_hold_puro (s/ stops)  |    3.4378 |    0.816 |     0.989 |  -0.6336 |             1     |          100   |             3.4378 |               0.666 | 0.189 |
| buy_and_hold (c/ breaker)     |   -0.0392 |   -0.029 |    -0.006 |  -0.2662 |             0.027 |            2.7 |             0.4087 |              -0.047 | 0.002 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: -0.40% a.a., t=-0.26, p=0.794, significativo=False
- modelo - exposicao_igualada: -1.10% a.a., t=-0.48, p=0.628, significativo=False

Diferenca de Sharpe contra o buy-and-hold puro (Memmel; historico, so descritivo -- H2/H3 sao decididas no paper trading). DSR com N=28 tentativas declaradas (limite inferior):

- H2 vol-target vs buy-and-hold: z=+0.54, p=0.588
- H3 tendencia+vol vs buy-and-hold: z=+0.60, p=0.551

## VALE3.SA (2018-04-02 a 2023-10-30, 1386 dias)

Taxa livre de risco / caixa: **BCB CDI (SGS 12)**

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_caixa |   sharpe_excesso_rf |   dsr |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------:|--------------------:|------:|
| kelly_modelo (c/ stops)       |    0.039  |    0.131 |     0.125 |  -0.1466 |             0.146 |          100   |             0.4583 |               0.002 | 0.283 |
| ablacao_sem_modelo (c/ stops) |    0.0442 |    0.235 |     0.296 |  -0.0753 |             0.091 |          100   |             0.4923 |               0.061 | 0.367 |
| exposicao_igualada (s/ stops) |    0.197  |    0.594 |     0.838 |  -0.0721 |             0.146 |          100   |             0.6729 |               0.414 | 0.689 |
| H2_vol_target (s/ stops)      |    0.3408 |    0.529 |     0.747 |  -0.1739 |             0.316 |          100   |             0.7495 |               0.326 | 0.633 |
| H3_tendencia_vol (s/ stops)   |    0.2752 |    0.512 |     0.589 |  -0.1594 |             0.225 |          100   |             0.7462 |               0.364 | 0.618 |
| buy_and_hold_puro (s/ stops)  |    1.3656 |    0.594 |     0.838 |  -0.4372 |             1     |          100   |             1.3656 |               0.414 | 0.689 |
| buy_and_hold (c/ breaker)     |    0.182  |    0.312 |     0.165 |  -0.205  |             0.132 |           13.2 |             0.672  |               0.245 | 0.436 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.13% a.a., t=0.07, p=0.942, significativo=False
- modelo - exposicao_igualada: -2.46% a.a., t=-1.31, p=0.191, significativo=False

Diferenca de Sharpe contra o buy-and-hold puro (Memmel; historico, so descritivo -- H2/H3 sao decididas no paper trading). DSR com N=28 tentativas declaradas (limite inferior):

- H2 vol-target vs buy-and-hold: z=-0.38, p=0.708
- H3 tendencia+vol vs buy-and-hold: z=-0.26, p=0.792
