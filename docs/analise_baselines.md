# Analise de baselines justos e ablacao

Fora da amostra (walk-forward purgado), periodo de desenvolvimento; holdout intocado. Gerado por `run_baseline_ablation.py`. Caixa a 0% nas colunas principais; as colunas `c/ juros caixa` usam taxas ASSUMIDAS (config), nao dados de mercado.

## SPY (2018-03-22 a 2023-09-22, 1386 dias)

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_juros_caixa |   sharpe_c/_juros_caixa |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------------:|------------------------:|
| kelly_modelo (c/ stops)       |    0.1806 |    0.565 |     0.689 |  -0.0889 |             0.221 |          100   |                   0.3123 |                   0.908 |
| ablacao_sem_modelo (c/ stops) |    0.0629 |    0.344 |     0.409 |  -0.0717 |             0.177 |          100   |                   0.1886 |                   0.942 |
| exposicao_igualada (s/ stops) |    0.1609 |    0.614 |     0.753 |  -0.0822 |             0.221 |          100   |                   0.2903 |                   1.034 |
| vol_target (s/ stops)         |    0.4823 |    0.739 |     0.969 |  -0.1289 |             0.687 |          100   |                   0.5466 |                   0.814 |
| buy_and_hold_puro (s/ stops)  |    0.7896 |    0.614 |     0.753 |  -0.3372 |             1     |          100   |                   0.7896 |                   0.614 |
| buy_and_hold (c/ breaker)     |   -0.0279 |    0.011 |     0.007 |  -0.2671 |             0.358 |           35.8 |                   0.0606 |                   0.152 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +2.01% a.a., t=1.53, p=0.127, significativo=False
- modelo - exposicao_igualada: +0.36% a.a., t=0.34, p=0.733, significativo=False

## AAPL (2018-03-22 a 2023-09-22, 1386 dias)

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_juros_caixa |   sharpe_c/_juros_caixa |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------------:|------------------------:|
| kelly_modelo (c/ stops)       |    0.2211 |    0.646 |     0.822 |  -0.0732 |             0.153 |          100   |                   0.3699 |                   1.001 |
| ablacao_sem_modelo (c/ stops) |    0.2198 |    0.793 |     1.075 |  -0.0572 |             0.143 |          100   |                   0.3702 |                   1.243 |
| exposicao_igualada (s/ stops) |    0.2998 |    0.986 |     1.367 |  -0.0684 |             0.153 |          100   |                   0.4582 |                   1.408 |
| vol_target (s/ stops)         |    1.1985 |    1.334 |     2.051 |  -0.1238 |             0.4   |          100   |                   1.385  |                   1.467 |
| buy_and_hold_puro (s/ stops)  |    3.3531 |    0.986 |     1.367 |  -0.3852 |             1     |          100   |                   3.3531 |                   0.987 |
| buy_and_hold (c/ breaker)     |    0.0585 |    0.156 |     0.071 |  -0.2351 |             0.122 |           12.2 |                   0.1924 |                   0.381 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.08% a.a., t=0.09, p=0.932, significativo=False
- modelo - exposicao_igualada: -1.09% a.a., t=-0.88, p=0.380, significativo=False

## PETR4.SA (2018-04-02 a 2023-10-30, 1386 dias)

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_juros_caixa |   sharpe_c/_juros_caixa |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------------:|------------------------:|
| kelly_modelo (c/ stops)       |   -0.0455 |   -0.235 |    -0.041 |  -0.1255 |             0.008 |            2.8 |                   0.6049 |                   2.576 |
| ablacao_sem_modelo (c/ stops) |   -0.0215 |   -0.315 |    -0.055 |  -0.05   |             0.003 |            2.8 |                   0.6496 |                   7.396 |
| exposicao_igualada (s/ stops) |    0.0173 |    0.816 |     0.989 |  -0.0069 |             0.008 |          100   |                   0.7104 |                  25.52  |
| vol_target (s/ stops)         |    0.7076 |    0.922 |     1.25  |  -0.1451 |             0.286 |          100   |                   1.4823 |                   1.528 |
| buy_and_hold_puro (s/ stops)  |    3.4378 |    0.816 |     0.989 |  -0.6336 |             1     |          100   |                   3.4378 |                   0.817 |
| buy_and_hold (c/ breaker)     |   -0.0392 |   -0.029 |    -0.006 |  -0.2662 |             0.027 |            2.7 |                   0.5997 |                   0.961 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: -0.40% a.a., t=-0.26, p=0.794, significativo=False
- modelo - exposicao_igualada: -1.10% a.a., t=-0.48, p=0.628, significativo=False

## VALE3.SA (2018-04-02 a 2023-10-30, 1386 dias)

|                               |   retorno |   sharpe |   sortino |   max_dd |   exposicao_media |   dias_ativo_% |   retorno_c/_juros_caixa |   sharpe_c/_juros_caixa |
|:------------------------------|----------:|---------:|----------:|---------:|------------------:|---------------:|-------------------------:|------------------------:|
| kelly_modelo (c/ stops)       |    0.0337 |    0.119 |     0.112 |  -0.1488 |             0.146 |          100   |                   0.617  |                   1.213 |
| ablacao_sem_modelo (c/ stops) |    0.0442 |    0.235 |     0.296 |  -0.0753 |             0.091 |          100   |                   0.6809 |                   2.63  |
| exposicao_igualada (s/ stops) |    0.1973 |    0.594 |     0.838 |  -0.0722 |             0.146 |          100   |                   0.8729 |                   1.998 |
| vol_target (s/ stops)         |    0.3408 |    0.529 |     0.747 |  -0.1739 |             0.316 |          100   |                   0.9182 |                   1.106 |
| buy_and_hold_puro (s/ stops)  |    1.3656 |    0.594 |     0.838 |  -0.4372 |             1     |          100   |                   1.3656 |                   0.594 |
| buy_and_hold (c/ breaker)     |    0.182  |    0.312 |     0.165 |  -0.205  |             0.132 |           13.2 |                   0.8623 |                   0.997 |

Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p

- modelo - ablacao: +0.03% a.a., t=0.02, p=0.981, significativo=False
- modelo - exposicao_igualada: -2.56% a.a., t=-1.36, p=0.174, significativo=False
