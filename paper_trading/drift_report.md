# Relatorio de drift (Etapa 7)
Gerado em 2026-11-03 03:53

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 41
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 22 dia(s) pos-correcao (dif. relativa max 1.02e-15, dif. de posicao max 5.55e-17)
- **Drift de features** (40 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.12 desvios, p empirico 0.631
  - momentum_20: deslocamento -0.26 desvios, p empirico 0.591
  - momentum_60: deslocamento -0.01 desvios, p empirico 0.991
  - volatility_20: deslocamento -0.71 desvios, p empirico 0.255
  - sma_distance_50: deslocamento -0.19 desvios, p empirico 0.749
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 21 dias com resultado maduro = 1.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0343, IC indisponivel
- **Challenger** (treinado ate 2026-09-17, avaliado a partir de 2026-09-21): **INSUFICIENTE** -- 0.5 amostras efetivas, vantagem media -0.0032, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 41
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 22 dia(s) pos-correcao (dif. relativa max 2.89e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (40 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.09 desvios, p empirico 0.813
  - momentum_20: deslocamento +0.27 desvios, p empirico 0.704
  - momentum_60: deslocamento +0.13 desvios, p empirico 0.855
  - volatility_20: deslocamento -0.62 desvios, p empirico 0.342
  - sma_distance_50: deslocamento +0.21 desvios, p empirico 0.806
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 21 dias com resultado maduro = 1.1 amostras efetivas (minimo 6); vantagem media de Brier -0.0009, IC indisponivel
- **Challenger** (treinado ate 2026-09-17, avaliado a partir de 2026-09-21): **INSUFICIENTE** -- 0.5 amostras efetivas, vantagem media +0.0014, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 39
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 20 dia(s) pos-correcao (dif. relativa max 4.25e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (38 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.37 desvios, p empirico 0.255
  - momentum_20: deslocamento +0.98 desvios, p empirico 0.120
  - momentum_60: deslocamento +1.10 desvios, p empirico 0.178
  - volatility_20: deslocamento -0.13 desvios, p empirico 0.881
  - sma_distance_50: deslocamento +1.17 desvios, p empirico 0.124
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 19 dias com resultado maduro = 0.9 amostras efetivas (minimo 6); vantagem media de Brier -0.0219, IC indisponivel
- **Challenger** (treinado ate 2026-09-16, avaliado a partir de 2026-09-17): **INSUFICIENTE** -- 0.5 amostras efetivas, vantagem media +0.0368, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 39
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 20 dia(s) pos-correcao (dif. relativa max 2.98e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (38 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.62 desvios, p empirico 0.066
  - momentum_20: deslocamento -0.74 desvios, p empirico 0.208
  - momentum_60: deslocamento -0.63 desvios, p empirico 0.432
  - volatility_20: deslocamento -0.58 desvios, p empirico 0.373
  - sma_distance_50: deslocamento -0.72 desvios, p empirico 0.353
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 19 dias com resultado maduro = 0.9 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-16, avaliado a partir de 2026-09-17): **INSUFICIENTE** -- 0.5 amostras efetivas, vantagem media -0.0154, IC indisponivel
