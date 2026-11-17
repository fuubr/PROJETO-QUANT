# Relatorio de drift (Etapa 7)
Gerado em 2026-11-17 05:00

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 51
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 32 dia(s) pos-correcao (dif. relativa max 1.17e-15, dif. de posicao max 5.55e-17)
- **Drift de features** (50 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.02 desvios, p empirico 0.902
  - momentum_20: deslocamento -0.22 desvios, p empirico 0.593
  - momentum_60: deslocamento -0.09 desvios, p empirico 0.859
  - volatility_20: deslocamento -0.73 desvios, p empirico 0.234
  - sma_distance_50: deslocamento -0.15 desvios, p empirico 0.743
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 31 dias com resultado maduro = 1.6 amostras efetivas (minimo 6); vantagem media de Brier +0.0198, IC indisponivel
- **Challenger** (treinado ate 2026-09-28, avaliado a partir de 2026-09-29): **INSUFICIENTE** -- 0.8 amostras efetivas, vantagem media -0.0005, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 51
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 32 dia(s) pos-correcao (dif. relativa max 2.89e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (50 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.10 desvios, p empirico 0.759
  - momentum_20: deslocamento +0.18 desvios, p empirico 0.760
  - momentum_60: deslocamento +0.16 desvios, p empirico 0.818
  - volatility_20: deslocamento -0.74 desvios, p empirico 0.233
  - sma_distance_50: deslocamento +0.15 desvios, p empirico 0.826
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 31 dias com resultado maduro = 1.6 amostras efetivas (minimo 6); vantagem media de Brier -0.0007, IC indisponivel
- **Challenger** (treinado ate 2026-09-28, avaliado a partir de 2026-09-29): **INSUFICIENTE** -- 0.8 amostras efetivas, vantagem media +0.0015, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 47
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 28 dia(s) pos-correcao (dif. relativa max 4.25e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (47 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.01 desvios, p empirico 0.965
  - momentum_20: deslocamento +0.51 desvios, p empirico 0.302
  - momentum_60: deslocamento +0.91 desvios, p empirico 0.229
  - volatility_20: deslocamento -0.02 desvios, p empirico 0.987
  - sma_distance_50: deslocamento +0.76 desvios, p empirico 0.247
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 28 dias com resultado maduro = 1.4 amostras efetivas (minimo 6); vantagem media de Brier +0.0027, IC indisponivel
- **Challenger** (treinado ate 2026-09-24, avaliado a partir de 2026-09-25): **INSUFICIENTE** -- 0.7 amostras efetivas, vantagem media +0.0150, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 47
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 28 dia(s) pos-correcao (dif. relativa max 5.93e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (47 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.45 desvios, p empirico 0.119
  - momentum_20: deslocamento -0.76 desvios, p empirico 0.163
  - momentum_60: deslocamento -0.69 desvios, p empirico 0.373
  - volatility_20: deslocamento -0.48 desvios, p empirico 0.471
  - sma_distance_50: deslocamento -0.80 desvios, p empirico 0.247
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 28 dias com resultado maduro = 1.4 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-24, avaliado a partir de 2026-09-25): **INSUFICIENTE** -- 0.7 amostras efetivas, vantagem media -0.0182, IC indisponivel
