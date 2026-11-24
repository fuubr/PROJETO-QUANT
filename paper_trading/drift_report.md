# Relatorio de drift (Etapa 7)
Gerado em 2026-11-24 06:52

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 56
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 37 dia(s) pos-correcao (dif. relativa max 1.17e-15, dif. de posicao max 5.55e-17)
- **Drift de features** (55 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.21 desvios, p empirico 0.321
  - momentum_20: deslocamento -0.27 desvios, p empirico 0.480
  - momentum_60: deslocamento -0.15 desvios, p empirico 0.797
  - volatility_20: deslocamento -0.64 desvios, p empirico 0.274
  - sma_distance_50: deslocamento -0.23 desvios, p empirico 0.652
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 36 dias com resultado maduro = 1.8 amostras efetivas (minimo 6); vantagem media de Brier +0.0045, IC indisponivel
- **Challenger** (treinado ate 2026-09-30, avaliado a partir de 2026-10-01): **INSUFICIENTE** -- 0.9 amostras efetivas, vantagem media +0.0130, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 56
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 37 dia(s) pos-correcao (dif. relativa max 5.75e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (55 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.05 desvios, p empirico 0.874
  - momentum_20: deslocamento +0.15 desvios, p empirico 0.789
  - momentum_60: deslocamento +0.16 desvios, p empirico 0.808
  - volatility_20: deslocamento -0.75 desvios, p empirico 0.213
  - sma_distance_50: deslocamento +0.13 desvios, p empirico 0.849
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 36 dias com resultado maduro = 1.8 amostras efetivas (minimo 6); vantagem media de Brier -0.0003, IC indisponivel
- **Challenger** (treinado ate 2026-09-30, avaliado a partir de 2026-10-01): **INSUFICIENTE** -- 0.9 amostras efetivas, vantagem media +0.0015, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 51
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 32 dia(s) pos-correcao (dif. relativa max 4.25e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (52 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.06 desvios, p empirico 0.814
  - momentum_20: deslocamento +0.26 desvios, p empirico 0.546
  - momentum_60: deslocamento +0.81 desvios, p empirico 0.279
  - volatility_20: deslocamento -0.00 desvios, p empirico 0.999
  - sma_distance_50: deslocamento +0.54 desvios, p empirico 0.363
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 33 dias com resultado maduro = 1.6 amostras efetivas (minimo 6); vantagem media de Brier +0.0106, IC indisponivel
- **Challenger** (treinado ate 2026-09-29, avaliado a partir de 2026-09-30): **INSUFICIENTE** -- 0.8 amostras efetivas, vantagem media +0.0065, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 51
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 32 dia(s) pos-correcao (dif. relativa max 5.97e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (52 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.55 desvios, p empirico 0.069
  - momentum_20: deslocamento -0.77 desvios, p empirico 0.138
  - momentum_60: deslocamento -0.74 desvios, p empirico 0.334
  - volatility_20: deslocamento -0.42 desvios, p empirico 0.538
  - sma_distance_50: deslocamento -0.87 desvios, p empirico 0.199
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 33 dias com resultado maduro = 1.6 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-29, avaliado a partir de 2026-09-30): **INSUFICIENTE** -- 0.8 amostras efetivas, vantagem media -0.0182, IC indisponivel
