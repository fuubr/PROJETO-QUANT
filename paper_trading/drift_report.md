# Relatorio de drift (Etapa 7)
Gerado em 2026-10-20 01:39

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 31
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 12 dia(s) pos-correcao (dif. relativa max 7.28e-16, dif. de posicao max 5.55e-17)
- **Drift de features** (30 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.11 desvios, p empirico 0.724
  - momentum_20: deslocamento -0.29 desvios, p empirico 0.619
  - momentum_60: deslocamento -0.02 desvios, p empirico 0.976
  - volatility_20: deslocamento -0.57 desvios, p empirico 0.357
  - sma_distance_50: deslocamento -0.15 desvios, p empirico 0.816
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 11 dias com resultado maduro = 0.6 amostras efetivas (minimo 6); vantagem media de Brier +0.0553, IC indisponivel
- **Challenger** (treinado ate 2026-09-08, avaliado a partir de 2026-09-09): **INSUFICIENTE** -- 0.2 amostras efetivas, vantagem media -0.0206, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 31
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 12 dia(s) pos-correcao (dif. relativa max 2.88e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (30 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.18 desvios, p empirico 0.657
  - momentum_20: deslocamento +0.41 desvios, p empirico 0.596
  - momentum_60: deslocamento +0.25 desvios, p empirico 0.744
  - volatility_20: deslocamento -0.40 desvios, p empirico 0.595
  - sma_distance_50: deslocamento +0.23 desvios, p empirico 0.799
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 11 dias com resultado maduro = 0.6 amostras efetivas (minimo 6); vantagem media de Brier -0.0007, IC indisponivel
- **Challenger** (treinado ate 2026-09-08, avaliado a partir de 2026-09-09): **INSUFICIENTE** -- 0.2 amostras efetivas, vantagem media +0.0065, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 30
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 11 dia(s) pos-correcao (dif. relativa max 2.81e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (29 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.77 desvios, p empirico 0.076
  - momentum_20: deslocamento +1.07 desvios, p empirico 0.115
  - momentum_60: deslocamento +1.09 desvios, p empirico 0.195
  - volatility_20: deslocamento -0.27 desvios, p empirico 0.668
  - sma_distance_50: deslocamento +1.27 desvios, p empirico 0.124
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 10 dias com resultado maduro = 0.5 amostras efetivas (minimo 6); vantagem media de Brier -0.0110, IC indisponivel
- **Challenger** (treinado ate 2026-09-04, avaliado a partir de 2026-09-08): **INSUFICIENTE** -- 0.2 amostras efetivas, vantagem media -0.0144, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 30
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 11 dia(s) pos-correcao (dif. relativa max 2.95e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (29 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.59 desvios, p empirico 0.104
  - momentum_20: deslocamento -0.58 desvios, p empirico 0.373
  - momentum_60: deslocamento -0.52 desvios, p empirico 0.518
  - volatility_20: deslocamento -0.58 desvios, p empirico 0.377
  - sma_distance_50: deslocamento -0.47 desvios, p empirico 0.539
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 10 dias com resultado maduro = 0.5 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-04, avaliado a partir de 2026-09-08): **INSUFICIENTE** -- 0.2 amostras efetivas, vantagem media -0.0067, IC indisponivel
