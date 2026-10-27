# Relatorio de drift (Etapa 7)
Gerado em 2026-10-27 01:56

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 36
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 17 dia(s) pos-correcao (dif. relativa max 8.73e-16, dif. de posicao max 5.55e-17)
- **Drift de features** (35 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.11 desvios, p empirico 0.691
  - momentum_20: deslocamento -0.27 desvios, p empirico 0.598
  - momentum_60: deslocamento -0.01 desvios, p empirico 0.984
  - volatility_20: deslocamento -0.64 desvios, p empirico 0.293
  - sma_distance_50: deslocamento -0.17 desvios, p empirico 0.783
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 16 dias com resultado maduro = 0.8 amostras efetivas (minimo 6); vantagem media de Brier +0.0404, IC indisponivel
- **Challenger** (treinado ate 2026-09-10, avaliado a partir de 2026-09-14): **INSUFICIENTE** -- 0.4 amostras efetivas, vantagem media -0.0025, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 36
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 17 dia(s) pos-correcao (dif. relativa max 2.88e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (35 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.13 desvios, p empirico 0.715
  - momentum_20: deslocamento +0.31 desvios, p empirico 0.666
  - momentum_60: deslocamento +0.18 desvios, p empirico 0.821
  - volatility_20: deslocamento -0.50 desvios, p empirico 0.480
  - sma_distance_50: deslocamento +0.23 desvios, p empirico 0.797
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 16 dias com resultado maduro = 0.8 amostras efetivas (minimo 6); vantagem media de Brier -0.0004, IC indisponivel
- **Challenger** (treinado ate 2026-09-10, avaliado a partir de 2026-09-14): **INSUFICIENTE** -- 0.4 amostras efetivas, vantagem media +0.0024, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 35
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 16 dia(s) pos-correcao (dif. relativa max 4.25e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (34 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.61 desvios, p empirico 0.112
  - momentum_20: deslocamento +1.06 desvios, p empirico 0.102
  - momentum_60: deslocamento +1.12 desvios, p empirico 0.173
  - volatility_20: deslocamento -0.21 desvios, p empirico 0.777
  - sma_distance_50: deslocamento +1.28 desvios, p empirico 0.109
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 15 dias com resultado maduro = 0.8 amostras efetivas (minimo 6); vantagem media de Brier -0.0169, IC indisponivel
- **Challenger** (treinado ate 2026-09-10, avaliado a partir de 2026-09-14): **INSUFICIENTE** -- 0.3 amostras efetivas, vantagem media +0.0246, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 35
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 16 dia(s) pos-correcao (dif. relativa max 2.96e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (34 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.63 desvios, p empirico 0.070
  - momentum_20: deslocamento -0.67 desvios, p empirico 0.276
  - momentum_60: deslocamento -0.57 desvios, p empirico 0.474
  - volatility_20: deslocamento -0.61 desvios, p empirico 0.354
  - sma_distance_50: deslocamento -0.62 desvios, p empirico 0.440
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 15 dias com resultado maduro = 0.8 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-10, avaliado a partir de 2026-09-14): **INSUFICIENTE** -- 0.3 amostras efetivas, vantagem media -0.0126, IC indisponivel
