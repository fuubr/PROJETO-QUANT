# Relatorio de drift (Etapa 7)
Gerado em 2026-10-06 02:49

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 21
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 2 dia(s) pos-correcao (dif. relativa max 1.46e-16, dif. de posicao max 5.55e-17)
- **Drift de features** (20 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.16 desvios, p empirico 0.644
  - momentum_20: deslocamento -0.40 desvios, p empirico 0.523
  - momentum_60: deslocamento +0.01 desvios, p empirico 0.987
  - volatility_20: deslocamento -0.56 desvios, p empirico 0.409
  - sma_distance_50: deslocamento -0.14 desvios, p empirico 0.848
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 1 dias com resultado maduro = 0.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0238, IC indisponivel
- **Challenger**: INSUFICIENTE (dados maduros apos o treino vigente ainda nao formam duas partes)

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 21
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 2 dia(s) pos-correcao (dif. relativa max 0.00e+00, dif. de posicao max 5.55e-17)
- **Drift de features** (20 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.21 desvios, p empirico 0.659
  - momentum_20: deslocamento +0.56 desvios, p empirico 0.504
  - momentum_60: deslocamento +0.44 desvios, p empirico 0.620
  - volatility_20: deslocamento -0.29 desvios, p empirico 0.713
  - sma_distance_50: deslocamento +0.21 desvios, p empirico 0.824
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 1 dias com resultado maduro = 0.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0048, IC indisponivel
- **Challenger**: INSUFICIENTE (dados maduros apos o treino vigente ainda nao formam duas partes)

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 21
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 2 dia(s) pos-correcao (dif. relativa max 0.00e+00, dif. de posicao max 8.33e-17)
- **Drift de features** (20 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.62 desvios, p empirico 0.173
  - momentum_20: deslocamento +1.07 desvios, p empirico 0.143
  - momentum_60: deslocamento +0.88 desvios, p empirico 0.304
  - volatility_20: deslocamento -0.39 desvios, p empirico 0.576
  - sma_distance_50: deslocamento +1.09 desvios, p empirico 0.184
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 1 dias com resultado maduro = 0.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0187, IC indisponivel
- **Challenger**: INSUFICIENTE (dados maduros apos o treino vigente ainda nao formam duas partes)

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 21
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 2 dia(s) pos-correcao (dif. relativa max 1.47e-16, dif. de posicao max 6.94e-17)
- **Drift de features** (20 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.55 desvios, p empirico 0.184
  - momentum_20: deslocamento -0.22 desvios, p empirico 0.778
  - momentum_60: deslocamento -0.48 desvios, p empirico 0.543
  - volatility_20: deslocamento -0.52 desvios, p empirico 0.458
  - sma_distance_50: deslocamento -0.21 desvios, p empirico 0.801
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 1 dias com resultado maduro = 0.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger**: INSUFICIENTE (dados maduros apos o treino vigente ainda nao formam duas partes)
