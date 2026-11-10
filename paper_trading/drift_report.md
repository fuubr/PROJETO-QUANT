# Relatorio de drift (Etapa 7)
Gerado em 2026-11-10 04:32

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 46
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 27 dia(s) pos-correcao (dif. relativa max 1.17e-15, dif. de posicao max 5.55e-17)
- **Drift de features** (45 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.11 desvios, p empirico 0.603
  - momentum_20: deslocamento -0.26 desvios, p empirico 0.541
  - momentum_60: deslocamento -0.06 desvios, p empirico 0.891
  - volatility_20: deslocamento -0.76 desvios, p empirico 0.214
  - sma_distance_50: deslocamento -0.20 desvios, p empirico 0.713
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 26 dias com resultado maduro = 1.3 amostras efetivas (minimo 6); vantagem media de Brier +0.0154, IC indisponivel
- **Challenger** (treinado ate 2026-09-23, avaliado a partir de 2026-09-24): **INSUFICIENTE** -- 0.7 amostras efetivas, vantagem media +0.0060, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 46
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 27 dia(s) pos-correcao (dif. relativa max 2.89e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (45 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.04 desvios, p empirico 0.924
  - momentum_20: deslocamento +0.21 desvios, p empirico 0.745
  - momentum_60: deslocamento +0.13 desvios, p empirico 0.855
  - volatility_20: deslocamento -0.71 desvios, p empirico 0.263
  - sma_distance_50: deslocamento +0.16 desvios, p empirico 0.838
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 26 dias com resultado maduro = 1.3 amostras efetivas (minimo 6); vantagem media de Brier -0.0005, IC indisponivel
- **Challenger** (treinado ate 2026-09-23, avaliado a partir de 2026-09-24): **INSUFICIENTE** -- 0.7 amostras efetivas, vantagem media +0.0008, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 43
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 24 dia(s) pos-correcao (dif. relativa max 4.25e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (42 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.16 desvios, p empirico 0.539
  - momentum_20: deslocamento +0.78 desvios, p empirico 0.169
  - momentum_60: deslocamento +1.01 desvios, p empirico 0.202
  - volatility_20: deslocamento -0.07 desvios, p empirico 0.946
  - sma_distance_50: deslocamento +0.99 desvios, p empirico 0.172
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 23 dias com resultado maduro = 1.1 amostras efetivas (minimo 6); vantagem media de Brier -0.0086, IC indisponivel
- **Challenger** (treinado ate 2026-09-21, avaliado a partir de 2026-09-23): **INSUFICIENTE** -- 0.6 amostras efetivas, vantagem media +0.0456, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 43
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 24 dia(s) pos-correcao (dif. relativa max 5.93e-16, dif. de posicao max 9.71e-17)
- **Drift de features** (42 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.57 desvios, p empirico 0.076
  - momentum_20: deslocamento -0.78 desvios, p empirico 0.163
  - momentum_60: deslocamento -0.67 desvios, p empirico 0.396
  - volatility_20: deslocamento -0.54 desvios, p empirico 0.399
  - sma_distance_50: deslocamento -0.79 desvios, p empirico 0.278
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 23 dias com resultado maduro = 1.1 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-21, avaliado a partir de 2026-09-23): **INSUFICIENTE** -- 0.6 amostras efetivas, vantagem media -0.0194, IC indisponivel
