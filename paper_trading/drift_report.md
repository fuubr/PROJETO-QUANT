# Relatorio de drift (Etapa 7)
Gerado em 2026-10-13 03:26

> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. Nenhum modelo e trocado automaticamente.

## SPY
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 26
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 7 dia(s) pos-correcao (dif. relativa max 4.37e-16, dif. de posicao max 5.55e-17)
- **Drift de features** (25 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.08 desvios, p empirico 0.797
  - momentum_20: deslocamento -0.34 desvios, p empirico 0.574
  - momentum_60: deslocamento -0.01 desvios, p empirico 0.989
  - volatility_20: deslocamento -0.55 desvios, p empirico 0.407
  - sma_distance_50: deslocamento -0.13 desvios, p empirico 0.850
- **Desempenho vs. previsor ingenuo (taxa-base 68.6%)**: **INSUFICIENTE** -- 6 dias com resultado maduro = 0.3 amostras efetivas (minimo 6); vantagem media de Brier +0.0430, IC indisponivel
- **Challenger** (treinado ate 2026-09-02, avaliado a partir de 2026-09-03): **INSUFICIENTE** -- 0.1 amostras efetivas, vantagem media -0.0161, IC indisponivel

## AAPL
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 26
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 7 dia(s) pos-correcao (dif. relativa max 1.44e-16, dif. de posicao max 5.55e-17)
- **Drift de features** (25 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.16 desvios, p empirico 0.723
  - momentum_20: deslocamento +0.47 desvios, p empirico 0.555
  - momentum_60: deslocamento +0.34 desvios, p empirico 0.677
  - volatility_20: deslocamento -0.31 desvios, p empirico 0.693
  - sma_distance_50: deslocamento +0.20 desvios, p empirico 0.829
- **Desempenho vs. previsor ingenuo (taxa-base 63.1%)**: **INSUFICIENTE** -- 6 dias com resultado maduro = 0.3 amostras efetivas (minimo 6); vantagem media de Brier +0.0031, IC indisponivel
- **Challenger** (treinado ate 2026-09-02, avaliado a partir de 2026-09-03): **INSUFICIENTE** -- 0.1 amostras efetivas, vantagem media -0.0068, IC indisponivel

## PETR4.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 25
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 6 dia(s) pos-correcao (dif. relativa max 1.42e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (24 dias ao vivo): sem alerta
  - momentum_5: deslocamento +0.77 desvios, p empirico 0.081
  - momentum_20: deslocamento +1.05 desvios, p empirico 0.135
  - momentum_60: deslocamento +0.98 desvios, p empirico 0.259
  - volatility_20: deslocamento -0.33 desvios, p empirico 0.626
  - sma_distance_50: deslocamento +1.17 desvios, p empirico 0.157
- **Desempenho vs. previsor ingenuo (taxa-base 59.7%)**: **INSUFICIENTE** -- 5 dias com resultado maduro = 0.2 amostras efetivas (minimo 6); vantagem media de Brier +0.0187, IC indisponivel
- **Challenger** (treinado ate 2026-09-02, avaliado a partir de 2026-09-03): **INSUFICIENTE** -- 0.1 amostras efetivas, vantagem media +0.0112, IC indisponivel

## VALE3.SA
- Modelo vigente: v1 (treinado ate 2026-08-31, 1 versao(oes) no total)
- Dias de paper trading: 25
- **Consistencia backtest-vs-real** (detecta BUG, nao drift): OK em 6 dia(s) pos-correcao (dif. relativa max 1.47e-16, dif. de posicao max 8.33e-17)
- **Drift de features** (24 dias ao vivo): sem alerta
  - momentum_5: deslocamento -0.55 desvios, p empirico 0.166
  - momentum_20: deslocamento -0.41 desvios, p empirico 0.561
  - momentum_60: deslocamento -0.49 desvios, p empirico 0.538
  - volatility_20: deslocamento -0.54 desvios, p empirico 0.434
  - sma_distance_50: deslocamento -0.33 desvios, p empirico 0.688
- **Desempenho vs. previsor ingenuo (taxa-base 56.5%)**: **INSUFICIENTE** -- 5 dias com resultado maduro = 0.2 amostras efetivas (minimo 6); vantagem media de Brier +0.0216, IC indisponivel
- **Challenger** (treinado ate 2026-09-02, avaliado a partir de 2026-09-03): **INSUFICIENTE** -- 0.1 amostras efetivas, vantagem media -0.0039, IC indisponivel
