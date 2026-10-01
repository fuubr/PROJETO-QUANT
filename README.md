# Projeto QUANT -- pipeline de IA para trading com ceticismo de overfitting

Pipeline em Python (dados reais, backtest vetorizado, modelo calibrado, gestao de
risco, paper trading automatico). Metodologia: **cada resultado so vale se passar
por um controle que nao deveria passar** (ativo sintetico sem edge, ablacao,
baselines justos, holdout intocado).

## Como rodar
```
pip install -r requirements.txt
python -m pytest                              # suite completa
python run_stage4_risk_management.py          # Kelly + stops, 4 ativos reais
python run_baseline_ablation.py               # baselines justos + ablacao (CDI/T-bill reais) -> docs/analise_baselines.md
python run_stage6_paper_trading_daily.py      # 1 dia de paper trading (roda sozinho via GitHub Actions)
python run_stage6_report.py                   # resumo + dashboard.html
python run_stage7_drift_check.py              # relatorio de drift (semanal, automatico)
```
`run_stage3_holdout_evaluation.py` **gasta o holdout** (2024+): so uma vez, com modelo final.

## Estrutura
`data/` dados | `models/` features, rotulos, calibracao, walk-forward | `backtest/` motor e
estatistica (Newey-West, Bonferroni) | `risk/` Kelly, stop-loss, circuit breaker |
`paper_trading/` estado, precos canonicos, drift, retreino | `tests/` | `docs/`

## Automacao (GitHub Actions)
- `paper_trading_daily.yml`: 1 dia de paper trading + dashboard + resumo (dias uteis, 19h BRT)
- `drift_check_weekly.yml`: relatorio de drift (segundas) -- nunca troca modelo sozinho
- `tests.yml`: pytest em todo push

## Documentos
- `docs/CRITERIOS.md` -- criterios de sucesso e de parada, definidos ANTES dos dados
- `docs/analise_baselines.md` -- baselines justos e ablacao (resultados)
- `docs/HISTORICO.md` -- diario tecnico completo (decisoes, bugs achados, correcoes)
