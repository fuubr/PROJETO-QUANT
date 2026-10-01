"""Stage 7: drift report (and optional, explicit retraining).

Default run = read-only report -> paper_trading/drift_report.md. Meant to
run on a FIXED weekly cadence (see .github/workflows/drift_check_weekly.yml),
not daily: checking "has it drifted?" every day is itself a repeated test,
the multiple-comparisons trap already corrected several times in this
project.

  python run_stage7_drift_check.py                       # report only
  python run_stage7_drift_check.py --retrain-eval        # + challenger comparison
  python run_stage7_drift_check.py --promote SPY         # register new version (needs evidence)
  python run_stage7_drift_check.py --promote SPY --force-promote   # override, loud
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_START_DATE, DRIFT_BOOTSTRAP_RESAMPLES, DRIFT_BOOTSTRAP_SEED, DRIFT_CI_ALPHA,
    DRIFT_MIN_EFFECTIVE_SAMPLES, DRIFT_MIN_LIVE_DAYS_FEATURES, FEATURE_DRIFT_ALERT_P, PAPER_TRADING_BUGGY_THROUGH,
    PAPER_TRADING_TICKERS,
    PREDICTION_HORIZON_DAYS, STOP_LOSS_PCT, VAR_CONFIDENCE_LEVEL,
)
from paper_trading.prices import load_canonical
from models.labels import build_forward_return_label
from paper_trading.consistency_check import check_consistency
from paper_trading.drift import feature_drift, performance_drift
from paper_trading.model_store import list_versions
from paper_trading.predictions import inference_features, versioned_probabilities
from paper_trading.retrain import evaluate_challenger, promote
from paper_trading.state import load_log
from run_stage6_paper_trading_daily import _asset_costs, _circuit_breaker_thresholds

REPORT_PATH = Path(__file__).resolve().parent / "paper_trading" / "drift_report.md"


def analyze_ticker(ticker: str, close: pd.Series, log: pd.DataFrame, versions, do_retrain_eval: bool):
    latest = versions[-1]
    lines = [f"## {ticker}", f"- Modelo vigente: v{latest.version} (treinado ate {latest.trained_through.date()}, "
             f"{len(versions)} versao(oes) no total)", f"- Dias de paper trading: {len(log)}"]
    out = {"comparison": None}

    # 1) implementation divergence (a bug signal, not market drift)
    fee, slip = _asset_costs(ticker)
    cb, acb = _circuit_breaker_thresholds(ticker)
    cons = check_consistency(ticker, close, versions, log, fee_bps=fee, slippage_bps=slip, circuit_breaker_pct=cb,
                             asset_circuit_breaker_pct=acb, stop_loss_pct=STOP_LOSS_PCT,
                             var_confidence_level=VAR_CONFIDENCE_LEVEL,
                             clean_after=pd.Timestamp(PAPER_TRADING_BUGGY_THROUGH))
    if cons.days_compared == 0:
        status = f"aguardando dias gerados pelo codigo corrigido (dados ate {PAPER_TRADING_BUGGY_THROUGH} vieram do bug do dia atrasado e nao sao usados)"
    else:
        status = (f"{'OK' if cons.matches else '*** DIVERGE ***'} em {cons.days_compared} dia(s) pos-correcao "
                  f"(dif. relativa max {cons.max_relative_equity_diff:.2e}, dif. de posicao max {cons.max_target_diff:.2e})")
    lines.append(f"- **Consistencia backtest-vs-real** (detecta BUG, nao drift): {status}")

    # 2) feature drift
    features = inference_features(close)
    reference = features[features.index <= latest.trained_through]
    live = features[features.index > latest.trained_through]
    fd = feature_drift(reference, live, FEATURE_DRIFT_ALERT_P, DRIFT_MIN_LIVE_DAYS_FEATURES)
    if fd is None:
        lines.append(f"- **Drift de features**: INSUFICIENTE ({len(live)} dias ao vivo; minimo "
                     f"{DRIFT_MIN_LIVE_DAYS_FEATURES} e referencia >= 3x a janela)")
    else:
        flagged = [r.feature for r in fd if r.alert]
        lines.append(f"- **Drift de features** ({len(live)} dias ao vivo): "
                     f"{'ATENCAO em ' + ', '.join(flagged) if flagged else 'sem alerta'}")
        for r in fd:
            lines.append(f"  - {r.feature}: deslocamento {r.standardized_shift:+.2f} desvios, p empirico {r.empirical_p:.3f}"
                         f"{' <- ATENCAO' if r.alert else ''}")

    # 3) performance drift vs. the base-rate predictor, on matured outcomes only
    labels = build_forward_return_label(close, PREDICTION_HORIZON_DAYS).dropna()
    base_rate = float(labels[labels.index <= versions[0].trained_through].mean())
    probs = versioned_probabilities(close, versions)
    live_labels = labels[labels.index >= versions[0].effective_from]
    pr = performance_drift(probs, live_labels, base_rate, PREDICTION_HORIZON_DAYS, DRIFT_MIN_EFFECTIVE_SAMPLES,
                           DRIFT_BOOTSTRAP_RESAMPLES, DRIFT_CI_ALPHA, DRIFT_BOOTSTRAP_SEED)
    ci = f"IC95% [{pr.ci_low:+.4f}, {pr.ci_high:+.4f}]" if pr.ci_low == pr.ci_low else "IC indisponivel"
    lines.append(f"- **Desempenho vs. previsor ingenuo (taxa-base {base_rate:.1%})**: **{pr.verdict}** -- "
                 f"{pr.n_matured} dias com resultado maduro = {pr.n_effective:.1f} amostras efetivas "
                 f"(minimo {DRIFT_MIN_EFFECTIVE_SAMPLES}); vantagem media de Brier {pr.mean_skill:+.4f}, {ci}")

    if do_retrain_eval:
        cmp_ = evaluate_challenger(close, latest)
        out["comparison"] = cmp_
        if cmp_.train_end is None:
            lines.append("- **Challenger**: INSUFICIENTE (dados maduros apos o treino vigente ainda nao formam duas partes)")
        else:
            cci = f"IC95% [{cmp_.ci_low:+.4f}, {cmp_.ci_high:+.4f}]" if cmp_.ci_low == cmp_.ci_low else "IC indisponivel"
            lines.append(f"- **Challenger** (treinado ate {cmp_.train_end.date()}, avaliado a partir de "
                         f"{cmp_.eval_start.date()}): **{cmp_.verdict}** -- {cmp_.n_effective:.1f} amostras efetivas, "
                         f"vantagem media {cmp_.mean_advantage:+.4f}, {cci}")
    return lines, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-eval", action="store_true")
    parser.add_argument("--promote", metavar="TICKER")
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()

    now = pd.Timestamp.now()
    report = [
        "# Relatorio de drift (Etapa 7)", f"Gerado em {now:%Y-%m-%d %H:%M}", "",
        "> **Como ler**: INSUFICIENTE nao e um defeito, e o resultado correto enquanto nao ha amostra "
        "independente suficiente. Limiares fixados em `config.py` antes de qualquer resultado. "
        "Nenhum modelo e trocado automaticamente.", "",
    ]
    for ticker in PAPER_TRADING_TICKERS:
        log, versions = load_log(ticker), list_versions(ticker)
        if log is None or not versions:
            report += [f"## {ticker}", "- sem historico de paper trading ainda", ""]
            continue
        close = load_canonical(ticker)
        if close is None:
            report += [f"## {ticker}", "- serie canonica de precos ainda nao existe (rode o paper trading diario)", ""]
            continue
        lines, out = analyze_ticker(ticker, close, log, versions, args.retrain_eval or args.promote == ticker)
        report += lines + [""]

        if args.promote == ticker:
            new = promote(ticker, close, versions[-1], out["comparison"], force=args.force_promote)
            msg = f"- **PROMOVIDO** para v{new.version}, vigente a partir de {new.effective_from.date()}" + \
                  (" (**FORCADO, sem evidencia estatistica**)" if out["comparison"].verdict != "CHALLENGER_MELHOR" else "")
            print(msg)
            report += [msg, ""]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nRelatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
