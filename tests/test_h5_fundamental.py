import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from data.cvm import parse_dfp_zip
from strategies.fundamental import earnings_yield, h5_scores, point_in_time

HDR = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;GRUPO_DFP;MOEDA;ESCALA_MOEDA;ORDEM_EXERC;DT_INI_EXERC;DT_FIM_EXERC;CD_CONTA;DS_CONTA;VL_CONTA;ST_CONTA_FIXA"


def _row(cnpj, code, desc, val, ordem="ÚLTIMO", ver="1"):
    return f"{cnpj};2018-12-31;{ver};X;1;G;REAL;MIL;{ordem};2018-01-01;2018-12-31;{code};{desc};{val};S"


def _zip(dre_con, bpp_con, dre_ind=(), bpp_ind=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        main = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
        main += "A;2018-12-31;1;X;1;DFP;1;2019-02-20;l\nA;2018-12-31;2;X;1;DFP;2;2019-04-01;l\n"
        main += "B;2018-12-31;1;X;2;DFP;3;2019-03-10;l\n"
        z.writestr("dfp_cia_aberta_2018.csv", main.encode("latin1"))
        for name, rows in (("DRE_con", dre_con), ("BPP_con", bpp_con), ("DRE_ind", dre_ind), ("BPP_ind", bpp_ind)):
            z.writestr(f"dfp_cia_aberta_{name}_2018.csv", "\n".join([HDR, *rows]).encode("latin1"))
    buf.seek(0)
    return buf


def test_parse_dfp_picks_net_income_equity_eps_and_first_receipt():
    z = _zip(
        dre_con=[_row("A", "3.09", "Lucro/Prejuízo Consolidado do Período", "100"),
                 _row("A", "3.09", "Lucro/Prejuízo Consolidado do Período", "70", ordem="PENÚLTIMO"),
                 _row("A", "3.99.01.01", "ON", "2.5"), _row("A", "3.99.01.02", "PN", "2.7")],
        bpp_con=[_row("A", "2.08", "Patrimônio Líquido Consolidado", "400")],
    )
    df = parse_dfp_zip(z).set_index("cnpj")
    a = df.loc["A"]
    assert a.net_income == 100 and a.equity == 400
    assert a.eps["ON"] == 2.5 and a.eps["PN"] == 2.7
    assert a.fy_end == pd.Timestamp("2018-12-31")
    # versao reapresentada nao atrasa nem adianta: vale o PRIMEIRO recebimento
    assert a.received == pd.Timestamp("2019-02-20")


def test_parse_dfp_falls_back_to_individual():
    z = _zip(dre_con=[], bpp_con=[],
             dre_ind=[_row("B", "3.11", "Lucro/Prejuízo do Período", "-5"), _row("B", "3.99.01.01", "ON", "-0.1")],
             bpp_ind=[_row("B", "2.03", "Patrimônio Líquido", "50")])
    b = parse_dfp_zip(z).set_index("cnpj").loc["B"]
    assert b.net_income == -5 and b.equity == 50


def _fund(rows):
    return pd.DataFrame(rows, columns=["ticker", "available", "fy_end", "eps", "net_income", "equity"])


def test_point_in_time_uses_only_published_data():
    f = _fund([["X", pd.Timestamp("2019-02-21"), pd.Timestamp("2018-12-31"), 1.0, 10, 100],
               ["X", pd.Timestamp("2020-02-21"), pd.Timestamp("2019-12-31"), 2.0, 20, 100]])
    assert point_in_time(f, pd.Timestamp("2019-02-20")).empty
    assert point_in_time(f, pd.Timestamp("2019-06-28")).loc["X", "eps"] == 1.0
    assert point_in_time(f, pd.Timestamp("2020-03-31")).loc["X", "eps"] == 2.0


def test_earnings_yield_adjusts_eps_for_later_splits():
    # LPA 2,0 no exercicio; desdobramento 2:1 depois do exercicio -> LPA na base atual 1,0
    splits = pd.Series([2.0], index=[pd.Timestamp("2019-05-02")])
    ep = earnings_yield(eps=2.0, fy_end=pd.Timestamp("2018-12-31"), price=10.0, splits=splits)
    assert ep == pytest.approx(0.10)
    # desdobramento ANTES do fim do exercicio ja esta no LPA
    ep2 = earnings_yield(2.0, pd.Timestamp("2018-12-31"), 10.0, pd.Series([2.0], index=[pd.Timestamp("2018-06-01")]))
    assert ep2 == pytest.approx(0.20)


def test_scores_rank_value_and_quality_and_drop_bad_data():
    pit = pd.DataFrame({"ep": [0.15, 0.05, 0.10, 3.0, 0.1], "net_income": [10, 10, 10, 10, 10],
                        "equity": [50, 200, 100, 100, -5], "eps": [1, 1, 1, 1, 1]},
                       index=["BARATA_BOA", "CARA_RUIM", "MEIO", "ERRO_DADO", "PL_NEG"])
    s = h5_scores(pit)
    assert set(s.index) == {"BARATA_BOA", "CARA_RUIM", "MEIO"}
    assert s.idxmax() == "BARATA_BOA" and s.idxmin() == "CARA_RUIM"


def test_zero_eps_with_profit_is_data_error():
    pit = pd.DataFrame({"ep": [0.0, 0.1], "net_income": [10, 10], "equity": [100, 100], "eps": [0.0, 1.0]},
                       index=["ZERADO", "OK"])
    assert list(h5_scores(pit).index) == ["OK"]


def test_scores_empty_input():
    empty = pd.DataFrame(columns=["ep", "net_income", "equity", "eps"], dtype=float)
    assert h5_scores(empty).empty
    assert np.isfinite(len(h5_scores(empty)))
