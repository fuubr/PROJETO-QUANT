"""Leitura dos DFP (dados abertos da CVM) com data de recebimento -- point-in-time.

Fonte: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_<ANO>.zip
Lucro e PL sao lidos pela DESCRICAO da conta de nivel 2 (o codigo muda entre setores:
3.11 na industria, 3.09 nos bancos, 2.03 vs 2.08 no PL). Valores sem escala: so a razao
lucro/PL e usada, e o LPA ja vem por acao.
"""

from __future__ import annotations

import io
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
CACHE = Path(__file__).resolve().parent / "cache" / "cvm"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()


def _statement(z: zipfile.ZipFile, kind: str) -> pd.DataFrame:
    """Consolidado, completado pelo individual para quem nao tem consolidado."""

    con = _read_kind(z, kind, "con")
    ind = _read_kind(z, kind, "ind")
    if con is None:
        return ind if ind is not None else pd.DataFrame()
    if ind is None:
        return con
    return pd.concat([con, ind[~ind.CNPJ_CIA.isin(con.CNPJ_CIA)]])


def _read_kind(z: zipfile.ZipFile, kind: str, scope: str) -> pd.DataFrame | None:
    name = next((n for n in z.namelist() if n.endswith(".csv") and f"_{kind}_{scope}_" in n), None)
    if name is None:
        return None
    df = pd.read_csv(z.open(name), sep=";", encoding="latin1", dtype=str)
    if df.empty:
        return None
    df = df[df.ORDEM_EXERC.map(_norm) == "ultimo"].copy()
    df["desc"] = df.DS_CONTA.map(_norm)
    df["val"] = pd.to_numeric(df.VL_CONTA, errors="coerce")
    df["level"] = df.CD_CONTA.str.count(r"\.")
    return df


def _net_income(dre: pd.DataFrame) -> pd.Series:
    lv2 = dre[(dre.level == 1) & dre.CD_CONTA.str.startswith("3.")]
    hit = lv2[lv2.desc.str.contains("lucro") & lv2.desc.str.contains("periodo|exercicio")]
    return hit.groupby("CNPJ_CIA").val.last()


def _equity(bpp: pd.DataFrame) -> pd.Series:
    lv2 = bpp[(bpp.level == 1) & bpp.CD_CONTA.str.startswith("2.")]
    return lv2[lv2.desc.str.startswith("patrimonio liquido")].groupby("CNPJ_CIA").val.last()


def _eps(dre: pd.DataFrame) -> pd.Series:
    rows = dre[dre.CD_CONTA.str.startswith("3.99.01.")]
    out = {}
    for cnpj, g in rows.groupby("CNPJ_CIA"):
        d: dict[str, float] = {}
        for desc, val in zip(g.DS_CONTA.str.strip().str.upper(), g.val, strict=False):
            if pd.notna(val) and (desc not in d or d[desc] == 0):
                d[desc] = float(val)
        out[cnpj] = d
    return pd.Series(out, dtype=object)


def parse_dfp_zip(source) -> pd.DataFrame:
    """Uma linha por empresa: cnpj, fy_end, received (PRIMEIRO recebimento),
    net_income, equity, eps (dict classe -> LPA basico)."""

    with zipfile.ZipFile(source) as z:
        main_name = next(n for n in z.namelist() if n.count("_") == 3 and n.endswith(".csv"))
        main = pd.read_csv(z.open(main_name), sep=";", encoding="latin1", dtype=str)
        received = main.groupby("CNPJ_CIA").DT_RECEB.min().map(pd.Timestamp)
        dre, bpp = _statement(z, "DRE"), _statement(z, "BPP")
    if dre.empty:
        return pd.DataFrame(columns=["cnpj", "fy_end", "received", "net_income", "equity", "eps"])
    fy_end = dre.groupby("CNPJ_CIA").DT_FIM_EXERC.max().map(pd.Timestamp)
    df = pd.DataFrame({"fy_end": fy_end, "received": received, "net_income": _net_income(dre),
                       "equity": _equity(bpp) if not bpp.empty else None, "eps": _eps(dre)})
    df = df[df.fy_end.notna() & df.received.notna()]
    return df.rename_axis("cnpj").reset_index()


def load_dfp_year(year: int) -> pd.DataFrame:
    path = CACHE / f"dfp_cia_aberta_{year}.zip"
    if not path.exists():
        import requests
        CACHE.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL.format(year=year), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return parse_dfp_zip(io.BytesIO(path.read_bytes()))
