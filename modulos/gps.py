"""
Módulo 1 — Leitura e tratamento dos dados de GPS
================================================

Implementa o **Passo 1** do plano: converter as coordenadas esféricas do GPX
para um plano cartesiano local (ENU — East / North / Up), no qual as operações
de geometria projetiva passam a ser lineares.

Cadeia de processamento:

    GPX -> DataFrame -> ENU -> suavização -> heading -> velocidade

O heading gravado pelo aplicativo é ignorado de propósito: ele costuma vir
rotacionado/ruidoso. Aqui ele é sempre **recalculado** a partir da evolução
temporal das posições já suavizadas.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1.1 Leitura bruta do GPX
# ----------------------------------------------------------------------


def carregar_gpx(caminho: str | Path) -> pd.DataFrame:
    """Lê um arquivo GPX e devolve um DataFrame ordenado por tempo.

    Colunas: ``lat``, ``lon``, ``ele``, ``tempo`` (UTC), ``t`` (s desde o
    primeiro ponto).
    """
    import gpxpy

    caminho = Path(caminho)
    with open(caminho, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    linhas = []
    for trilha in gpx.tracks:
        for segmento in trilha.segments:
            for p in segmento.points:
                t = p.time
                if t is not None and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                linhas.append(
                    {
                        "lat": p.latitude,
                        "lon": p.longitude,
                        "ele": p.elevation if p.elevation is not None else 0.0,
                        "tempo": t,
                    }
                )

    if not linhas:
        raise ValueError(f"Nenhum ponto de trajeto encontrado em {caminho}")

    df = pd.DataFrame(linhas).dropna(subset=["tempo"]).sort_values("tempo")
    df = df.reset_index(drop=True)
    df["t"] = (df["tempo"] - df["tempo"].iloc[0]).dt.total_seconds()

    # pontos com timestamp repetido quebram qualquer interpolação
    df = df.drop_duplicates(subset="t").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 1.2 Passo 1 do plano: esférico -> cartesiano local (ENU)
# ----------------------------------------------------------------------


def para_enu(
    df: pd.DataFrame,
    lat0: float | None = None,
    lon0: float | None = None,
    ele0: float | None = None,
) -> pd.DataFrame:
    """Converte lat/lon/ele para coordenadas ENU locais, em metros.

    O caminho é o clássico geodésico -> ECEF -> ENU:

    1. (lat, lon, h) é convertido para ECEF (EPSG:4978), um referencial
       cartesiano cujo centro é o centro da Terra;
    2. o vetor ECEF relativo à origem local é rotacionado para o plano
       tangente naquele ponto, produzindo (east, north, up).

    A origem padrão é o primeiro ponto da trilha.
    """
    from pyproj import Transformer

    lat0 = float(df["lat"].iloc[0]) if lat0 is None else lat0
    lon0 = float(df["lon"].iloc[0]) if lon0 is None else lon0
    ele0 = float(df["ele"].iloc[0]) if ele0 is None else ele0

    # geodésico (lon, lat, h) -> ECEF (x, y, z)
    transf = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = transf.transform(df["lon"].values, df["lat"].values, df["ele"].values)
    x0, y0, z0 = transf.transform(lon0, lat0, ele0)

    dx, dy, dz = x - x0, y - y0, z - z0

    lam = np.deg2rad(lon0)  # longitude da origem
    phi = np.deg2rad(lat0)  # latitude  da origem

    # matriz de rotação ECEF -> ENU no ponto de origem
    rot = np.array(
        [
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ]
    )
    enu = rot @ np.vstack([dx, dy, dz])

    out = df.copy()
    out["east"] = enu[0]
    out["north"] = enu[1]
    out["up"] = enu[2]
    out.attrs["origem_enu"] = (lat0, lon0, ele0)
    return out


def para_utm(df: pd.DataFrame) -> pd.DataFrame:
    """Alternativa ao ENU: projeta em UTM e centra no primeiro ponto.

    Mais simples, porém sujeita à distorção da projeção; para trechos de
    poucos quilômetros a diferença em relação ao ENU é milimétrica.
    """
    from pyproj import Transformer

    lat0 = float(df["lat"].iloc[0])
    lon0 = float(df["lon"].iloc[0])
    zona = int((lon0 + 180) // 6) + 1
    epsg = 32600 + zona if lat0 >= 0 else 32700 + zona

    transf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = transf.transform(df["lon"].values, df["lat"].values)

    out = df.copy()
    out["east"] = e - e[0]
    out["north"] = n - n[0]
    out["up"] = df["ele"].values - df["ele"].iloc[0]
    out.attrs["epsg_utm"] = epsg
    return out


# ----------------------------------------------------------------------
# 1.3 Suavização
# ----------------------------------------------------------------------


def suavizar(df: pd.DataFrame, janela: int = 9, ordem: int = 2) -> pd.DataFrame:
    """Aplica Savitzky-Golay em east/north/up.

    O GPS de celular oscila na casa de metros entre amostras consecutivas;
    sem suavizar, o heading derivado fica inutilizável.
    """
    from scipy.signal import savgol_filter

    out = df.copy()
    n = len(out)
    janela = min(janela, n if n % 2 == 1 else n - 1)
    if janela <= ordem + 1:
        out.attrs["suavizacao"] = "ignorada (poucos pontos)"
        return out

    for col in ("east", "north", "up"):
        out[col] = savgol_filter(out[col].values, janela, ordem)

    out.attrs["suavizacao"] = f"savgol(janela={janela}, ordem={ordem})"
    return out


# ----------------------------------------------------------------------
# 1.4 Heading e velocidade
# ----------------------------------------------------------------------


def calcular_heading(df: pd.DataFrame, suavizar_saida: bool = True) -> pd.DataFrame:
    """Recalcula o heading a partir da evolução temporal do ENU.

    Convenção: ``heading = atan2(dEast, dNorth)`` — 0 rad aponta para o Norte
    e o ângulo cresce no sentido horário, como numa bússola.

    O ângulo é *unwrapped* antes de ser suavizado para não haver salto na
    passagem de 359° para 0°.
    """
    out = df.copy()
    de = np.gradient(out["east"].values)
    dn = np.gradient(out["north"].values)

    heading = np.arctan2(de, dn)

    if suavizar_saida and len(out) >= 5:
        from scipy.signal import savgol_filter

        janela = min(9, len(out) if len(out) % 2 == 1 else len(out) - 1)
        if janela > 3:
            heading = savgol_filter(np.unwrap(heading), janela, 2)

    out["heading"] = np.mod(heading, 2 * np.pi)
    return out


def calcular_velocidade(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona ``dist_acum`` (m) e ``velocidade`` (m/s)."""
    out = df.copy()
    de = np.diff(out["east"].values, prepend=out["east"].values[0])
    dn = np.diff(out["north"].values, prepend=out["north"].values[0])
    passo = np.hypot(de, dn)

    dt = np.diff(out["t"].values, prepend=out["t"].values[0] - 1e-6)
    dt[dt <= 0] = np.nan

    out["dist_acum"] = np.cumsum(passo)
    out["velocidade"] = np.nan_to_num(passo / dt)
    return out


# ----------------------------------------------------------------------
# 1.5 Pipeline do módulo
# ----------------------------------------------------------------------


def preparar(
    caminho_gpx: str | Path,
    janela_savgol: int = 9,
    ordem_savgol: int = 2,
    projecao: str = "enu",
) -> pd.DataFrame:
    """Executa o módulo 1 inteiro e devolve a trilha pronta para uso.

    Colunas de saída: ``lat lon ele tempo t east north up heading
    dist_acum velocidade``.
    """
    df = carregar_gpx(caminho_gpx)
    df = para_enu(df) if projecao == "enu" else para_utm(df)
    df = suavizar(df, janela_savgol, ordem_savgol)
    df = calcular_heading(df)
    df = calcular_velocidade(df)
    return df


def resumo(df: pd.DataFrame) -> str:
    """Texto curto com as estatísticas da trilha, para exibir no notebook."""
    dur = df["t"].iloc[-1]
    v = df["velocidade"]
    return "\n".join(
        [
            f"pontos ............. {len(df)}",
            f"duração ............ {dur:.0f} s ({dur / 60:.1f} min)",
            f"distância .......... {df['dist_acum'].iloc[-1]:.0f} m",
            f"intervalo médio .... {dur / max(len(df) - 1, 1):.2f} s",
            f"velocidade ......... média {v.mean() * 3.6:.1f} km/h | "
            f"máx {v.max() * 3.6:.1f} km/h",
            f"extensão ENU ....... east [{df['east'].min():.0f}, {df['east'].max():.0f}] m | "
            f"north [{df['north'].min():.0f}, {df['north'].max():.0f}] m",
        ]
    )
