"""
Módulo 2b — Sincronização GPS <-> vídeo e rota futura
=====================================================

Une o módulo 1 (trilha GPS em ENU) ao módulo 2a (frames). Responde a duas
perguntas:

1. **Onde o carro estava** no instante em que o frame alvo foi capturado?
2. **Por onde ele vai passar** nos próximos segundos, escrito no referencial
   do veículo naquele instante (X frente, Y esquerda, Z cima)?

A saída deste módulo é exatamente o conjunto de pontos ``X_W`` do Passo 1 do
plano, prontos para receberem os extrínsecos no Passo 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .video import InfoVideo


# ----------------------------------------------------------------------
# 2b.1 Alinhamento temporal
# ----------------------------------------------------------------------


def alinhar_tempos(df: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    """Cria a coluna ``t_video``: tempo de cada ponto GPS no relógio do vídeo.

    ``t_video = t_gpx + offset_s``

    Como ``t_gpx`` já é relativo ao primeiro ponto da trilha, um offset
    negativo significa que o GPX começou a gravar **antes** do vídeo.
    """
    out = df.copy()
    out["t_video"] = out["t"] + offset_s
    return out


def diagnosticar(df: pd.DataFrame, info_video: InfoVideo, offset_s: float) -> str:
    """Relatório de sanidade da sincronização, para inspeção no notebook."""
    dfa = alinhar_tempos(df, offset_s)
    ini, fim = dfa["t_video"].iloc[0], dfa["t_video"].iloc[-1]
    sobrepos = min(fim, info_video.duracao_s) - max(ini, 0.0)
    return "\n".join(
        [
            f"vídeo .............. 0.0 s -> {info_video.duracao_s:.1f} s "
            f"({info_video.n_frames} frames @ {info_video.fps:.2f} fps)",
            f"GPX (alinhado) ..... {ini:.1f} s -> {fim:.1f} s (offset {offset_s:+.2f} s)",
            f"sobreposição ....... {sobrepos:.1f} s "
            f"({'OK' if sobrepos > 0 else 'SEM SOBREPOSIÇÃO — revise o offset'})",
        ]
    )


# ----------------------------------------------------------------------
# 2b.2 Estado do veículo num frame
# ----------------------------------------------------------------------


@dataclass
class EstadoVeiculo:
    """Pose do carro no instante do frame, no referencial ENU."""

    frame: int
    t_video: float
    east: float
    north: float
    up: float
    heading: float  # rad, 0 = Norte, horário
    velocidade: float  # m/s

    @property
    def heading_deg(self) -> float:
        return float(np.rad2deg(self.heading) % 360)

    def __str__(self) -> str:
        return (
            f"frame {self.frame} | t={self.t_video:.2f}s | "
            f"ENU=({self.east:.1f}, {self.north:.1f}) m | "
            f"heading={self.heading_deg:.1f}° | v={self.velocidade * 3.6:.1f} km/h"
        )


def _interp(t: float, ts: np.ndarray, valores: np.ndarray) -> float:
    return float(np.interp(t, ts, valores))


def estado_no_frame(
    df: pd.DataFrame, frame: int, info_video: InfoVideo, offset_s: float
) -> EstadoVeiculo:
    """Interpola a posição/heading do carro no instante exato do frame.

    O heading é interpolado sobre o ângulo *unwrapped*, evitando que a média
    entre 359° e 1° caia em 180°.
    """
    dfa = alinhar_tempos(df, offset_s)
    t = info_video.tempo_do_frame(frame)

    ts = dfa["t_video"].values
    if not (ts[0] <= t <= ts[-1]):
        raise ValueError(
            f"O frame {frame} (t={t:.2f}s) está fora da janela do GPX "
            f"[{ts[0]:.2f}, {ts[-1]:.2f}] s. Ajuste o offset de sincronização."
        )

    heading_unwrap = np.unwrap(dfa["heading"].values)

    return EstadoVeiculo(
        frame=frame,
        t_video=t,
        east=_interp(t, ts, dfa["east"].values),
        north=_interp(t, ts, dfa["north"].values),
        up=_interp(t, ts, dfa["up"].values),
        heading=float(np.mod(_interp(t, ts, heading_unwrap), 2 * np.pi)),
        velocidade=_interp(t, ts, dfa["velocidade"].values),
    )


# ----------------------------------------------------------------------
# 2b.3 Rota futura
# ----------------------------------------------------------------------


def trajetoria_futura(
    df: pd.DataFrame,
    estado: EstadoVeiculo,
    offset_s: float,
    horizonte_s: float = 12.0,
    distancia_max_m: float = 40.0,
    passo_m: float = 0.5,
) -> pd.DataFrame:
    """Reamostra a rota à frente do carro em passos regulares de distância.

    Reamostrar por **distância** (e não por tempo) é importante: o GPS grava a
    cada poucos segundos, então em velocidade alta os pontos ficam esparsos
    justamente onde a projeção precisa de resolução.

    Devolve um DataFrame com ``east``, ``north``, ``up`` e ``s`` (distância
    percorrida a partir do carro, em metros).
    """
    dfa = alinhar_tempos(df, offset_s)
    t0 = estado.t_video

    janela = dfa[(dfa["t_video"] >= t0) & (dfa["t_video"] <= t0 + horizonte_s)]
    if len(janela) < 2:
        raise ValueError(
            "Menos de 2 pontos GPS no horizonte pedido — aumente `horizonte_s`."
        )

    # o primeiro ponto é a posição interpolada do carro, não a amostra bruta
    east = np.concatenate([[estado.east], janela["east"].values])
    north = np.concatenate([[estado.north], janela["north"].values])
    up = np.concatenate([[estado.up], janela["up"].values])

    # distância acumulada ao longo da poligonal
    passos = np.hypot(np.diff(east), np.diff(north))
    s = np.concatenate([[0.0], np.cumsum(passos)])

    # remove pontos coincidentes (carro parado) para poder interpolar
    validos = np.concatenate([[True], np.diff(s) > 1e-6])
    east, north, up, s = east[validos], north[validos], up[validos], s[validos]

    if len(s) < 2:
        raise ValueError("Veículo parado no intervalo: não há rota futura.")

    s_max = min(s[-1], distancia_max_m)
    s_novo = np.arange(0.0, s_max + 1e-9, passo_m)

    return pd.DataFrame(
        {
            "s": s_novo,
            "east": np.interp(s_novo, s, east),
            "north": np.interp(s_novo, s, north),
            "up": np.interp(s_novo, s, up),
        }
    )


# ----------------------------------------------------------------------
# 2b.4 ENU -> referencial do veículo  (os pontos X_W do Passo 1)
# ----------------------------------------------------------------------


def enu_para_veiculo(
    rota: pd.DataFrame, estado: EstadoVeiculo, usar_elevacao: bool = False
) -> np.ndarray:
    """Reescreve a rota no referencial do veículo (X frente, Y esquerda, Z cima).

    Com ``heading`` medido a partir do Norte no sentido horário, os versores
    da base do veículo no plano ENU são::

        frente   f = ( sin h,  cos h)
        esquerda l = (-cos h,  sin h)

    de modo que, para um deslocamento ``d = (de, dn)``:

        X =  de·sin h + dn·cos h
        Y = -de·cos h + dn·sin h

    ``Z`` vem da elevação relativa (se ``usar_elevacao``) ou é zerado, adotando
    o modelo de solo plano — mais estável, já que a altitude de GPS de celular
    tem erro da ordem de metros.
    """
    h = estado.heading
    de = rota["east"].values - estado.east
    dn = rota["north"].values - estado.north

    x = de * np.sin(h) + dn * np.cos(h)
    y = -de * np.cos(h) + dn * np.sin(h)

    if usar_elevacao:
        z = rota["up"].values - estado.up
    else:
        z = np.zeros_like(x)

    return np.column_stack([x, y, z])


def apenas_a_frente(pontos: np.ndarray, x_min: float = 0.5) -> np.ndarray:
    """Descarta waypoints atrás da câmera (X <= x_min).

    Sem isso, pontos com profundidade negativa produzem projeções espúrias
    (a divisão por ``z_c`` inverte o sinal e a rota "aparece" no céu).
    """
    return pontos[pontos[:, 0] > x_min]
