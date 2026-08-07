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
# 2b.1b Estimativa automática do offset
# ----------------------------------------------------------------------


def perfil_de_movimento(
    caminho_video,
    passo_frames: int = 14,
    largura: int = 160,
    frame_inicial: int = 0,
    frame_final: int | None = None,
    recorte_superior: float = 0.45,
) -> pd.DataFrame:
    """Mede o movimento aparente ao longo do vídeo, via fluxo óptico denso.

    Para cada par de frames amostrados calcula o fluxo (Farnebäck) numa versão
    reduzida da imagem e extrai dois sinais:

    * ``fluxo``   — magnitude mediana, um substituto da **velocidade**: quando
      o carro para, cai a quase zero; quando acelera, cresce junto;
    * ``fluxo_x`` — componente horizontal média, um substituto da **taxa de
      guinada**: numa curva à direita a cena inteira desliza para a esquerda.

    ``recorte_superior`` limita a análise à fração de cima do quadro. Isso é
    essencial em vídeo gravado de dentro do carro: o painel e o capô ocupam a
    metade inferior e são estáticos, então incluí-los faria a mediana do fluxo
    ficar presa em zero, independentemente da velocidade.

    Devolve ``t`` (s de vídeo), ``fluxo`` e ``fluxo_x`` (px/amostra).
    """
    import cv2

    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_final = total if frame_final is None else min(frame_final, total)

    if frame_inicial:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)

    linhas, anterior, indice = [], None, frame_inicial
    try:
        while indice < frame_final:
            ok = cap.grab()
            if not ok:
                break
            if (indice - frame_inicial) % passo_frames == 0:
                ok, quadro = cap.retrieve()
                if not ok:
                    break
                h, w = quadro.shape[:2]
                pequeno = cv2.cvtColor(
                    cv2.resize(quadro, (largura, int(h * largura / w))),
                    cv2.COLOR_BGR2GRAY,
                )
                pequeno = pequeno[: max(int(pequeno.shape[0] * recorte_superior), 8)]

                if anterior is not None:
                    fluxo = cv2.calcOpticalFlowFarneback(
                        anterior, pequeno, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    linhas.append(
                        {
                            "t": indice / fps,
                            "fluxo": float(np.median(np.linalg.norm(fluxo, axis=2))),
                            "fluxo_x": float(np.mean(fluxo[..., 0])),
                        }
                    )
                anterior = pequeno
            indice += 1
    finally:
        cap.release()

    return pd.DataFrame(linhas)


def _normalizar(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def estimar_offset(
    trilha: pd.DataFrame,
    perfil: pd.DataFrame,
    faixa_s: tuple[float, float] = (-30.0, 30.0),
    passo_s: float = 0.1,
    sinal: str = "guinada",
) -> tuple[float, float, pd.DataFrame]:
    """Encontra o offset que melhor alinha o vídeo ao GPX, por correlação.

    Para cada candidato ``o``, o sinal derivado do GPX é reamostrado nos
    instantes ``t_vídeo - o`` e comparado ao perfil de movimento pela
    correlação de Pearson. O máximo da curva é o alinhamento procurado.

    ``sinal`` escolhe o que comparar:

    * ``"guinada"`` (padrão) — taxa de variação do heading contra o fluxo
      horizontal. Curvas são eventos bem localizados no tempo, o que produz um
      pico estreito e confiável;
    * ``"velocidade"`` — velocidade do GPS contra a magnitude do fluxo. Só
      funciona bem se houver paradas ou variações fortes de velocidade;
    * ``"ambos"`` — média das duas correlações.

    Devolve ``(offset, correlação, curva)``. Correlação baixa (< ~0.4) ou um
    pico pouco destacado indicam alinhamento não confiável — normalmente
    porque o trecho é retilíneo e de velocidade constante, sem âncora.
    """
    if len(perfil) < 5:
        raise ValueError("Perfil de movimento curto demais.")

    t_perfil = perfil["t"].values

    # sinais do vídeo
    mag = _normalizar(perfil["fluxo"].values)
    # cena deslizando para a esquerda (fluxo_x < 0) = carro virando à direita
    guinada_video = _normalizar(-perfil["fluxo_x"].values) if "fluxo_x" in perfil else None

    # sinais do GPX
    t_gps = trilha["t"].values
    vel_gps = trilha["velocidade"].values
    guinada_gps = np.gradient(np.unwrap(trilha["heading"].values), t_gps)

    candidatos = np.arange(faixa_s[0], faixa_s[1] + 1e-9, passo_s)
    linhas = []

    for o in candidatos:
        t = t_perfil - o
        dentro = (t >= t_gps[0]) & (t <= t_gps[-1])
        if dentro.sum() < 10:
            linhas.append((o, np.nan, np.nan))
            continue

        c_vel = float(np.mean(_normalizar(np.interp(t[dentro], t_gps, vel_gps)) * mag[dentro]))

        c_gui = np.nan
        if guinada_video is not None:
            c_gui = float(
                np.mean(
                    _normalizar(np.interp(t[dentro], t_gps, guinada_gps))
                    * guinada_video[dentro]
                )
            )
        linhas.append((o, c_vel, c_gui))

    curva = pd.DataFrame(linhas, columns=["offset", "corr_velocidade", "corr_guinada"])

    coluna = {
        "velocidade": "corr_velocidade",
        "guinada": "corr_guinada",
        "ambos": "corr_media",
    }[sinal]
    if sinal == "ambos":
        curva["corr_media"] = curva[["corr_velocidade", "corr_guinada"]].mean(axis=1)

    curva["correlacao"] = curva[coluna]
    melhor = curva.loc[curva["correlacao"].idxmax()]
    return float(melhor["offset"]), float(melhor["correlacao"]), curva


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
