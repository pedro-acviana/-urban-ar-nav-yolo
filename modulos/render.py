"""
Módulo 6b — Renderização
========================

Desenha o resultado sobre o frame do motorista: a fita de navegação colada
no asfalto, um minimapa em ENU e o painel de diagnóstico.

O gradiente de cor ao longo da fita não é enfeite: ele codifica distância,
dando ao motorista uma referência de profundidade que a perspectiva sozinha
não transmite bem em foto estática.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .calibration import Intrinsecos, PoseCamera
from .projection import gerar_fita, projetar_opencv, projetar_pinhole
from .sync import EstadoVeiculo

COR_PERTO = (0, 220, 255)  # BGR — âmbar
COR_LONGE = (0, 200, 80)  # BGR — verde


def interpolar_cor(c1, c2, t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


# ----------------------------------------------------------------------
# 6b.1 Fita de navegação
# ----------------------------------------------------------------------


def desenhar_rota(
    frame: np.ndarray,
    pontos_veiculo: np.ndarray,
    intr: Intrinsecos,
    pose: PoseCamera,
    largura_m: float = 1.8,
    com_distorcao: bool = False,
    alpha: float = 0.55,
    cor_perto=COR_PERTO,
    cor_longe=COR_LONGE,
    contorno: bool = True,
) -> np.ndarray:
    """Projeta a fita 3D e a compõe sobre o frame.

    A fita é fatiada em quadriláteros consecutivos e desenhada do mais
    distante para o mais próximo, de modo que os segmentos próximos fiquem
    por cima — a ordem de pintura correta para respeitar a profundidade.
    """
    if len(pontos_veiculo) < 2:
        return frame.copy()

    esq, dir_ = gerar_fita(pontos_veiculo, largura_m)
    projeta = (
        (lambda p: projetar_opencv(p, intr, pose, com_distorcao=True).uv)
        if com_distorcao
        else (lambda p: projetar_pinhole(p, intr, pose).uv)
    )
    uv_e, uv_d = projeta(esq), projeta(dir_)

    camada = frame.copy()
    n = len(uv_e) - 1

    for i in range(n - 1, -1, -1):
        quad = np.array(
            [uv_e[i], uv_e[i + 1], uv_d[i + 1], uv_d[i]], dtype=np.float64
        )
        if not np.isfinite(quad).all():
            continue
        cor = interpolar_cor(cor_perto, cor_longe, i / max(n - 1, 1))
        cv2.fillConvexPoly(camada, np.int32(quad), cor, cv2.LINE_AA)

    out = cv2.addWeighted(camada, alpha, frame, 1 - alpha, 0)

    if contorno:
        for uv in (uv_e, uv_d):
            pts = uv[np.isfinite(uv).all(axis=1)]
            if len(pts) > 1:
                cv2.polylines(
                    out, [np.int32(pts).reshape(-1, 1, 2)], False,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
    return out


def desenhar_marcadores_distancia(
    frame: np.ndarray,
    pontos_veiculo: np.ndarray,
    intr: Intrinsecos,
    pose: PoseCamera,
    passo_m: float = 10.0,
    com_distorcao: bool = False,
) -> np.ndarray:
    """Escreve a distância a cada N metros ao longo da rota."""
    out = frame.copy()
    p = np.asarray(pontos_veiculo)
    if len(p) < 2:
        return out

    for alvo in np.arange(passo_m, p[:, 0].max() + 1e-6, passo_m):
        i = int(np.argmin(np.abs(p[:, 0] - alvo)))
        proj = (
            projetar_opencv(p[i : i + 1], intr, pose, com_distorcao=True)
            if com_distorcao
            else projetar_pinhole(p[i : i + 1], intr, pose)
        )
        uv = proj.uv[0]
        if not np.isfinite(uv).all():
            continue
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(out, (u, v), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out, f"{alvo:.0f} m", (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 0), 5, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"{alvo:.0f} m", (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return out


def desenhar_horizonte(
    frame: np.ndarray, v_horizonte: float, cor=(80, 80, 255)
) -> np.ndarray:
    """Marca a linha do horizonte — conferência visual do pitch."""
    out = frame.copy()
    if not np.isfinite(v_horizonte):
        return out
    v = int(round(v_horizonte))
    if 0 <= v < out.shape[0]:
        cv2.line(out, (0, v), (out.shape[1], v), cor, 2, cv2.LINE_AA)
        cv2.putText(
            out, "horizonte", (12, max(v - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, cor, 2, cv2.LINE_AA,
        )
    return out


# ----------------------------------------------------------------------
# 6b.2 Minimapa
# ----------------------------------------------------------------------


def desenhar_minimapa(
    frame: np.ndarray,
    trilha: pd.DataFrame,
    rota_futura: pd.DataFrame,
    estado: EstadoVeiculo,
    tamanho: int = 320,
    margem: int = 24,
    raio_m: float = 60.0,
    supersample: int = 3,
    alpha: float = 0.88,
) -> np.ndarray:
    """Insere no canto superior direito um mapa ENU centrado no veículo.

    Renderizado em resolução ampliada e reduzido com ``INTER_AREA``: sem
    isso as linhas de 1 px ficam serrilhadas depois da composição.
    """
    R = tamanho * supersample
    mapa = np.full((R, R, 3), 18, dtype=np.uint8)

    escala = (R - 2 * int(16 * supersample)) / (2 * raio_m)

    def para_mapa(e, n):
        return (
            int(R / 2 + (e - estado.east) * escala),
            int(R / 2 - (n - estado.north) * escala),
        )

    for d in (10, 25, 50):
        r = int(d * escala)
        if r < R / 2:
            cv2.circle(mapa, (R // 2, R // 2), r, (50, 50, 50), supersample, cv2.LINE_AA)

    pts = [para_mapa(e, n) for e, n in zip(trilha["east"], trilha["north"])]
    if len(pts) > 1:
        cv2.polylines(
            mapa, [np.int32(pts).reshape(-1, 1, 2)], False,
            (120, 120, 120), 2 * supersample, cv2.LINE_AA,
        )

    pts_f = [para_mapa(e, n) for e, n in zip(rota_futura["east"], rota_futura["north"])]
    for i in range(1, len(pts_f)):
        cor = interpolar_cor(COR_PERTO, COR_LONGE, i / max(len(pts_f) - 1, 1))
        cv2.line(mapa, pts_f[i - 1], pts_f[i], cor, 3 * supersample, cv2.LINE_AA)

    cx, cy = para_mapa(estado.east, estado.north)
    ponta = (
        int(cx + 14 * supersample * np.sin(estado.heading)),
        int(cy - 14 * supersample * np.cos(estado.heading)),
    )
    cv2.arrowedLine(mapa, (cx, cy), ponta, (255, 255, 255), 2 * supersample, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(mapa, (cx, cy), 5 * supersample, (0, 0, 255), -1, cv2.LINE_AA)

    mapa = cv2.resize(mapa, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    cv2.rectangle(mapa, (0, 0), (tamanho - 1, tamanho - 1), (210, 210, 210), 2, cv2.LINE_AA)

    out = frame.copy()
    h, w = out.shape[:2]
    x0, y0 = w - tamanho - margem, margem
    if x0 < 0 or y0 + tamanho > h:
        return out
    roi = out[y0 : y0 + tamanho, x0 : x0 + tamanho]
    out[y0 : y0 + tamanho, x0 : x0 + tamanho] = cv2.addWeighted(
        mapa, alpha, roi, 1 - alpha, 0
    )
    return out


# ----------------------------------------------------------------------
# 6b.3 Painel de texto
# ----------------------------------------------------------------------


def desenhar_hud(
    frame: np.ndarray, linhas: list[str], margem: int = 24, escala: float = 0.9
) -> np.ndarray:
    """Caixa semitransparente com os parâmetros do frame."""
    out = frame.copy()
    if not linhas:
        return out

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    espessura = max(1, int(round(escala * 2)))
    alturas = [cv2.getTextSize(t, fonte, escala, espessura)[0] for t in linhas]
    larg = max(a[0] for a in alturas) + 2 * margem
    passo = int(max(a[1] for a in alturas) * 1.9)
    alt = passo * len(linhas) + margem

    caixa = out[margem : margem + alt, margem : margem + larg].copy()
    caixa[:] = (0, 0, 0)
    out[margem : margem + alt, margem : margem + larg] = cv2.addWeighted(
        caixa, 0.55, out[margem : margem + alt, margem : margem + larg], 0.45, 0
    )

    y = margem + passo - 6
    for texto in linhas:
        cv2.putText(out, texto, (margem + 14, y), fonte, escala, (255, 255, 255), espessura, cv2.LINE_AA)
        y += passo
    return out


def montar_painel(imagens: dict[str, np.ndarray], largura_col: int = 640) -> np.ndarray:
    """Empilha vistas nomeadas numa tira horizontal, para comparação rápida."""
    tiras = []
    for titulo, img in imagens.items():
        if img is None:
            continue
        v = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h = int(v.shape[0] * largura_col / v.shape[1])
        v = cv2.resize(v, (largura_col, h))
        faixa = np.zeros((36, largura_col, 3), dtype=np.uint8)
        cv2.putText(faixa, titulo, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiras.append(np.vstack([faixa, v]))

    if not tiras:
        raise ValueError("Nada para montar.")

    alt = max(t.shape[0] for t in tiras)
    tiras = [
        np.vstack([t, np.zeros((alt - t.shape[0], t.shape[1], 3), dtype=np.uint8)])
        for t in tiras
    ]
    return np.hstack(tiras)
