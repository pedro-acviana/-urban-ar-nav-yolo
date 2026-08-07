"""
Módulo 6a — Projeção 3D -> 2D
=============================

Fecha os **Passos 2, 3 e parte do 4** do plano: leva os waypoints do
referencial do veículo até coordenadas de pixel, e aplica os critérios de
recorte (limites da tela, área dirigível e oclusão por obstáculos).

Duas implementações equivalentes:

* :func:`projetar_pinhole` — o modelo puro, escrito passo a passo, para o
  notebook mostrar a matemática;
* :func:`projetar_opencv` — usa ``cv2.projectPoints``, que aplica também os
  coeficientes de distorção (necessário se a imagem de fundo **não** foi
  corrigida).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Intrinsecos, PoseCamera


# ----------------------------------------------------------------------
# 6a.1 Projeção
# ----------------------------------------------------------------------


@dataclass
class Projecao:
    uv: np.ndarray  # (N,2) coordenadas de pixel
    profundidade: np.ndarray  # (N,) z_c em metros
    a_frente: np.ndarray  # (N,) bool: z_c > 0

    def __len__(self) -> int:
        return len(self.uv)


def mundo_para_camera(pontos_veiculo: np.ndarray, pose: PoseCamera) -> np.ndarray:
    """Passo 2 — ``X_C = R · X_W + t``.

    Aceita (N,2) [assume Z=0] ou (N,3). Devolve (N,3) no referencial da câmera.
    """
    p = np.asarray(pontos_veiculo, dtype=np.float64).reshape(-1, np.shape(pontos_veiculo)[-1])
    if p.shape[1] == 2:
        p = np.column_stack([p, np.zeros(len(p))])
    return (pose.R @ p.T + pose.t).T


def projetar_pinhole(
    pontos_veiculo: np.ndarray, intr: Intrinsecos, pose: PoseCamera
) -> Projecao:
    """Passos 2 e 3 explícitos, sem distorção.

        X_C = R·X_W + t
        ỹ   = K·X_C
        u = ỹ₀/ỹ₂ ,  v = ỹ₁/ỹ₂

    A divisão por ``ỹ₂ = z_c`` é o que introduz a perspectiva: o mesmo
    deslocamento lateral em metros ocupa menos pixels quanto mais longe.
    Pontos com ``z_c <= 0`` estão atrás da câmera e são marcados como
    inválidos (projetá-los produziria uma imagem espelhada no céu).
    """
    Xc = mundo_para_camera(pontos_veiculo, pose)
    y_til = (intr.K @ Xc.T).T

    z = y_til[:, 2]
    a_frente = z > 1e-6
    z_seguro = np.where(a_frente, z, np.nan)

    uv = np.column_stack([y_til[:, 0] / z_seguro, y_til[:, 1] / z_seguro])
    return Projecao(uv=uv, profundidade=Xc[:, 2], a_frente=a_frente)


def projetar_opencv(
    pontos_veiculo: np.ndarray,
    intr: Intrinsecos,
    pose: PoseCamera,
    com_distorcao: bool = True,
) -> Projecao:
    """Mesma projeção via ``cv2.projectPoints``, opcionalmente com distorção.

    Use ``com_distorcao=True`` para desenhar sobre o frame **original**, e
    ``False`` (ou :func:`projetar_pinhole`) sobre o frame já corrigido.
    """
    p = np.asarray(pontos_veiculo, dtype=np.float64).reshape(-1, np.shape(pontos_veiculo)[-1])
    if p.shape[1] == 2:
        p = np.column_stack([p, np.zeros(len(p))])

    dist = intr.dist if com_distorcao else np.zeros((1, 5))
    uv, _ = cv2.projectPoints(p, pose.rvec, pose.tvec, intr.K, dist)

    Xc = mundo_para_camera(p, pose)
    return Projecao(
        uv=uv.reshape(-1, 2), profundidade=Xc[:, 2], a_frente=Xc[:, 2] > 1e-6
    )


def projetar_homografia(pontos_xy: np.ndarray, H_solo_para_imagem: np.ndarray) -> np.ndarray:
    """Atalho para pontos no plano do solo: ``(u,v,w) = H·(X,Y,1)``.

    Equivale à projeção completa quando ``Z = 0``, e é o caminho de volta da
    bird's-eye view para a imagem.
    """
    p = np.asarray(pontos_xy, dtype=np.float64).reshape(-1, np.shape(pontos_xy)[-1])[:, :2]
    hom = np.column_stack([p, np.ones(len(p))])
    proj = (H_solo_para_imagem @ hom.T).T
    w = proj[:, 2]
    w = np.where(np.abs(w) > 1e-12, w, np.nan)
    return np.column_stack([proj[:, 0] / w, proj[:, 1] / w])


# ----------------------------------------------------------------------
# 6a.2 Passo 4 — recortes e oclusão
# ----------------------------------------------------------------------


def dentro_da_imagem(
    uv: np.ndarray, largura: int, altura: int, folga: int = 0
) -> np.ndarray:
    u, v = uv[:, 0], uv[:, 1]
    return (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= -folga)
        & (u < largura + folga)
        & (v >= -folga)
        & (v < altura + folga)
    )


def sobre_a_via(uv: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """True para os pontos que caem dentro da máscara de área dirigível."""
    h, w = mascara.shape[:2]
    ok = dentro_da_imagem(uv, w, h)
    saida = np.zeros(len(uv), dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx):
        u = np.clip(uv[idx, 0].astype(int), 0, w - 1)
        v = np.clip(uv[idx, 1].astype(int), 0, h - 1)
        saida[idx] = mascara[v, u] > 0
    return saida


def ocluido_por_obstaculo(
    uv: np.ndarray, caixas: np.ndarray, so_borda_inferior: bool = True
) -> np.ndarray:
    """True para os pontos escondidos atrás de um obstáculo detectado.

    Um veículo à frente ocupa a imagem do asfalto que está **depois** dele.
    Como a rota é desenhada no chão, basta testar se o ponto cai dentro da
    caixa: por construção ele estaria sob o carro da frente. Com
    ``so_borda_inferior=True`` a checagem se restringe ao terço inferior da
    caixa (onde os pneus tocam o solo), reduzindo falsos positivos com
    caixas altas.
    """
    if caixas is None or len(caixas) == 0:
        return np.zeros(len(uv), dtype=bool)

    ocluido = np.zeros(len(uv), dtype=bool)
    u, v = uv[:, 0], uv[:, 1]
    for x1, y1, x2, y2, *_ in caixas:
        topo = y1 + (y2 - y1) * (0.66 if so_borda_inferior else 0.0)
        ocluido |= (u >= x1) & (u <= x2) & (v >= topo) & (v <= y2)
    return ocluido


def truncar_no_primeiro_corte(
    pontos_veiculo: np.ndarray, valido: np.ndarray
) -> np.ndarray:
    """Mantém apenas o trecho contínuo que sai do capô.

    Uma rota que reaparece depois de um obstáculo confunde mais do que
    ajuda: o desenho deve terminar no primeiro ponto invisível.
    """
    if not len(valido) or not valido[0]:
        # tolera alguns pontos iniciais fora (borda inferior da imagem)
        primeiro = int(np.argmax(valido)) if valido.any() else len(valido)
        if primeiro >= len(valido):
            return pontos_veiculo[:0]
        valido = valido[primeiro:]
        pontos_veiculo = pontos_veiculo[primeiro:]

    corte = np.flatnonzero(~valido)
    fim = int(corte[0]) if len(corte) else len(valido)
    return pontos_veiculo[:fim]


def filtrar_rota(
    pontos_veiculo: np.ndarray,
    intr: Intrinsecos,
    pose: PoseCamera,
    mascara_via: np.ndarray | None = None,
    obstaculos: np.ndarray | None = None,
    com_distorcao: bool = False,
    exigir_via: bool = True,
    truncar: bool = True,
) -> tuple[np.ndarray, Projecao]:
    """Aplica todos os critérios do Passo 4 e devolve a rota utilizável.

    Retorna ``(pontos_no_veiculo, projecao)`` já recortados, para que o
    módulo de renderização possa gerar a fita 3D a partir dos mesmos pontos.
    """
    pontos = np.asarray(pontos_veiculo, dtype=np.float64)
    proj = (
        projetar_opencv(pontos, intr, pose, com_distorcao=True)
        if com_distorcao
        else projetar_pinhole(pontos, intr, pose)
    )

    largura, altura = intr.resolucao
    valido = proj.a_frente & dentro_da_imagem(proj.uv, largura, altura, folga=40)

    if mascara_via is not None and exigir_via:
        valido &= sobre_a_via(proj.uv, mascara_via)
    if obstaculos is not None and len(obstaculos):
        valido &= ~ocluido_por_obstaculo(proj.uv, obstaculos)

    pontos_ok = truncar_no_primeiro_corte(pontos, valido) if truncar else pontos[valido]

    proj_ok = (
        projetar_opencv(pontos_ok, intr, pose, com_distorcao=True)
        if com_distorcao
        else projetar_pinhole(pontos_ok, intr, pose)
    )
    return pontos_ok, proj_ok


# ----------------------------------------------------------------------
# 6a.3 Fita 3D (rota com largura)
# ----------------------------------------------------------------------


def gerar_fita(pontos_veiculo: np.ndarray, largura_m: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """Cria as bordas esquerda e direita da rota, em metros.

    A largura é dada **no mundo**, não em pixels: depois de projetada, a
    fita afina naturalmente com a distância, que é justamente o que dá a
    sensação de estar colada no asfalto.
    """
    p = np.asarray(pontos_veiculo, dtype=np.float64)
    if len(p) < 2:
        vazio = np.zeros((0, p.shape[1] if p.ndim > 1 else 3))
        return vazio, vazio

    xy = p[:, :2]
    tang = np.gradient(xy, axis=0)
    norma = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(norma > 1e-9, norma, 1.0)

    # normal à esquerda no plano do solo: (-ty, tx)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    meia = largura_m / 2.0

    z = p[:, 2:3] if p.shape[1] > 2 else np.zeros((len(p), 1))
    esq = np.hstack([xy + meia * normal, z])
    dir_ = np.hstack([xy - meia * normal, z])
    return esq, dir_
