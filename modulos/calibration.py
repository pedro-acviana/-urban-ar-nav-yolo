"""
Módulo 3 — Calibração: intrínsecos e extrínsecos
================================================

Implementa os **Passos 2 e 3** do plano.

**Intrínsecos (K, dist)** vêm do ``calibracao_camera.json`` produzido pelo
tabuleiro de xadrez. Como as fotos de calibração têm resolução diferente da
do vídeo, ``K`` precisa ser reescalado.

**Extrínsecos (R, t)** *não* podem ser reaproveitados daquele JSON: os
``rvec``/``tvec`` gravados lá descrevem a pose de cada tabuleiro em relação à
câmera, não a pose da câmera dentro do carro. Por isso a pose de montagem é
declarada explicitamente (altura, pitch, yaw, roll) e ajustada visualmente.

Encadeamento completo (mundo -> pixel)::

    X_C = R · X_V + t              (Passo 2, extrínsecos)
    ỹ   = K · X_C                  (Passo 3, intrínsecos)
    u = ỹ₀/ỹ₂ ,  v = ỹ₁/ỹ₂         (divisão pela profundidade)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 3.1 Intrínsecos
# ----------------------------------------------------------------------


@dataclass
class Intrinsecos:
    K: np.ndarray  # 3x3
    dist: np.ndarray  # 1x5
    resolucao: tuple[int, int]  # (largura, altura) a que este K se refere
    erro_rms: float | None = None

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def fov_horizontal_deg(self) -> float:
        return float(np.rad2deg(2 * np.arctan(self.resolucao[0] / (2 * self.fx))))

    @property
    def fov_vertical_deg(self) -> float:
        return float(np.rad2deg(2 * np.arctan(self.resolucao[1] / (2 * self.fy))))

    def __str__(self) -> str:
        return (
            f"K para {self.resolucao[0]}x{self.resolucao[1]}: "
            f"fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f} | "
            f"FOV {self.fov_horizontal_deg:.1f}° x {self.fov_vertical_deg:.1f}°"
            + (f" | RMS {self.erro_rms:.3f} px" if self.erro_rms else "")
        )


def carregar_intrinsecos(caminho: str | Path) -> Intrinsecos:
    """Lê o JSON de calibração.

    A resolução original não está gravada no arquivo, mas o ponto principal
    fica muito próximo do centro da imagem, então ela é recuperada como
    ``(2·cx, 2·cy)``.
    """
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

    intr = dados["parametros_intrinsecos"]
    K = np.array(intr["matriz_camera_mtx"], dtype=np.float64)
    dist = np.array(intr["coeficientes_distorcao_dist"], dtype=np.float64).reshape(1, -1)

    largura = int(round(2 * K[0, 2]))
    altura = int(round(2 * K[1, 2]))

    return Intrinsecos(
        K=K,
        dist=dist,
        resolucao=(largura, altura),
        erro_rms=dados.get("erro_reprojecao_rms"),
    )


def reescalar(intr: Intrinsecos, resolucao_destino: tuple[int, int]) -> Intrinsecos:
    """Adapta ``K`` para a resolução do vídeo.

    Distância focal e ponto principal escalam linearmente com a resolução;
    os coeficientes de distorção são adimensionais (normalizados por f) e
    permanecem inalterados.
    """
    lw, lh = intr.resolucao
    dw, dh = resolucao_destino
    sx, sy = dw / lw, dh / lh

    K = intr.K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    return Intrinsecos(
        K=K, dist=intr.dist.copy(), resolucao=(dw, dh), erro_rms=intr.erro_rms
    )


def corrigir_distorcao(imagem: np.ndarray, intr: Intrinsecos) -> np.ndarray:
    """Remove a distorção radial/tangencial mantendo o mesmo ``K``.

    A partir daqui a imagem obedece ao modelo pinhole puro, o que torna a
    projeção e a IPM exatamente inversas uma da outra.
    """
    return cv2.undistort(imagem, intr.K, intr.dist, None, intr.K)


# ----------------------------------------------------------------------
# 3.2 Extrínsecos — pose da câmera no veículo
# ----------------------------------------------------------------------


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Alinhamento entre a base do veículo (X frente, Y esquerda, Z cima) e a
# base da câmera OpenCV (X direita, Y baixo, Z frente), com pose neutra:
#     X_cam = -Y_veh ,  Y_cam = -Z_veh ,  Z_cam = X_veh
R_BASE = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


@dataclass
class PoseCamera:
    """Pose de montagem da câmera no veículo (ajuste manual).

    altura_m : altura do centro óptico em relação ao asfalto.
    pitch_deg: NEGATIVO = apontando para baixo (dashcam típica: -5° a -15°).
    yaw_deg  : POSITIVO = girada para a esquerda.
    roll_deg : POSITIVO = horizonte caindo para a direita.
    """

    altura_m: float = 1.50
    pitch_deg: float = -10.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    # ------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        """Matriz de rotação mundo(veículo) -> câmera."""
        return (
            _rot_z(np.deg2rad(self.roll_deg))
            @ _rot_x(-np.deg2rad(self.pitch_deg))
            @ _rot_y(np.deg2rad(self.yaw_deg))
            @ R_BASE
        )

    @property
    def centro(self) -> np.ndarray:
        """Posição do centro óptico no referencial do veículo (3x1)."""
        return np.array([[0.0], [0.0], [self.altura_m]])

    @property
    def t(self) -> np.ndarray:
        """Vetor de translação: ``t = -R · C``, tal que ``X_C = R·X_V + t``."""
        return -self.R @ self.centro

    @property
    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0]

    @property
    def tvec(self) -> np.ndarray:
        return self.t

    @property
    def Rt(self) -> np.ndarray:
        """Matriz extrínseca 4x4 em coordenadas homogêneas."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3:] = self.t
        return M

    def __str__(self) -> str:
        return (
            f"altura={self.altura_m:.2f} m | pitch={self.pitch_deg:+.1f}° "
            f"| yaw={self.yaw_deg:+.1f}° | roll={self.roll_deg:+.1f}°"
        )


def matriz_projecao(intr: Intrinsecos, pose: PoseCamera) -> np.ndarray:
    """Matriz de projeção completa ``P = K·[R|t]`` (3x4)."""
    return intr.K @ np.hstack([pose.R, pose.t])


# ----------------------------------------------------------------------
# 3.3 Homografia do plano do solo
# ----------------------------------------------------------------------


def homografia_solo_para_imagem(intr: Intrinsecos, pose: PoseCamera) -> np.ndarray:
    """Homografia 3x3 que leva (X, Y, 1) do asfalto para (u, v, 1) na imagem.

    Para pontos com ``Z = 0`` a terceira coluna de ``R`` some, e a projeção
    ``P·(X, Y, 0, 1)`` colapsa em::

        H = K · [ r₁ | r₂ | t ]

    Essa é a matriz que a IPM (módulo 5) inverte para gerar a bird's-eye view.
    """
    R, t = pose.R, pose.t
    return intr.K @ np.hstack([R[:, 0:1], R[:, 1:2], t])


def linha_do_horizonte(intr: Intrinsecos, pose: PoseCamera) -> float:
    """Coordenada ``v`` (linha) do horizonte na imagem.

    Corresponde à projeção do ponto de fuga do plano do solo — o limite
    acima do qual nada de solo pode ser desenhado. Serve de sanidade para o
    pitch: se o horizonte cair fora da imagem, o ângulo está exagerado.
    """
    # direção "frente" do veículo vista pela câmera (ponto no infinito)
    d = pose.R @ np.array([1.0, 0.0, 0.0])
    if abs(d[2]) < 1e-9:
        return float("inf")
    p = intr.K @ d
    return float(p[1] / p[2])


def pitch_pelo_horizonte(intr: Intrinsecos, v_horizonte: float) -> float:
    """Inverte :func:`linha_do_horizonte`: dado o horizonte, devolve o pitch.

    Atalho prático de ajuste — em vez de tentar valores de pitch às cegas,
    basta identificar na imagem a linha onde o solo encontra o céu e chamar
    esta função. Com ``roll = 0``::

        v_horizonte = c_y - f_y · tan(-pitch)

    Note que a altura da câmera **não** entra: o horizonte é a projeção de um
    ponto no infinito e independe de onde a câmera está.
    """
    return float(-np.rad2deg(np.arctan((intr.cy - v_horizonte) / intr.fy)))


def altura_por_referencia(
    intr: Intrinsecos, pitch_deg: float, v: float, distancia_m: float
) -> float:
    """Estima a altura da câmera a partir de um ponto de distância conhecida.

    Se você souber que um objeto no asfalto (faixa de pedestre, tampa de bueiro)
    está a ``distancia_m`` do carro e ele aparece na linha ``v`` da imagem,
    esta função devolve a altura coerente com essa observação.
    """
    ang = np.arctan((v - intr.cy) / intr.fy) - np.deg2rad(pitch_deg)
    if ang <= 1e-6:
        raise ValueError("Ponto acima do horizonte: não há solução no solo.")
    return float(distancia_m * np.tan(ang))


def ajustar_pose_por_referencias(
    intr: Intrinsecos,
    referencias: list[tuple[float, float]],
    altura_inicial: float = 1.50,
    pitch_inicial: float = -10.0,
) -> tuple[PoseCamera, dict]:
    """Ajusta altura **e** pitch a partir de pontos de distância conhecida.

    ``referencias`` é uma lista de pares ``(v, distancia_m)``: a linha da
    imagem em que um ponto do asfalto aparece e a distância real dele até o
    carro. Dois pontos já bastam; três ou mais permitem ver o resíduo.

    Por que não ajustar só a altura: no eixo central, um ponto a distância
    ``d`` cai na linha

    .. math::
        v = c_y + f_y \\, \\frac{h\\cos\\theta - d\\sin\\theta}
                              {h\\sin\\theta + d\\cos\\theta},
        \\quad \\theta = -\\text{pitch}

    A altura escala as distâncias de forma aproximadamente proporcional; o
    pitch as desloca de forma não linear, com efeito muito maior ao longe.
    Um erro que aparece como "5 m a mais em toda a grade" não é explicável só
    pela altura — ela teria que assumir um valor diferente a cada distância —
    e por isso os dois parâmetros são estimados juntos, por mínimos quadrados
    sobre os resíduos em pixels.

    Devolve ``(pose, relatorio)``.
    """
    from scipy.optimize import least_squares

    if len(referencias) < 2:
        raise ValueError(
            "São necessários ao menos 2 pontos de referência para separar "
            "altura de pitch. Com 1 ponto, use altura_por_referencia()."
        )

    vs = np.array([r[0] for r in referencias], dtype=float)
    ds = np.array([r[1] for r in referencias], dtype=float)

    if np.any(ds <= 0):
        raise ValueError("Distâncias de referência devem ser positivas.")

    def linha_prevista(h: float, pitch_deg: float, d: np.ndarray) -> np.ndarray:
        th = -np.deg2rad(pitch_deg)
        num = h * np.cos(th) - d * np.sin(th)
        den = h * np.sin(th) + d * np.cos(th)
        return intr.cy + intr.fy * num / den

    def residuo(p):
        return linha_prevista(p[0], p[1], ds) - vs

    sol = least_squares(
        residuo,
        x0=[altura_inicial, pitch_inicial],
        bounds=([0.3, -45.0], [3.0, 45.0]),
    )
    altura, pitch = float(sol.x[0]), float(sol.x[1])

    res_px = residuo(sol.x)
    # o mesmo resíduo, lido em metros de distância
    dist_ajustada = np.array(
        [distancia_no_solo(intr, PoseCamera(altura, pitch), v) for v in vs]
    )

    relatorio = {
        "altura_m": altura,
        "pitch_deg": pitch,
        "residuo_px_rms": float(np.sqrt(np.mean(res_px**2))),
        "residuo_px_max": float(np.max(np.abs(res_px))),
        "distancia_esperada_m": ds,
        "distancia_ajustada_m": dist_ajustada,
        "erro_m": dist_ajustada - ds,
        "convergiu": bool(sol.success),
    }
    return PoseCamera(altura_m=altura, pitch_deg=pitch), relatorio


def relatorio_referencias(relatorio: dict) -> str:
    """Formata o resultado de :func:`ajustar_pose_por_referencias`."""
    linhas = [
        f"altura .......... {relatorio['altura_m']:.3f} m",
        f"pitch ........... {relatorio['pitch_deg']:+.2f}°",
        f"resíduo ......... {relatorio['residuo_px_rms']:.1f} px RMS "
        f"(máx {relatorio['residuo_px_max']:.1f} px)",
        "",
        f"{'esperado':>10} {'ajustado':>10} {'erro':>8}",
    ]
    for e, a, err in zip(
        relatorio["distancia_esperada_m"],
        relatorio["distancia_ajustada_m"],
        relatorio["erro_m"],
    ):
        linhas.append(f"{e:>9.1f}m {a:>9.1f}m {err:>+7.2f}m")
    return "\n".join(linhas)


def distancia_no_solo(intr: Intrinsecos, pose: PoseCamera, v: float) -> float:
    """Distância à frente (m) correspondente à linha ``v`` no eixo central.

    Útil para ler a escala da imagem: "esta linha de pixels está a X metros".
    Devolve ``inf`` para linhas na altura ou acima do horizonte.
    """
    H = homografia_solo_para_imagem(intr, pose)
    Hinv = np.linalg.inv(H)
    p = Hinv @ np.array([intr.cx, v, 1.0])
    if abs(p[2]) < 1e-12:
        return float("inf")
    x = p[0] / p[2]
    return float(x) if x > 0 else float("inf")


# ----------------------------------------------------------------------
# 3.4 Pipeline do módulo
# ----------------------------------------------------------------------


def preparar(
    caminho_json: str | Path,
    resolucao_video: tuple[int, int],
    altura_m: float = 1.50,
    pitch_deg: float = -10.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[Intrinsecos, PoseCamera]:
    """Carrega ``K``/``dist``, reescala para o vídeo e monta a pose manual."""
    intr = reescalar(carregar_intrinsecos(caminho_json), resolucao_video)
    pose = PoseCamera(
        altura_m=altura_m, pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg
    )
    return intr, pose
