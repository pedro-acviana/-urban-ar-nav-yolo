"""
Módulo 5 — IPM / Bird's-Eye View
================================

Implementa o "truque" descrito no plano: em vez de projetar a rota direto na
foto, primeiro leva-se **a imagem** para uma vista superior métrica
(Inverse Perspective Mapping), onde o asfalto é um plano sem distorção
projetiva. Nessa vista:

1. a rota do GPS e a área dirigível do YOLOPv2 vivem no mesmo sistema de
   coordenadas, em metros;
2. dá para casar uma com a outra — empurrar a rota para dentro do corredor
   detectado, mantendo distância das bordas;
3. o ajuste de curva (spline) acontece num espaço euclidiano, então a
   suavização é isotrópica e não fica "esmagada" pela perspectiva.

Só depois a curva volta para a imagem original (módulo 6).

A IPM só é válida sob a hipótese de **solo plano**: subidas, lombadas e
buracos introduzem erro proporcional à distância.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 5.1 Grade métrica da vista superior
# ----------------------------------------------------------------------


@dataclass
class GradeBEV:
    """Define o recorte do solo mostrado na bird's-eye view.

    Convenção da imagem BEV: o topo é o ponto mais distante à frente e a
    esquerda da imagem é a esquerda do veículo — igual ao que o motorista
    veria num mapa alinhado ao carro.
    """

    x_min: float = 2.0  # início do recorte à frente (m)
    x_max: float = 40.0  # alcance (m)
    y_meia_largura: float = 12.0  # metade da largura lateral (m)
    px_por_m: float = 12.0

    @property
    def largura(self) -> int:
        return int(round(2 * self.y_meia_largura * self.px_por_m))

    @property
    def altura(self) -> int:
        return int(round((self.x_max - self.x_min) * self.px_por_m))

    @property
    def tamanho(self) -> tuple[int, int]:
        return (self.largura, self.altura)

    # --- transformações veículo <-> BEV --------------------------------
    @property
    def H_veiculo_para_bev(self) -> np.ndarray:
        """Homografia que leva (X, Y, 1) do solo para (u, v, 1) na BEV::

            u = (y_meia_largura - Y) · px_por_m
            v = (x_max          - X) · px_por_m
        """
        s = self.px_por_m
        return np.array(
            [
                [0.0, -s, s * self.y_meia_largura],
                [-s, 0.0, s * self.x_max],
                [0.0, 0.0, 1.0],
            ]
        )

    def para_bev(self, pontos_xy: np.ndarray) -> np.ndarray:
        """(N,2) em metros -> (N,2) em pixels da BEV."""
        p = np.asarray(pontos_xy, dtype=np.float64).reshape(-1, 2)
        u = (self.y_meia_largura - p[:, 1]) * self.px_por_m
        v = (self.x_max - p[:, 0]) * self.px_por_m
        return np.column_stack([u, v])

    def para_veiculo(self, pontos_uv: np.ndarray) -> np.ndarray:
        """(N,2) em pixels da BEV -> (N,2) em metros."""
        p = np.asarray(pontos_uv, dtype=np.float64).reshape(-1, 2)
        y = self.y_meia_largura - p[:, 0] / self.px_por_m
        x = self.x_max - p[:, 1] / self.px_por_m
        return np.column_stack([x, y])

    def x_da_linha(self, v: int) -> float:
        """Distância à frente (m) correspondente à linha ``v`` da BEV."""
        return self.x_max - v / self.px_por_m


# ----------------------------------------------------------------------
# 5.2 Warping
# ----------------------------------------------------------------------


def homografia_imagem_para_bev(H_solo_para_imagem: np.ndarray, grade: GradeBEV) -> np.ndarray:
    """Compõe imagem -> solo -> BEV.

    ``H_solo_para_imagem`` vem de ``calibration.homografia_solo_para_imagem``;
    invertê-la é literalmente o "mapeamento de perspectiva inversa".
    """
    return grade.H_veiculo_para_bev @ np.linalg.inv(H_solo_para_imagem)


def para_bev(
    imagem: np.ndarray,
    H_solo_para_imagem: np.ndarray,
    grade: GradeBEV,
    interpolacao: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Gera a vista superior de uma imagem (ou de uma máscara)."""
    H = homografia_imagem_para_bev(H_solo_para_imagem, grade)
    return cv2.warpPerspective(
        imagem, H, grade.tamanho, flags=interpolacao, borderValue=0
    )


def mascara_para_bev(
    mascara: np.ndarray, H_solo_para_imagem: np.ndarray, grade: GradeBEV
) -> np.ndarray:
    """Versão para máscaras binárias (vizinho mais próximo, sem borrar bordas)."""
    bev = para_bev(mascara, H_solo_para_imagem, grade, cv2.INTER_NEAREST)
    return (bev > 127).astype(np.uint8) * 255


# ----------------------------------------------------------------------
# 5.3 Corredor dirigível
# ----------------------------------------------------------------------


@dataclass
class Corredor:
    """Extensão lateral da área dirigível, linha a linha da BEV.

    Todos os vetores têm o mesmo comprimento e são indexados por ``x``
    (distância à frente, em metros, decrescente do topo para a base).
    """

    x: np.ndarray  # distância à frente (m)
    y_esq: np.ndarray  # borda esquerda (m, positivo à esquerda)
    y_dir: np.ndarray  # borda direita (m)
    valido: np.ndarray  # bool: havia via detectada nessa linha

    @property
    def y_centro(self) -> np.ndarray:
        return (self.y_esq + self.y_dir) / 2.0

    @property
    def largura(self) -> np.ndarray:
        return self.y_esq - self.y_dir

    @property
    def alcance_visivel(self) -> float:
        """Maior distância à frente com via detectada de forma contínua.

        É o limite natural da rota: além dele não há asfalto visível, e
        desenhar a faixa faria a linha "flutuar" sobre o horizonte.
        """
        if not self.valido.any():
            return 0.0
        # percorre do carro para longe e para no primeiro buraco relevante
        ordem = np.argsort(self.x)
        x_ord, val_ord = self.x[ordem], self.valido[ordem]
        limite = x_ord[0]
        falhas = 0
        for xi, vi in zip(x_ord, val_ord):
            if vi:
                limite, falhas = xi, 0
            else:
                falhas += 1
                if falhas > 3:
                    break
        return float(limite)


def extrair_corredor(
    mascara_bev: np.ndarray, grade: GradeBEV, largura_min_m: float = 1.5
) -> Corredor:
    """Mede, em cada linha da BEV, a faixa de asfalto contínua à frente do carro.

    Para cada linha considera-se apenas o segmento **conectado à linha de
    baixo** (onde o carro está), evitando que a via da mão contrária ou um
    trecho de calçada sejam confundidos com o corredor atual.
    """
    altura, largura = mascara_bev.shape[:2]
    m = mascara_bev > 0

    xs = np.array([grade.x_da_linha(v) for v in range(altura)])
    y_esq = np.full(altura, np.nan)
    y_dir = np.full(altura, np.nan)
    valido = np.zeros(altura, dtype=bool)

    # coluna de referência: começa no centro (posição do carro) e "sobe"
    col_ref = largura // 2

    for v in range(altura - 1, -1, -1):  # da base (perto) para o topo (longe)
        linha = m[v]
        if not linha.any():
            continue

        c = int(np.clip(col_ref, 0, largura - 1))
        if not linha[c]:
            # procura o pixel de via mais próximo da referência
            candidatos = np.flatnonzero(linha)
            c = int(candidatos[np.argmin(np.abs(candidatos - c))])
            if abs(c - col_ref) > 0.35 * largura:
                continue  # salto grande demais: provavelmente outra via

        esq = c
        while esq > 0 and linha[esq - 1]:
            esq -= 1
        dir_ = c
        while dir_ < largura - 1 and linha[dir_ + 1]:
            dir_ += 1

        y_e = grade.y_meia_largura - esq / grade.px_por_m
        y_d = grade.y_meia_largura - dir_ / grade.px_por_m

        if (y_e - y_d) < largura_min_m:
            continue

        y_esq[v], y_dir[v] = y_e, y_d
        valido[v] = True
        col_ref = (esq + dir_) // 2

    # preenche buracos curtos por interpolação
    if valido.any():
        idx = np.flatnonzero(valido)
        y_esq = np.interp(np.arange(altura), idx, y_esq[idx])
        y_dir = np.interp(np.arange(altura), idx, y_dir[idx])

    return Corredor(x=xs, y_esq=y_esq, y_dir=y_dir, valido=valido)


# ----------------------------------------------------------------------
# 5.4 Ajuste da rota GPS ao corredor
# ----------------------------------------------------------------------


def ajustar_rota_ao_corredor(
    rota_xy: np.ndarray,
    corredor: Corredor,
    margem_m: float = 0.8,
    peso_centro: float = 0.35,
    limitar_ao_visivel: bool = True,
) -> np.ndarray:
    """Concilia a rota do GPS com a via realmente vista pela câmera.

    Duas correções, nesta ordem:

    1. **Atração ao centro** — mistura o deslocamento lateral do GPS com o
       centro do corredor (``peso_centro``). Absorve o erro sistemático de
       poucos metros do GPS de celular, que costuma jogar a rota para cima
       da calçada.
    2. **Confinamento** — trunca o resultado para que fique a pelo menos
       ``margem_m`` das bordas detectadas. Se o corredor for mais estreito
       que ``2·margem_m``, cai para o centro.

    Este é o passo que materializa a premissa do projeto: *o espaço visível
    da rua faz parte do trajeto do GPS*.
    """
    rota = np.asarray(rota_xy, dtype=np.float64).reshape(-1, rota_xy.shape[-1])
    x, y = rota[:, 0].copy(), rota[:, 1].copy()

    ordem = np.argsort(corredor.x)
    xs = corredor.x[ordem]
    y_c = corredor.y_centro[ordem]
    y_e = corredor.y_esq[ordem]
    y_d = corredor.y_dir[ordem]

    finito = np.isfinite(y_c)
    if finito.sum() < 2:
        return rota  # sem via detectada: mantém o GPS puro

    xs, y_c, y_e, y_d = xs[finito], y_c[finito], y_e[finito], y_d[finito]

    centro = np.interp(x, xs, y_c)
    borda_e = np.interp(x, xs, y_e)
    borda_d = np.interp(x, xs, y_d)

    # 1) atração ao centro do corredor
    y_ajust = (1.0 - peso_centro) * y + peso_centro * centro

    # 2) confinamento com margem
    lim_e = borda_e - margem_m
    lim_d = borda_d + margem_m
    estreito = lim_e < lim_d
    y_ajust = np.where(estreito, centro, np.clip(y_ajust, lim_d, lim_e))

    saida = rota.copy()
    saida[:, 1] = y_ajust

    if limitar_ao_visivel:
        alcance = corredor.alcance_visivel
        saida = saida[saida[:, 0] <= max(alcance, 0.0)]

    return saida


def suavizar_rota(rota_xy: np.ndarray, suavidade: float = 2.0, n_saida: int = 200) -> np.ndarray:
    """Ajusta uma spline cúbica à rota já corrigida, no plano métrico da BEV.

    Suavizar aqui (e não em pixels) evita o artefato clássico de a curva
    parecer certinha perto do carro e serrilhada ao longe.
    """
    from scipy.interpolate import splev, splprep

    rota = np.asarray(rota_xy, dtype=np.float64)
    if len(rota) < 4:
        return rota

    try:
        tck, _ = splprep([rota[:, 0], rota[:, 1]], s=suavidade)
        u = np.linspace(0, 1, n_saida)
        x, y = splev(u, tck)
        saida = np.column_stack([x, y])
        if rota.shape[1] > 2:
            z = np.interp(
                np.linspace(0, 1, n_saida), np.linspace(0, 1, len(rota)), rota[:, 2]
            )
            saida = np.column_stack([saida, z])
        return saida
    except Exception:
        return rota


# ----------------------------------------------------------------------
# 5.5 Visualização
# ----------------------------------------------------------------------


def desenhar_bev(
    bev: np.ndarray,
    grade: GradeBEV,
    rota_gps: np.ndarray | None = None,
    rota_ajustada: np.ndarray | None = None,
    corredor: Corredor | None = None,
) -> np.ndarray:
    """Painel de conferência da BEV com grade métrica, corredor e rotas."""
    out = bev.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # grade de 5 em 5 metros
    for d in np.arange(np.ceil(grade.x_min / 5) * 5, grade.x_max + 1, 5):
        v = int((grade.x_max - d) * grade.px_por_m)
        cv2.line(out, (0, v), (out.shape[1], v), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(
            out, f"{int(d)}m", (5, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (150, 150, 150), 1, cv2.LINE_AA,
        )
    for dy in np.arange(-grade.y_meia_largura, grade.y_meia_largura + 1, 5):
        u = int((grade.y_meia_largura - dy) * grade.px_por_m)
        cv2.line(out, (u, 0), (u, out.shape[0]), (60, 60, 60), 1, cv2.LINE_AA)

    if corredor is not None:
        for y_borda, cor in ((corredor.y_esq, (0, 200, 255)), (corredor.y_dir, (0, 200, 255))):
            pts = [
                grade.para_bev(np.array([[xi, yi]]))[0]
                for xi, yi, ok in zip(corredor.x, y_borda, corredor.valido)
                if ok and np.isfinite(yi)
            ]
            if len(pts) > 1:
                cv2.polylines(
                    out, [np.int32(pts).reshape(-1, 1, 2)], False, cor, 1, cv2.LINE_AA
                )

    for rota, cor, esp in ((rota_gps, (255, 120, 0), 2), (rota_ajustada, (0, 255, 0), 3)):
        if rota is None or len(rota) < 2:
            continue
        pts = grade.para_bev(np.asarray(rota)[:, :2])
        cv2.polylines(
            out, [np.int32(pts).reshape(-1, 1, 2)], False, cor, esp, cv2.LINE_AA
        )

    # posição do veículo (X=0 pode estar fora do recorte)
    car = grade.para_bev(np.array([[max(grade.x_min, 0.0), 0.0]]))[0]
    cv2.circle(out, (int(car[0]), int(car[1])), 6, (0, 0, 255), -1, cv2.LINE_AA)
    return out
