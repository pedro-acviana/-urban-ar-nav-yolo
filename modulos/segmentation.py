"""
Módulo 4 — Segmentação da via com YOLOPv2
=========================================

Extrai do frame três informações complementares:

* **área dirigível** (``da_seg``) — o asfalto por onde é possível trafegar;
* **faixas** (``ll_seg``) — marcações de solo, usadas como referência lateral;
* **obstáculos** (detecção) — caixas de veículos, para truncar a rota por
  oclusão (Passo 4 do plano).

O modelo é um TorchScript (``yolopv2.pt``): não precisa da definição da rede
em Python, apenas de ``torch``. As funções auxiliares do repositório YOLOPv2
são usadas quando disponíveis, com implementação local de reserva — assim o
módulo funciona mesmo se o submódulo não estiver inicializado.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 4.1 Pré-processamento (letterbox)
# ----------------------------------------------------------------------


def letterbox(
    img: np.ndarray, tamanho: int = 640, stride: int = 32, cor=(114, 114, 114)
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Redimensiona preservando a proporção e completa com bordas cinza.

    Devolve ``(imagem, razao, (pad_x, pad_y))``. Guardar o padding é essencial:
    é ele que permite devolver a máscara ao enquadramento original sem
    esticar a cena — um erro que desalinharia a máscara do asfalto.
    """
    h, w = img.shape[:2]
    r = min(tamanho / h, tamanho / w)
    novo_w, novo_h = int(round(w * r)), int(round(h * r))

    dw, dh = tamanho - novo_w, tamanho - novo_h
    dw, dh = dw % stride, dh % stride  # padding mínimo múltiplo do stride
    dw, dh = dw / 2, dh / 2

    if (w, h) != (novo_w, novo_h):
        img = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)

    topo, base = int(round(dh - 0.1)), int(round(dh + 0.1))
    esq, dir_ = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, topo, base, esq, dir_, cv2.BORDER_CONSTANT, value=cor
    )
    return img, r, (dw, dh)


def _remover_padding(
    mascara: np.ndarray, razao: float, pad: tuple[float, float], forma_alvo: tuple[int, int]
) -> np.ndarray:
    """Desfaz o letterbox: recorta o padding e volta à resolução original."""
    h, w = mascara.shape[:2]
    dw, dh = pad
    topo, esq = int(round(dh - 0.1)), int(round(dw - 0.1))
    base, dir_ = h - int(round(dh + 0.1)), w - int(round(dw + 0.1))
    recorte = mascara[topo:base, esq:dir_]
    return cv2.resize(
        recorte, (forma_alvo[1], forma_alvo[0]), interpolation=cv2.INTER_NEAREST
    )


# ----------------------------------------------------------------------
# 4.2 Modelo
# ----------------------------------------------------------------------


@dataclass
class ModeloYolop:
    modelo: object
    device: object
    meia_precisao: bool = False


def carregar_modelo(
    pesos: str | Path, repo_yolop: str | Path | None = None, device: str | None = None
) -> ModeloYolop:
    """Carrega o TorchScript do YOLOPv2 e o coloca em modo de avaliação."""
    import torch

    pesos = Path(pesos)
    if not pesos.exists():
        raise FileNotFoundError(
            f"Pesos não encontrados: {pesos}\n"
            "Baixe em https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        )

    if repo_yolop is not None:
        repo_yolop = str(Path(repo_yolop).resolve())
        if repo_yolop not in sys.path:
            sys.path.insert(0, repo_yolop)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    modelo = torch.jit.load(str(pesos), map_location=dev)
    modelo.eval().to(dev)

    meia = dev.type == "cuda"
    if meia:
        modelo.half()

    return ModeloYolop(modelo=modelo, device=dev, meia_precisao=meia)


# ----------------------------------------------------------------------
# 4.3 Inferência
# ----------------------------------------------------------------------


@dataclass
class ResultadoSegmentacao:
    area_dirigivel: np.ndarray  # uint8 0/255, resolução do frame
    faixas: np.ndarray  # uint8 0/255
    obstaculos: np.ndarray = field(  # Nx5: x1, y1, x2, y2, confiança
        default_factory=lambda: np.zeros((0, 5), dtype=np.float32)
    )

    @property
    def cobertura_pct(self) -> float:
        return float((self.area_dirigivel > 0).mean() * 100)

    def __str__(self) -> str:
        return (
            f"área dirigível: {self.cobertura_pct:.1f}% da imagem | "
            f"faixas: {(self.faixas > 0).mean() * 100:.2f}% | "
            f"obstáculos: {len(self.obstaculos)}"
        )


def segmentar(
    yolop: ModeloYolop,
    frame: np.ndarray,
    tamanho: int = 640,
    limiar_area: float = 0.50,
    limiar_faixas: float = 0.30,
    detectar_obstaculos: bool = True,
    conf_obstaculo: float = 0.35,
) -> ResultadoSegmentacao:
    """Roda o YOLOPv2 num frame BGR e devolve as máscaras na resolução original."""
    import torch

    forma_original = frame.shape[:2]
    img, razao, pad = letterbox(frame, tamanho)

    tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy())
    tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(yolop.device)
    if yolop.meia_precisao:
        tensor = tensor.half()

    with torch.no_grad():
        saida = yolop.modelo(tensor)

    det_bruta, seg, faixa = saida[0], saida[1], saida[2]

    # --- área dirigível: 2 canais (fundo, via) -> argmax ------------------
    area = torch.softmax(seg.float(), dim=1)[0, 1].cpu().numpy()

    # --- faixas: canal único com probabilidade ---------------------------
    faixa = faixa.float()
    if faixa.shape[1] > 1:
        faixa = torch.softmax(faixa, dim=1)[:, 1:2]
    faixa = faixa[0, 0].cpu().numpy()

    area = (_remover_padding(area, razao, pad, forma_original) > limiar_area)
    faixa = (_remover_padding(faixa, razao, pad, forma_original) > limiar_faixas)

    resultado = ResultadoSegmentacao(
        area_dirigivel=(area.astype(np.uint8) * 255),
        faixas=(faixa.astype(np.uint8) * 255),
    )

    if detectar_obstaculos:
        resultado.obstaculos = _detectar(
            det_bruta, img.shape[:2], forma_original, conf_obstaculo
        )
    return resultado


def _detectar(det_bruta, forma_entrada, forma_original, conf: float) -> np.ndarray:
    """Extrai caixas de obstáculos usando os utilitários do repositório YOLOPv2.

    Falha silenciosamente (devolve vazio) se o repositório não estiver
    disponível — a oclusão por obstáculo é opcional no pipeline.
    """
    try:
        import torch
        from utils.utils import (  # type: ignore
            non_max_suppression,
            scale_coords,
            split_for_trace_model,
        )

        pred, grade = det_bruta
        pred = split_for_trace_model(pred, grade)
        pred = non_max_suppression(pred, conf_thres=conf, iou_thres=0.45)

        caixas = []
        for det in pred:
            if det is None or not len(det):
                continue
            det = det.clone()
            det[:, :4] = scale_coords(
                forma_entrada, det[:, :4], forma_original
            ).round()
            caixas.append(det[:, :5].cpu().numpy())

        if not caixas:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate(caixas).astype(np.float32)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32)


# ----------------------------------------------------------------------
# 4.4 Pós-processamento das máscaras
# ----------------------------------------------------------------------


def limpar_mascara(
    mascara: np.ndarray,
    fechamento: int = 15,
    abertura: int = 5,
    manter_maior: bool = True,
) -> np.ndarray:
    """Fecha buracos, remove ruído e (opcionalmente) mantém só o maior blob.

    A rota do carro percorre uma superfície contínua; componentes soltas
    (calçadas, o asfalto da via transversal) só atrapalham o ajuste lateral.
    """
    m = (mascara > 0).astype(np.uint8) * 255

    if fechamento > 0:
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE, np.ones((fechamento, fechamento), np.uint8)
        )
    if abertura > 0:
        m = cv2.morphologyEx(
            m, cv2.MORPH_OPEN, np.ones((abertura, abertura), np.uint8)
        )

    if manter_maior:
        n, rotulos, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.where(rotulos == maior, 255, 0).astype(np.uint8)

    return cv2.medianBlur(m, 5)


def sobrepor(
    frame: np.ndarray,
    resultado: ResultadoSegmentacao,
    cor_area=(0, 180, 0),
    cor_faixa=(255, 0, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    """Visualização de conferência: máscaras coloridas sobre o frame."""
    out = frame.copy()
    area = resultado.area_dirigivel > 0
    faixa = resultado.faixas > 0

    out[area] = ((1 - alpha) * out[area] + alpha * np.array(cor_area)).astype(np.uint8)
    out[faixa] = (0.5 * out[faixa] + 0.5 * np.array(cor_faixa)).astype(np.uint8)

    for x1, y1, x2, y2, c in resultado.obstaculos:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"{c:.2f}",
            (int(x1), max(int(y1) - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return out
