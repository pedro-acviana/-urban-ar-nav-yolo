"""
Módulo 2a — Vídeo
=================

Acesso ao vídeo gravado pelo celular: metadados e extração de frames.
Isolado num módulo próprio porque o resto do pipeline só precisa de
``InfoVideo`` (fps, resolução) e de um array BGR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class InfoVideo:
    caminho: Path
    fps: float
    n_frames: int
    largura: int
    altura: int

    @property
    def duracao_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def resolucao(self) -> tuple[int, int]:
        """(largura, altura) em pixels."""
        return (self.largura, self.altura)

    def tempo_do_frame(self, indice: int) -> float:
        """Instante (s desde o início do vídeo) em que o frame foi capturado."""
        return indice / self.fps

    def frame_do_tempo(self, t: float) -> int:
        return int(round(t * self.fps))

    def __str__(self) -> str:
        return (
            f"{self.caminho.name}: {self.largura}x{self.altura} @ {self.fps:.2f} fps | "
            f"{self.n_frames} frames | {self.duracao_s:.1f} s"
        )


def info(caminho: str | Path) -> InfoVideo:
    """Lê os metadados do vídeo sem decodificar o conteúdo."""
    caminho = Path(caminho)
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho}")
    try:
        return InfoVideo(
            caminho=caminho,
            fps=cap.get(cv2.CAP_PROP_FPS),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            largura=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            altura=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def extrair_frame(caminho: str | Path, indice: int) -> np.ndarray:
    """Devolve um frame (BGR) pelo índice.

    Usa busca por índice; em contêineres com GOP longo o decodificador pode
    devolver o keyframe mais próximo, por isso a posição efetiva é conferida.
    """
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Falha ao ler o frame {indice} de {caminho}")
        return frame
    finally:
        cap.release()


def extrair_varios(caminho: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Extrai vários frames numa única passagem (leitura sequencial)."""
    indices = sorted(set(int(i) for i in indices))
    restantes = set(indices)
    saida: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho}")
    try:
        atual = 0
        while restantes:
            ok, frame = cap.read()
            if not ok:
                break
            if atual in restantes:
                saida[atual] = frame
                restantes.discard(atual)
            atual += 1
    finally:
        cap.release()
    return saida
