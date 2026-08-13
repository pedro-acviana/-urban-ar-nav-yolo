"""
Módulo 2a — Vídeo
=================

Acesso às imagens do percurso: metadados e leitura de frames.
Isolado num módulo próprio porque o resto do pipeline só precisa de
``InfoVideo`` (fps, resolução) e de um array BGR.

Duas fontes possíveis, com a mesma interface (:class:`Fonte`):

``frames``
    Uma pasta ``data/frames/<percurso>/`` com PNGs avulsos e um ``info.json``
    guardando os metadados que antes vinham do contêiner. É a fonte
    **principal**: os `.mp4` brutos têm centenas de MB, ficam fora do Git e
    ninguém além de quem gravou os tem, enquanto alguns frames versionados
    bastam para rodar e reproduzir o pipeline.

``video``
    O `.mp4` original. Continua sendo a fonte de onde os frames saem
    (:func:`exportar_frames`) e o único jeito de alcançar um frame que ainda
    não foi exportado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

NOME_METADADOS = "info.json"
"""Arquivo com fps/resolução dentro da pasta de frames."""


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
    """Extrai vários frames numa única passagem (leitura sequencial).

    ATENÇÃO — em vídeo de taxa variável (é o caso dos `.mp4` de celular deste
    projeto) o *n*-ésimo frame decodificado **não** é o mesmo que
    :func:`extrair_frame` devolve para o índice *n*: aqui conta-se frame a
    frame, lá o seek é por tempo (``n/fps``). Em ``volta_menor`` a diferença
    chega a 1,7 s no índice 692 — o suficiente para invalidar o offset de
    sincronização. O resto do pipeline usa a convenção de :func:`extrair_frame`;
    esta função só serve para varreduras em que a identidade do índice não
    importa.
    """
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


# ----------------------------------------------------------------------
# Frames exportados
# ----------------------------------------------------------------------
def caminho_frame(dir_frames: str | Path, indice: int) -> Path:
    """Caminho do PNG de um frame. O zero à esquerda mantém a ordem no `ls`."""
    return Path(dir_frames) / f"frame{int(indice):06d}.png"


def frames_disponiveis(dir_frames: str | Path) -> list[int]:
    """Índices já exportados, lidos do nome dos arquivos."""
    dir_frames = Path(dir_frames)
    if not dir_frames.is_dir():
        return []
    indices = []
    for p in dir_frames.glob("frame*.png"):
        try:
            indices.append(int(p.stem.removeprefix("frame")))
        except ValueError:
            continue
    return sorted(indices)


def info_de_frames(dir_frames: str | Path) -> InfoVideo:
    """Reconstrói ``InfoVideo`` a partir do ``info.json`` da pasta de frames.

    Os campos são os do vídeo de origem, não os da pasta: ``fps`` e
    ``n_frames`` descrevem a gravação inteira, e é deles que a sincronização
    com o GPX depende (``tempo_do_frame = indice / fps``). Guardá-los aqui é o
    que permite descartar o `.mp4` sem perder o alinhamento temporal.
    """
    dir_frames = Path(dir_frames)
    meta = dir_frames / NOME_METADADOS
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} não existe — exporte os frames com "
            f"`scripts/exportar_frames.py` antes de usar a fonte 'frames'."
        )
    dados = json.loads(meta.read_text(encoding="utf-8"))
    return InfoVideo(
        caminho=dir_frames,
        fps=float(dados["fps"]),
        n_frames=int(dados["n_frames"]),
        largura=int(dados["largura"]),
        altura=int(dados["altura"]),
    )


def carregar_frame(dir_frames: str | Path, indice: int) -> np.ndarray:
    """Lê um frame exportado (BGR), como se tivesse saído do vídeo."""
    caminho = caminho_frame(dir_frames, indice)
    if not caminho.exists():
        disponiveis = frames_disponiveis(dir_frames)
        raise FileNotFoundError(
            f"Frame {indice} não foi exportado ({caminho}).\n"
            f"Disponíveis: {disponiveis or 'nenhum'}.\n"
            f"Para exportá-lo é preciso o vídeo original: "
            f"`python scripts/exportar_frames.py <percurso> --frames {indice}`"
        )
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Falha ao ler o frame: {caminho}")
    return img


def exportar_frames(
    caminho_video: str | Path,
    dir_frames: str | Path,
    indices: list[int],
) -> dict[int, Path]:
    """Extrai ``indices`` do vídeo para PNG e grava o ``info.json``.

    Usa :func:`extrair_frame` (seek por índice), e **não**
    :func:`extrair_varios`, mesmo custando uma busca por frame: os `.mp4` são
    de taxa variável, e a leitura sequencial devolveria imagens de outro
    instante — no ``volta_menor`` o índice 692 sai 1,7 s adiantado. Como o
    offset de sincronização foi calibrado sobre a convenção do seek, exportar
    pelo caminho sequencial trocaria a imagem sob uma calibração que continua
    apontando para o instante antigo.

    PNG, e não JPEG: a segmentação roda sobre esses pixels, e artefato de
    compressão numa borda de faixa é exatamente o tipo de ruído que a máscara
    do YOLOPv2 propaga para a rota.
    """
    caminho_video = Path(caminho_video)
    dir_frames = Path(dir_frames)
    dir_frames.mkdir(parents=True, exist_ok=True)

    meta = info(caminho_video)
    fora = [i for i in indices if not 0 <= i < meta.n_frames]
    if fora:
        raise ValueError(
            f"Índices fora do vídeo: {fora} "
            f"({caminho_video.name} tem {meta.n_frames} frames)."
        )

    salvos: dict[int, Path] = {}
    for indice in sorted(set(int(i) for i in indices)):
        img = extrair_frame(caminho_video, indice)
        destino = caminho_frame(dir_frames, indice)
        if not cv2.imwrite(str(destino), img):
            raise RuntimeError(f"Falha ao gravar {destino}")
        salvos[indice] = destino

    (dir_frames / NOME_METADADOS).write_text(
        json.dumps(
            {
                "video": caminho_video.name,
                "fps": meta.fps,
                "n_frames": meta.n_frames,
                "largura": meta.largura,
                "altura": meta.altura,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return salvos


# ----------------------------------------------------------------------
# Fonte de imagens (frames ou vídeo)
# ----------------------------------------------------------------------
@dataclass
class Fonte:
    """De onde o pipeline tira os frames, sem que ele precise saber qual é."""

    tipo: str
    """``"frames"`` ou ``"video"``."""

    caminho: Path
    info: InfoVideo

    def frame(self, indice: int) -> np.ndarray:
        if self.tipo == "frames":
            return carregar_frame(self.caminho, indice)
        return extrair_frame(self.caminho, indice)

    def __str__(self) -> str:
        if self.tipo == "frames":
            disponiveis = frames_disponiveis(self.caminho)
            return (
                f"frames exportados ({len(disponiveis)}): {self.caminho} | "
                f"{self.info.largura}x{self.info.altura} @ {self.info.fps:.2f} fps"
            )
        return f"vídeo: {self.info}"


def abrir(
    dir_frames: str | Path,
    caminho_video: str | Path | None = None,
    preferencia: str = "auto",
) -> Fonte:
    """Escolhe a fonte de imagens.

    ``preferencia``:
        ``"frames"``  exige a pasta de frames;
        ``"video"``   exige o `.mp4`;
        ``"auto"``    usa os frames quando existirem, senão o vídeo.
    """
    dir_frames = Path(dir_frames)
    caminho_video = Path(caminho_video) if caminho_video else None
    tem_frames = (dir_frames / NOME_METADADOS).exists()
    tem_video = caminho_video is not None and caminho_video.exists()

    if preferencia not in {"auto", "frames", "video"}:
        raise ValueError(
            f"fonte_frames inválida: {preferencia!r} "
            f"(use 'auto', 'frames' ou 'video')"
        )

    if preferencia == "frames" or (preferencia == "auto" and tem_frames):
        if not tem_frames:
            raise FileNotFoundError(
                f"Fonte 'frames' pedida, mas {dir_frames / NOME_METADADOS} não existe."
            )
        return Fonte("frames", dir_frames, info_de_frames(dir_frames))

    if not tem_video:
        raise FileNotFoundError(
            "Nenhuma fonte de imagens disponível.\n"
            f"  frames ... {dir_frames} (sem {NOME_METADADOS})\n"
            f"  vídeo .... {caminho_video or '(não informado)'}\n"
            "Os `.mp4` não são versionados; veja o README."
        )
    return Fonte("video", caminho_video, info(caminho_video))
