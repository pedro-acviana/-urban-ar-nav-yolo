"""
Módulo 0 — Configuração

Centraliza caminhos e parâmetros do experimento. Todo o resto do pipeline
recebe um objeto ``Config``, de modo que nenhum módulo precise conhecer a
estrutura de pastas do projeto.

Convenções de eixos usadas em TODO o pipeline
---------------------------------------------
Referencial do VEÍCULO (origem no chão, sob o centro óptico da câmera):

    X -> frente        Y -> esquerda        Z -> cima

Referencial da CÂMERA (padrão OpenCV):

    X -> direita       Y -> baixo           Z -> frente (eixo óptico)

Referencial ENU (local, origem no primeiro ponto do GPX):

    east -> leste      north -> norte       up -> cima

Heading: 0 rad = Norte, crescendo no sentido horário (bússola).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _raiz_padrao() -> Path:
    """Retorna a raiz do projeto (pasta que contém ``modulos/``)."""
    return Path(__file__).resolve().parent.parent


# Offsets já calibrados, em segundos, por percurso.
#
# Obtidos com `render.painel_offsets`: o valor correto é aquele em que a rota
# projetada acompanha o asfalto visível. Registre aqui cada percurso novo em
# vez de repetir o número solto pelos notebooks.
OFFSETS_CALIBRADOS: dict[str, float] = {
    # frame 692: a rota dobra à direita junto com a via, e o GPS concorda —
    # 12 km/h e heading girando de 342° para 310°, o carro entrando na curva.
    "volta_menor": 2.0,
}


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Caminhos
    # ------------------------------------------------------------------
    raiz: Path = field(default_factory=_raiz_padrao)
    nome_percurso: str = "volta_menor"
    saida: Path | None = None
    pesos_yolop: Path | None = None
    repo_yolop: Path | None = None
    arquivo_calibracao: Path | None = None

    # ------------------------------------------------------------------
    # Sincronização GPS <-> vídeo
    # ------------------------------------------------------------------
    frame_alvo: int = 692
    """Índice do frame do vídeo a ser processado."""

    offset_sync_s: float | None = None
    """Deslocamento temporal (s) entre o relógio do GPX e o t=0 do vídeo.

    ``t_video = (t_gpx - t_gpx[0]) + offset_sync_s``
    Valor negativo = o GPX começou a gravar ANTES do vídeo.

    Deixando ``None``, usa o valor calibrado para o percurso
    (``OFFSETS_CALIBRADOS``); se o percurso não estiver na tabela, cai em 0.0
    e o offset precisa ser calibrado — veja ``render.painel_offsets``.
    """

    horizonte_s: float = 12.0
    """Janela de tempo à frente usada para montar a rota futura."""

    distancia_max_m: float = 40.0
    """Alcance máximo (m) da rota projetada."""

    # ------------------------------------------------------------------
    # Tratamento do GPS
    # ------------------------------------------------------------------
    janela_savgol: int = 9
    """Janela (ímpar, em amostras) do filtro Savitzky-Golay aplicado ao ENU."""

    ordem_savgol: int = 2
    usar_elevacao: bool = False
    """Se True, usa a elevação do GPX como Z dos waypoints.
    Padrão False: a elevação de GPS de celular é ruidosa demais e o
    modelo de solo plano (Z=0) dá resultados melhores."""

    # ------------------------------------------------------------------
    # Pose da câmera no veículo (extrínsecos — ajuste manual)
    # ------------------------------------------------------------------
    altura_camera_m: float = 1.50
    """Altura do centro óptico em relação ao asfalto."""

    pitch_deg: float = -10.0
    """Inclinação vertical. NEGATIVO = câmera apontando para BAIXO."""

    yaw_deg: float = 0.0
    """Rotação horizontal. POSITIVO = câmera girada para a ESQUERDA."""

    roll_deg: float = 0.0
    """Rotação em torno do eixo óptico. POSITIVO = horizonte cai à direita."""

    # ------------------------------------------------------------------
    # Segmentação (YOLOPv2)
    # ------------------------------------------------------------------
    limiar_area_dirigivel: float = 0.50
    limiar_faixas: float = 0.30
    tamanho_inferencia: int = 640

    # ------------------------------------------------------------------
    # Bird's-eye view (IPM)
    # ------------------------------------------------------------------
    bev_x_min: float = 2.0
    bev_x_max: float = 40.0
    bev_y_meia_largura: float = 12.0
    bev_px_por_m: float = 12.0

    # ------------------------------------------------------------------
    # Ajuste da rota ao corredor dirigível
    # ------------------------------------------------------------------
    margem_borda_m: float = 0.8
    """Distância mínima entre a rota e a borda da área dirigível."""

    peso_centro: float = 0.35
    """0 = confia só no GPS; 1 = cola no centro da via detectada."""

    largura_faixa_ar_m: float = 1.8
    """Largura da fita de realidade aumentada desenhada sobre o asfalto."""

    def __post_init__(self) -> None:
        self.raiz = Path(self.raiz)
        if self.offset_sync_s is None:
            self.offset_sync_s = OFFSETS_CALIBRADOS.get(self.nome_percurso, 0.0)
        if self.saida is None:
            self.saida = self.raiz / "output" / self.nome_percurso
        if self.pesos_yolop is None:
            self.pesos_yolop = self.raiz / "YOLOPv2" / "data" / "weights" / "yolopv2.pt"
        if self.repo_yolop is None:
            self.repo_yolop = self.raiz / "YOLOPv2"
        if self.arquivo_calibracao is None:
            self.arquivo_calibracao = (
                self.raiz / "camera_calibration" / "calibracao_camera.json"
            )
        self.saida = Path(self.saida)
        self.saida.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def gpx(self) -> Path:
        return self.raiz / "data" / f"{self.nome_percurso}.gpx"

    @property
    def video(self) -> Path:
        return self.raiz / "data" / f"{self.nome_percurso}.mp4"

    def checar_arquivos(self) -> dict[str, bool]:
        """Verifica a existência dos insumos. Útil como primeira célula."""
        itens = {
            "gpx": self.gpx,
            "video": self.video,
            "calibracao": self.arquivo_calibracao,
            "pesos_yolop": self.pesos_yolop,
            "repo_yolop": self.repo_yolop,
        }
        return {k: Path(v).exists() for k, v in itens.items()}

    def resumo(self) -> str:
        linhas = [
            "Configuração do experimento",
            f"  percurso ............ {self.nome_percurso}",
            f"  frame alvo .......... {self.frame_alvo}",
            f"  offset de sync ...... {self.offset_sync_s:+.2f} s"
            + ("  (calibrado)" if self.nome_percurso in OFFSETS_CALIBRADOS else "  (NÃO calibrado)"),
            f"  horizonte ........... {self.horizonte_s:.1f} s / {self.distancia_max_m:.0f} m",
            f"  altura da câmera .... {self.altura_camera_m:.2f} m",
            f"  pitch / yaw / roll .. {self.pitch_deg:+.1f}° / {self.yaw_deg:+.1f}° / {self.roll_deg:+.1f}°",
            f"  BEV ................. x[{self.bev_x_min:.0f}, {self.bev_x_max:.0f}] m, "
            f"y ±{self.bev_y_meia_largura:.0f} m, {self.bev_px_por_m:.0f} px/m",
            f"  saída ............... {self.saida}",
        ]
        return "\n".join(linhas)
