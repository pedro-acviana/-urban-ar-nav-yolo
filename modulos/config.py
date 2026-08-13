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
    # Fonte das imagens
    # ------------------------------------------------------------------
    fonte_frames: str = "auto"
    """De onde vêm os frames: ``"frames"``, ``"video"`` ou ``"auto"``.

    Os frames exportados são a fonte **principal**: os `.mp4` brutos passam de
    400 MB, ficam fora do Git e só existem na máquina de quem gravou, então um
    clone do repositório não conseguia rodar nada. Uma pasta de frames
    versionada resolve isso — e como ``info.json`` guarda fps e contagem total
    do vídeo original, a sincronização com o GPX continua exata.

    ``"auto"`` usa os frames quando a pasta existe e cai no `.mp4` quando não.
    Use ``"video"`` para alcançar um frame que ainda não foi exportado.
    """

    # ------------------------------------------------------------------
    # Sincronização GPS <-> vídeo
    # ------------------------------------------------------------------
    frame_alvo: int = 692
    """Índice do frame a ser processado, contado no vídeo original.

    Continua sendo o índice no `.mp4` mesmo quando a fonte são os frames
    exportados — é ele que amarra a imagem ao instante do GPX.
    """

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
    ajustar_ao_corredor: bool = False
    """Se True, empurra a rota para o centro da via e a trunca pela faixa visível.

    Padrão False (V2): confia na rota crua do GPS e deixa **só a máscara de
    segmentação** decidir onde ela aparece, via recorte na renderização. É o
    comportamento mais previsível — o ajuste ao corredor introduz três
    truncamentos em série que encurtam o desenho sem que fique claro o motivo,
    e o rastreio lateral do corredor é frágil em cruzamentos.

    Ligue quando quiser corrigir o erro sistemático do GPS (rota subindo na
    calçada) e o corredor estiver bem definido — via reta e sem bifurcação.
    """

    exigir_via_no_filtro: bool = False
    """Se True, trunca a rota no ponto em que ela sai da área dirigível.

    Padrão False: o recorte visual já é feito na renderização, então truncar
    aqui só encurta a geometria sem ganho de imagem.
    """

    margem_borda_m: float = 0.8
    """Distância mínima entre a rota e a borda da área dirigível.
    Só tem efeito com ``ajustar_ao_corredor=True``."""

    peso_centro: float = 0.35
    """0 = confia só no GPS; 1 = cola no centro da via detectada.
    Só tem efeito com ``ajustar_ao_corredor=True``."""

    largura_faixa_ar_m: float = 1.8
    """Largura da fita de realidade aumentada desenhada sobre o asfalto."""

    tolerancia_fora_via_m: float = 3.0
    """Trecho fora da via (m) perdoado antes de truncar a rota.

    Com 0, um único ponto raspando o meio-fio ou a ilha central de um
    cruzamento encerra o desenho. Em metros, e não em amostras, para não
    depender da densidade da reamostragem.
    """

    folga_tela_px: float = 600.0
    """Folga além da borda da imagem antes de considerar um ponto perdido.

    Sair do quadro não encerra a rota — o rasterizador recorta na borda. Com
    folga pequena, a fita termina visivelmente antes da margem, o que salta aos
    olhos em vídeo de celular em retrato (FOV horizontal de ~40°).
    """

    salto_max_corredor: float = 0.35
    """Deslocamento lateral máximo entre linhas no rastreio do corredor.

    Fração da largura da BEV. Aumente em curvas fechadas e cruzamentos, onde
    a via anda depressa para o lado e o rastreio desiste cedo demais.
    """

    def __post_init__(self) -> None:
        self.raiz = Path(self.raiz)
        if self.offset_sync_s is None:
            self.offset_sync_s = OFFSETS_CALIBRADOS.get(self.nome_percurso, 0.0)
        if self.saida is None:
            self.saida = self.raiz / "output" / self.nome_percurso
        if self.pesos_yolop is None:
            # Fora do submódulo de propósito: dentro de YOLOPv2/ os 156 MB de
            # pesos seriam apagados por um `git submodule update --force` ou
            # por qualquer reclone do submódulo.
            self.pesos_yolop = self.raiz / "pesos" / "yolopv2.pt"
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
    def dir_frames(self) -> Path:
        """Pasta dos frames exportados — a fonte principal de imagens."""
        return self.raiz / "data" / "frames" / self.nome_percurso

    @property
    def video(self) -> Path:
        """`.mp4` bruto. Não versionado: só existe para exportar frames novos."""
        return self.raiz / "data" / f"{self.nome_percurso}.mp4"

    def abrir_fonte(self):
        """Devolve a fonte de imagens já resolvida (``video.Fonte``)."""
        from . import video

        return video.abrir(self.dir_frames, self.video, self.fonte_frames)

    def frames_exportados(self) -> list[int]:
        from . import video

        return video.frames_disponiveis(self.dir_frames)

    def checar_arquivos(self) -> dict[str, bool]:
        """Verifica a existência dos insumos. Útil como primeira célula.

        ``video`` fica de fora: com os frames exportados o `.mp4` deixou de
        ser insumo obrigatório, e listá-lo como ausente assustava sem motivo
        quem tinha clonado o repositório.
        """
        from . import video as _video

        itens = {
            "gpx": self.gpx,
            "frames": self.dir_frames / _video.NOME_METADADOS,
            "calibracao": self.arquivo_calibracao,
            "pesos_yolop": self.pesos_yolop,
            "repo_yolop": self.repo_yolop,
        }
        estado = {k: Path(v).exists() for k, v in itens.items()}
        estado["video (opcional)"] = self.video.exists()
        return estado

    def resumo(self) -> str:
        linhas = [
            "Configuração do experimento",
            f"  percurso ............ {self.nome_percurso}",
            f"  fonte de imagens .... {self.fonte_frames}"
            + (
                f"  ({len(exportados)} frames exportados)"
                if (exportados := self.frames_exportados())
                else "  (nenhum frame exportado)"
            ),
            f"  frame alvo .......... {self.frame_alvo}"
            + ("" if self.frame_alvo in exportados else "  (NÃO exportado)"),
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
