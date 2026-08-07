"""
Pipeline de navegação AR — projeção da rota GPS sobre a imagem do motorista.

Módulos sequenciais (execute nesta ordem):

    1. config        Parâmetros e caminhos do experimento
    2. gps           Leitura do GPX, ENU, suavização, heading
    3. video         Leitura do vídeo e extração de frames
    4. sync          Sincronização GPS <-> frame e rota futura no referencial do veículo
    5. calibration   Intrínsecos (K, dist) + extrínsecos manuais (altura, pitch, yaw)
    6. segmentation  YOLOPv2: máscara de área dirigível e de faixas
    7. ipm           Bird's-eye view, corredor dirigível e ajuste da rota
    8. projection    Projeção pinhole 3D -> 2D, spline e oclusão
    9. render        Desenho da faixa AR, minimapa e composição final
   10. pipeline      Orquestração ponta a ponta
"""

__version__ = "0.1.0"

from .config import Config  # noqa: F401

__all__ = [
    "Config",
    "config",
    "gps",
    "video",
    "sync",
    "calibration",
    "segmentation",
    "ipm",
    "projection",
    "render",
    "pipeline",
]

# Os submódulos são importados sob demanda (``from modulos import gps``) para
# que uma dependência pesada ausente — torch, por exemplo — não impeça o uso
# das etapas anteriores do pipeline.
