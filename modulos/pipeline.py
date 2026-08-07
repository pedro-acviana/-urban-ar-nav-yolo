"""
Módulo 7 — Orquestração ponta a ponta
=====================================

Encadeia os módulos 1 a 6 num único comando. O notebook executa os mesmos
passos separadamente (para inspecionar cada etapa); esta função existe para
reprodução em lote e para processar vários frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from . import calibration, gps, ipm, projection, render, segmentation, sync, video
from .config import Config


@dataclass
class Resultado:
    config: Config
    trilha: pd.DataFrame
    info_video: video.InfoVideo
    estado: sync.EstadoVeiculo
    rota_enu: pd.DataFrame
    rota_veiculo: np.ndarray
    intr: calibration.Intrinsecos
    pose: calibration.PoseCamera
    frame: np.ndarray
    frame_corrigido: np.ndarray
    seg: segmentation.ResultadoSegmentacao | None
    grade: ipm.GradeBEV
    bev: np.ndarray | None
    corredor: ipm.Corredor | None
    rota_ajustada: np.ndarray
    rota_final: np.ndarray
    imagem_final: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def salvar(self, pasta: str | Path | None = None) -> dict[str, Path]:
        pasta = Path(pasta or self.config.saida)
        pasta.mkdir(parents=True, exist_ok=True)
        sufixo = f"frame{self.config.frame_alvo:06d}"
        salvos: dict[str, Path] = {}

        def _grava(nome: str, img):
            if img is None:
                return
            caminho = pasta / f"{sufixo}_{nome}.png"
            cv2.imwrite(str(caminho), img)
            salvos[nome] = caminho

        _grava("original", self.frame)
        _grava("corrigido", self.frame_corrigido)
        if self.seg is not None:
            _grava("segmentacao", segmentation.sobrepor(self.frame_corrigido, self.seg))
        _grava("bev", self.extras.get("bev_anotada"))
        _grava("final", self.imagem_final)
        return salvos

    def diagnostico(self) -> str:
        """Mostra quanto da rota sobrevive a cada estágio do pipeline.

        O painel de varredura de offset desenha a rota **crua** do GPS, sem
        nenhum corte; o resultado final passa por três truncamentos em série.
        Quando o desenho final sai bem mais curto que o do painel, é aqui que
        se vê qual estágio é o responsável.
        """
        def _extensao(pts) -> str:
            if pts is None or len(pts) == 0:
                return "vazia"
            return f"{pts[:, 0].min():5.1f} -> {pts[:, 0].max():5.1f} m"

        linhas = [
            "Sobrevivência da rota, estágio a estágio",
            f"  1. GPS cru (como no painel de offset) ... "
            f"{len(self.rota_veiculo):4d} pts  {_extensao(self.rota_veiculo)}",
        ]

        if self.corredor is not None:
            ini, fim = self.corredor.faixa_visivel
            estado = "aplicado" if self.config.ajustar_ao_corredor else "só referência"
            linhas.append(
                f"  2. corredor dirigivel ({estado:13s}) {'':2s}"
                f"{ini:5.1f} -> {fim:5.1f} m"
            )
        else:
            linhas.append("  2. corredor ............................ sem segmentação")

        linhas.append(
            f"  3. apos ajuste + spline ................ "
            f"{len(self.rota_ajustada):4d} pts  {_extensao(self.rota_ajustada)}"
        )
        linhas.append(
            f"  4. apos filtro de projecao ............. "
            f"{len(self.rota_final):4d} pts  {_extensao(self.rota_final)}"
        )

        d = self.extras.get("diagnostico_filtro") or {}
        if d:
            linhas += [
                "",
                "Motivos de descarte no filtro de projeção:",
                f"  fora da tela ....... {d.get('fora_da_tela', 0)}",
                f"  fora da via ........ {d.get('fora_da_via', 0)}",
                f"  ocluidos ........... {d.get('ocluidos', 0)}",
                f"  validos ............ {d.get('validos', 0)} de {d.get('entrada', 0)}",
                f"  apos truncar ....... {d.get('apos_truncar', 0)}"
                f"  (tolerancia {d.get('tolerancia_fora_m', 0):.1f} m)",
            ]

        perda = len(self.rota_veiculo) and (
            self.rota_veiculo[:, 0].max() - (self.rota_final[:, 0].max() if len(self.rota_final) else 0)
        )
        if perda and perda > 5:
            linhas += [
                "",
                f"A rota final termina {perda:.0f} m antes da rota crua. Se a via segue",
                "visível além disso, relaxe `salto_max_frac` (rastreio do corredor)",
                "ou `tolerancia_fora_via_m` (filtro de projeção).",
            ]
        return "\n".join(linhas)


def executar(
    cfg: Config,
    usar_yolop: bool = True,
    yolop_carregado: segmentation.ModeloYolop | None = None,
    verboso: bool = True,
) -> Resultado:
    """Roda o pipeline inteiro para ``cfg.frame_alvo``."""

    def log(*args):
        if verboso:
            print(*args)

    # -- Módulo 1: GPS -------------------------------------------------
    log("[1/6] GPS")
    trilha = gps.preparar(cfg.gpx, cfg.janela_savgol, cfg.ordem_savgol)

    # -- Módulo 2: vídeo e sincronização -------------------------------
    log("[2/6] vídeo e sincronização")
    info = video.info(cfg.video)
    estado = sync.estado_no_frame(trilha, cfg.frame_alvo, info, cfg.offset_sync_s)
    rota_enu = sync.trajetoria_futura(
        trilha, estado, cfg.offset_sync_s, cfg.horizonte_s, cfg.distancia_max_m
    )
    rota_veiculo = sync.apenas_a_frente(
        sync.enu_para_veiculo(rota_enu, estado, cfg.usar_elevacao)
    )
    frame = video.extrair_frame(cfg.video, cfg.frame_alvo)

    # -- Módulo 3: calibração ------------------------------------------
    log("[3/6] calibração")
    intr, pose = calibration.preparar(
        cfg.arquivo_calibracao,
        (frame.shape[1], frame.shape[0]),
        cfg.altura_camera_m,
        cfg.pitch_deg,
        cfg.yaw_deg,
        cfg.roll_deg,
    )
    frame_corrigido = calibration.corrigir_distorcao(frame, intr)
    H_solo = calibration.homografia_solo_para_imagem(intr, pose)

    # -- Módulo 4: segmentação -----------------------------------------
    seg = None
    if usar_yolop:
        log("[4/6] segmentação YOLOPv2")
        modelo = yolop_carregado or segmentation.carregar_modelo(
            cfg.pesos_yolop, cfg.repo_yolop
        )
        seg = segmentation.segmentar(
            modelo,
            frame_corrigido,
            cfg.tamanho_inferencia,
            cfg.limiar_area_dirigivel,
            cfg.limiar_faixas,
        )
        seg.area_dirigivel = segmentation.limpar_mascara(seg.area_dirigivel)
    else:
        log("[4/6] segmentação IGNORADA (usar_yolop=False)")

    # -- Módulo 5: IPM e ajuste ----------------------------------------
    log("[5/6] bird's-eye view e ajuste ao corredor")
    grade = ipm.GradeBEV(
        cfg.bev_x_min, cfg.bev_x_max, cfg.bev_y_meia_largura, cfg.bev_px_por_m
    )
    bev = ipm.para_bev(frame_corrigido, H_solo, grade)

    corredor = None
    rota_ajustada = rota_veiculo
    if seg is not None:
        mascara_bev = ipm.mascara_para_bev(seg.area_dirigivel, H_solo, grade)
        # o corredor é sempre extraído: mesmo sem ajustar a rota, ele é a
        # referência para conferir o alinhamento na bird's-eye view
        corredor = ipm.extrair_corredor(
            mascara_bev, grade, salto_max_frac=cfg.salto_max_corredor
        )
        if cfg.ajustar_ao_corredor:
            rota_ajustada = ipm.ajustar_rota_ao_corredor(
                rota_veiculo, corredor, cfg.margem_borda_m, cfg.peso_centro
            )
    rota_ajustada = ipm.suavizar_rota(rota_ajustada)

    # -- Módulo 6: projeção e render -----------------------------------
    log("[6/6] projeção e renderização")
    diag_filtro: dict = {}
    rota_final, _ = projection.filtrar_rota(
        rota_ajustada,
        intr,
        pose,
        mascara_via=seg.area_dirigivel if seg else None,
        obstaculos=seg.obstaculos if seg else None,
        exigir_via=(seg is not None) and cfg.exigir_via_no_filtro,
        tolerancia_fora_m=cfg.tolerancia_fora_via_m,
        folga_tela_px=cfg.folga_tela_px,
        diagnostico=diag_filtro,
    )

    imagem = render.desenhar_rota(
        frame_corrigido,
        rota_final,
        intr,
        pose,
        cfg.largura_faixa_ar_m,
        # a segmentação tem a palavra final: nada é pintado fora do asfalto
        mascara_recorte=seg.area_dirigivel if seg else None,
    )
    imagem = render.desenhar_marcadores_distancia(imagem, rota_final, intr, pose)
    imagem = render.desenhar_minimapa(imagem, trilha, rota_enu, estado)
    imagem = render.desenhar_hud(
        imagem,
        [
            f"frame {cfg.frame_alvo}  t={estado.t_video:.1f}s",
            f"heading {estado.heading_deg:.0f} deg   v {estado.velocidade * 3.6:.0f} km/h",
            f"cam h={cfg.altura_camera_m:.2f}m pitch={cfg.pitch_deg:+.0f} deg",
            f"rota visivel: {rota_final[:, 0].max():.0f} m"
            if len(rota_final)
            else "rota visivel: --",
        ],
    )

    extras = {
        "diagnostico_filtro": diag_filtro,
        "H_solo_para_imagem": H_solo,
        "bev_anotada": ipm.desenhar_bev(
            bev, grade, rota_veiculo, rota_ajustada, corredor
        ),
        "horizonte_v": calibration.linha_do_horizonte(intr, pose),
    }

    return Resultado(
        config=cfg,
        trilha=trilha,
        info_video=info,
        estado=estado,
        rota_enu=rota_enu,
        rota_veiculo=rota_veiculo,
        intr=intr,
        pose=pose,
        frame=frame,
        frame_corrigido=frame_corrigido,
        seg=seg,
        grade=grade,
        bev=bev,
        corredor=corredor,
        rota_ajustada=rota_ajustada,
        rota_final=rota_final,
        imagem_final=imagem,
        extras=extras,
    )


def executar_varios(cfg: Config, frames: list[int], **kwargs) -> dict[int, Resultado]:
    """Processa vários frames reaproveitando o modelo já carregado na memória."""
    modelo = None
    if kwargs.get("usar_yolop", True):
        modelo = segmentation.carregar_modelo(cfg.pesos_yolop, cfg.repo_yolop)

    from dataclasses import replace

    saida: dict[int, Resultado] = {}
    for f in frames:
        saida[f] = executar(replace(cfg, frame_alvo=f), yolop_carregado=modelo, **kwargs)
    return saida
