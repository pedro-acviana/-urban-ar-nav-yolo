"""
Exporta frames do `.mp4` bruto para `data/frames/<percurso>/`.

Os vídeos de campo passam de 400 MB, ficam fora do Git e só existem na
máquina de quem gravou — quem clonasse o repositório não conseguia rodar o
pipeline. Este script produz a fonte versionável: alguns PNGs e um
`info.json` com fps/resolução/contagem do vídeo original, que é o que a
sincronização com o GPX precisa.

Uso
---
    # os frames que o notebook já usa
    python scripts/exportar_frames.py volta_menor

    # frames avulsos
    python scripts/exportar_frames.py volta_menor --frames 692 1200

    # tudo a cada 200 frames
    python scripts/exportar_frames.py volta_menor --passo 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos import video  # noqa: E402
from modulos.config import Config  # noqa: E402

# O conjunto que o notebook precisa para rodar sem o `.mp4`:
#   - seção 9, frames principais: 400, 692, 1200, 2000, 2600, 3400, 4166
#     (reta, curva e cruzamento ao longo do percurso);
#   - calibração de offset, frames de curva: 692, 2013, 2600, 3607 — só curva
#     discrimina offset, numa reta qualquer valor parece certo.
FRAMES_PRINCIPAIS = [400, 692, 1200, 2000, 2013, 2600, 3400, 3607, 4166]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("percurso", nargs="?", default="volta_menor")
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help=f"índices a exportar (padrão: {FRAMES_PRINCIPAIS})",
    )
    p.add_argument(
        "--passo",
        type=int,
        help="exporta de N em N frames, do início ao fim do vídeo",
    )
    args = p.parse_args(argv)

    cfg = Config(nome_percurso=args.percurso)
    if not cfg.video.exists():
        print(
            f"Vídeo não encontrado: {cfg.video}\n"
            "Os `.mp4` não são versionados — veja o README.",
            file=sys.stderr,
        )
        return 1

    if args.passo:
        meta = video.info(cfg.video)
        indices = list(range(0, meta.n_frames, args.passo))
    else:
        indices = args.frames or FRAMES_PRINCIPAIS

    print(f"Lendo {cfg.video.name} ({len(indices)} frames)...")
    salvos = video.exportar_frames(cfg.video, cfg.dir_frames, indices)

    total_mb = sum(p.stat().st_size for p in salvos.values()) / 1e6
    print(f"{len(salvos)} frames em {cfg.dir_frames} ({total_mb:.1f} MB)")
    print(f"metadados: {cfg.dir_frames / video.NOME_METADADOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
