# Navegação AR urbana — projeção de rota GPS sobre a imagem do motorista

Protótipo do PIBIC 26/27. Dada uma rota gravada por GPS, os parâmetros de
calibração da câmera e a segmentação da via por uma rede neural, o sistema
desenha o trajeto futuro **sobre o asfalto**, na perspectiva do motorista.

## Ideia

O trajeto vem do GPS; o asfalto visível vem da câmera. Cada um sozinho é
insuficiente — o GPS de celular erra alguns metros (a rota "sobe na calçada")
e a câmera não sabe para onde o carro vai. A premissa do projeto é que **o
espaço de rua visível faz parte do trajeto GPS**, então as duas fontes são
conciliadas numa vista superior métrica (bird's-eye view), onde a rota é
empurrada para dentro do corredor dirigível antes de voltar para a imagem.

## Pipeline

| # | Módulo | Arquivo | Entrada → Saída |
|---|--------|---------|-----------------|
| 1 | GPS | `modulos/gps.py` | GPX → ENU, heading recalculado, velocidade |
| 2 | Sincronização | `modulos/video.py`, `sync.py` | frame ↔ posição; rota futura no referencial do veículo |
| 3 | Calibração | `modulos/calibration.py` | `K`, distorção, pose da câmera (`R`, `t`) |
| 4 | Segmentação | `modulos/segmentation.py` | frame → área dirigível, faixas, obstáculos (YOLOPv2) |
| 5 | IPM / BEV | `modulos/ipm.py` | homografia do solo, corredor dirigível, ajuste da rota |
| 6 | Projeção e render | `modulos/projection.py`, `render.py` | rota → pixels, oclusão, faixa AR |
| 7 | Orquestração | `modulos/pipeline.py` | tudo acima, num comando |

### A matemática, em quatro passos

**1. Esférico → cartesiano local.** As coordenadas do GPX passam por ECEF e
chegam ao plano tangente local (ENU), onde a geometria projetiva é linear.
A rota é então reescrita no referencial do veículo:

```
X =  Δe·sin(h) + Δn·cos(h)        (frente)
Y = -Δe·cos(h) + Δn·sin(h)        (esquerda)
```

**2. Mundo → câmera (extrínsecos).** `X_C = R·X_W + t`, com
`R = Rz(roll)·Rx(-pitch)·Ry(yaw)·R_base` e `t = -R·(0,0,h)ᵀ`.

**3. Câmera → pixels (intrínsecos).** `ỹ = K·X_C`, seguido da divisão pela
profundidade: `u = ỹ₀/ỹ₂`, `v = ỹ₁/ỹ₂`.

**4. Suavização, oclusão e recorte.** Spline cúbica no plano métrico, corte
por profundidade negativa, por área não dirigível e por obstáculos detectados.

### O "truque" da bird's-eye view

Para pontos do solo (`Z = 0`) a projeção colapsa numa homografia
`H = K·[r₁|r₂|t]`. Invertê-la produz a vista superior. Nela:

1. mede-se o **corredor dirigível** linha a linha (largura e centro em metros);
2. a rota do GPS é atraída ao centro do corredor (`peso_centro`) e confinada
   a uma margem das bordas;
3. o ajuste de curva acontece num espaço euclidiano, então a suavização não
   fica deformada pela perspectiva;
4. a curva volta para a imagem pela mesma homografia.

## Instalação

Desenvolvido em **Python 3.14.5**. O único submódulo é o YOLOPv2.

```bash
git clone --recurse-submodules https://github.com/pedro-acviana/-urban-ar-nav-yolo.git
cd -urban-ar-nav-yolo
pip install -r requirements.txt
```

Se já tiver clonado sem o submódulo:

```bash
git submodule update --init YOLOPv2
```

Falta só baixar os pesos, e o pipeline roda:

```bash
curl -L -o YOLOPv2/data/weights/yolopv2.pt \
  https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
```

### O que vem no repositório, e o que não vem

Os frames de trabalho **são versionados** (`data/frames/`), então um clone roda
o pipeline de ponta a ponta sem precisar dos vídeos brutos. Também estão no
Git os `.gpx`/`.kml` e o `calibracao_camera.json`.

Fica de fora, por tamanho:

| Arquivo | Precisa? | Como obter |
|---|---|---|
| `YOLOPv2/data/weights/yolopv2.pt` | **sim** | [release oficial](https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt) (~156 MB) |
| `data/*.mp4` | só para exportar frames novos | vídeos gravados em campo, ~400 MB cada |
| `camera_calibration/imagens_calibracao/` | só para recalibrar | fotos do tabuleiro de xadrez |

## Uso

```bash
jupyter lab notebooks/workflow_prototipo.ipynb
```

Ou direto em Python:

```python
from modulos.config import Config
from modulos import pipeline

cfg = Config(nome_percurso="volta_menor", frame_alvo=692,
             altura_camera_m=1.50, pitch_deg=-10.0)

resultado = pipeline.executar(cfg)
resultado.salvar()
```

### Frames como fonte principal

O pipeline lê as imagens de `data/frames/<percurso>/`, não do `.mp4`. Cada
pasta tem os PNGs e um `info.json` com os metadados do vídeo de origem — é
dele que sai o `fps` usado para casar o frame com o relógio do GPX, de modo
que a sincronização continua exata sem o vídeo por perto.

```python
cfg.frames_exportados()      # [400, 692, 1200, 2000, 2600, 3400, 4166]
cfg.checar_arquivos()        # o .mp4 aparece como opcional
```

`Config.fonte_frames` controla a escolha: `"auto"` (padrão) usa os frames
quando existem e cai no `.mp4` quando não; `"frames"` e `"video"` forçam um
dos dois. Para trabalhar num frame que ainda não foi exportado, é preciso ter
o vídeo:

```bash
python scripts/exportar_frames.py volta_menor --frames 1500 1800
```

> **Taxa variável.** Os `.mp4` de celular são VFR: a leitura sequencial e o
> seek por índice apontam para instantes diferentes — no `volta_menor` a
> diferença chega a 1,7 s no frame 692. Todo o pipeline usa a convenção do
> seek (`t = índice / fps`), que é a mesma de `video.extrair_frame` e a que
> `exportar_frames` reproduz. `video.extrair_varios` conta frame a frame e
> **não** serve para exportar: trocaria a imagem sob uma calibração de offset
> que continua apontando para o instante antigo.

### Offset de sincronização

O parâmetro mais sensível do pipeline: a 30 km/h, um erro de 2 s desloca o
carro em ~17 m e a rota projetada passa a descrever uma curva que ainda vai
acontecer. Os valores já calibrados ficam em `modulos/config.py`:

```python
OFFSETS_CALIBRADOS = {"volta_menor": 2.0}
```

`Config` usa a tabela automaticamente. Para calibrar um percurso novo, use
`render.painel_offsets` — o offset certo é aquele em que a rota acompanha o
asfalto visível — e registre o valor ali.

## Calibrando a pose da câmera

`K` e os coeficientes de distorção vêm do tabuleiro de xadrez, mas os
`rvec`/`tvec` daquele JSON descrevem a pose de cada **tabuleiro**, não a da
câmera dentro do carro. Altura, pitch, yaw e roll são parâmetros manuais.

Para acertá-los, a seção 3 do notebook projeta uma grade métrica sobre o
frame — basta ajustar até as linhas assentarem no asfalto. Dois atalhos:

```python
# o horizonte depende só do pitch
cal.pitch_pelo_horizonte(intr, v_horizonte=780)

# um ponto no asfalto de distância conhecida dá a altura
cal.altura_por_referencia(intr, pitch_deg=0.0, v=1350, distancia_m=8.0)
```

## Estrutura

```
modulos/          pacote com as etapas do pipeline
notebooks/        workflow_prototipo.ipynb (+ legado/)
data/             .gpx e .kml versionados; .mp4 ignorados
  frames/<percurso>/   PNGs + info.json — a fonte de imagens, versionada
scripts/          exportar_frames.py, destravar_git.sh
camera_calibration/  calibracao_camera.json + script de calibração
output/           resultados do pipeline (ignorado)
YOLOPv2/          submódulo — segmentação de via
```

## Limitações

1. **Pose manual** — altura e pitch são ajustados a olho. Próximo passo:
   estimar pitch/roll pelo ponto de fuga das faixas.
2. **Solo plano** — a IPM assume `Z = 0`; em ladeira o erro cresce com a
   distância.
3. **Sincronização por offset constante**, sem correção de deriva. O
   estimador automático (`sync.perfil_de_movimento` + `sync.estimar_offset`)
   correlaciona o fluxo óptico do vídeo com a velocidade e a guinada do GPX,
   mas precisa de eventos marcantes; em trajetos retilíneos a correlação não
   converge e o ajuste recai sobre `render.painel_offsets`, visual.
4. **Calibração com fotos**, cuja proporção difere levemente da do vídeo:
   `fx` e `fy` recebem fatores de escala distintos. Recalibrar gravando vídeo
   elimina a aproximação.
5. **Um frame por vez**, sem filtro temporal.
6. **Só os frames exportados** são alcançáveis num clone. Varrer o percurso
   inteiro (o painel de offset, por exemplo) ainda exige o `.mp4` original.

## Referências

- [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) — Han et al., 2022
- [HybridNets](https://github.com/datvuthanh/HybridNets) — Vu et al., 2022
- Artigo base da arquitetura GAN-LSTM: *Sensors* 25(3), 820
