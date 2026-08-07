Acho que você está indo exatamente para um problema de pesquisa muito interessante. O ponto principal é que **o HybridNets sozinho não resolve a projeção da rota**. Ele resolve a **percepção** (o que existe na imagem), enquanto você precisa resolver também a **geometria** (onde desenhar a rota).

Eu dividiria o sistema em cinco módulos.

---

# Arquitetura geral

```text
              GPX / KML / KMZ
                     │
                     ▼
           Trajetória GPS (lat, lon)
                     │
                     ▼
          Conversão para ENU (x,y,z)
                     │
                     ▼
        Pose do veículo (frame atual)
                     │
                     ▼
        Projeção 3D → imagem (OpenCV)
                     │
                     ▼
     HybridNets (faixas + área dirigível)
                     │
                     ▼
 Ajuste da rota à pista detectada (lane fitting)
                     │
                     ▼
      Overlay estilo GTA / Google Maps AR
```

---

# Etapa 1 — Obter a trajetória

Você já possui:

* vídeo
* GPX
* KML
* KMZ

O GPX já contém uma sequência de pontos:

```text
lat
lon
tempo
```

Exemplo:

```text
-15.7632
-47.8845
09:31:20

-15.7631
-47.8841
09:31:21

...
```

---

# Etapa 2 — Sincronizar vídeo e GPS

Essa é uma das partes mais importantes.

Você precisa saber:

> Em qual frame o carro estava em qual posição do GPX.

Se o vídeo foi gravado pelo Poco X8 Pro, normalmente há timestamp no vídeo.

Então:

```
Frame 523

↓

09:31:24.2

↓

GPS 09:31:24.2
```

Essa sincronização pode ser feita por interpolação.

---

# Etapa 3 — Converter GPS para coordenadas locais

GPS em latitude/longitude não serve diretamente.

Converta para um sistema cartesiano.

Normalmente usa-se ENU.

```
GPS

↓

ECEF

↓

ENU
```

Você obtém algo assim:

```
0 m
0 m

↓

3.2 m
1.1 m

↓

7.4 m
1.8 m

↓

10.5 m
2.6 m
```

Agora você possui uma trajetória em metros.

---

# Etapa 4 — Descobrir a pose da câmera

Esse é o grande desafio.

Você precisa saber:

* posição da câmera
* orientação

Seu Poco X8 Pro possui:

* câmera
* IMU
* GPS

Mas se você estiver usando apenas o vídeo, pode estimar isso por:

* ORB-SLAM3
* OpenVSLAM
* VINS-Fusion

No PIBIC, pode começar assumindo que:

* câmera fixa
* altura conhecida
* inclinação conhecida

Por exemplo:

```
altura = 1.35 m

pitch = 4°

yaw = direção do GPS
```

Isso já permite bons resultados.

---

# Etapa 5 — Calibração da câmera

Você precisa dos parâmetros intrínsecos.

```
fx

fy

cx

cy
```

Isso pode ser obtido com OpenCV usando um tabuleiro de xadrez.

```python
cv2.calibrateCamera()
```

Isso gera a matriz K.

---

# Etapa 6 — Projetar a rota

Agora entra a matemática.

Você possui:

```
Trajetória 3D

↓

OpenCV

↓

projectPoints()

↓

Imagem
```

O OpenCV faz toda a projeção.

Você obtém:

```
(150,820)

↓

(170,760)

↓

(230,700)

↓

(310,640)
```

Esses são pixels.

Basta desenhar uma linha.

---

# Onde entra o HybridNets?

Aqui.

O HybridNets produz algo parecido com:

```
########################

Rua

########################

Faixa esquerda

Faixa direita

Área dirigível

Carros
```

Você pode usar isso para corrigir a projeção.

Imagine que a projeção do GPS fique assim:

```
       ______

------/
```

Mas a pista detectada é:

```
#############

#############
```

Você ajusta a linha para permanecer dentro da área dirigível.

---

# Melhor ainda

O HybridNets fornece a máscara da área dirigível.

Você pode fazer:

```
Linha GPS

↓

Máscara da pista

↓

Skeleton

↓

Bezier

↓

Linha suavizada
```

O resultado fica muito parecido com GTA.

---

# Curvas

Em vez de desenhar segmentos retos:

```
---------

---------
```

Você pode gerar:

Spline cúbica

ou

Bezier.

Fica assim:

```
~~~~~~~
```

Muito mais natural.

---

# Oclusão por carros

O HybridNets detecta carros.

Se um carro estiver à frente:

```
rota

↓

carro

↓

não desenha atrás do carro
```

Isso gera um efeito muito parecido com HUD automotivo.

---

# Como fica o pipeline completo?

```text
Vídeo
          │
          ▼
     HybridNets
          │
 ┌────────┴─────────┐
 │                  │
 ▼                  ▼
Faixas         Área dirigível
 │                  │
 └────────┬─────────┘
          ▼
   Ajuste da rota
          ▲
          │
GPS → ENU → projectPoints
          │
          ▼
Overlay final
```

---

# O que eu faria no PIBIC

Eu faria em quatro etapas.

### Etapa 1

Projetar a trajetória GPS sobre o vídeo.

Sem IA.

Só OpenCV.

---

### Etapa 2

Adicionar HybridNets.

Detectar:

* faixa
* pista
* carros

---

### Etapa 3

Ajustar automaticamente a trajetória para permanecer dentro da área dirigível detectada.

---

### Etapa 4 (TCC)

Substituir o GPS puro por uma fusão de sensores:

* GPS
* IMU
* odometria visual (ORB-SLAM3 ou VINS-Fusion)

Assim, a trajetória projetada permanece estável mesmo quando o GPS oscila alguns metros.

## Um diferencial que pode tornar o trabalho bem interessante

Em vez de apenas desenhar uma linha sobre o vídeo, faça a rota parecer **"pintada no asfalto"**, como nos jogos GTA ou Forza. Para isso:

* projete a trajetória no plano da estrada usando a calibração da câmera;
* restrinja a rota à máscara da área dirigível fornecida pelo HybridNets;
* suavize a trajetória com splines ou curvas de Bézier;
* aplique perspectiva (a linha fica mais estreita à medida que se afasta);
* desenhe a rota parcialmente ocluída por veículos detectados (a linha desaparece atrás de carros).

Esse tipo de renderização aproxima bastante o resultado de sistemas comerciais de navegação em realidade aumentada e representa uma evolução natural do PIBIC para um TCC com contribuição técnica própria.
