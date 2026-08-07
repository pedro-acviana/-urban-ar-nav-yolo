import json
import glob
import numpy as np
import cv2

# ==========================================================
# CONFIGURAÇÕES DO TABULEIRO
# ==========================================================

# O tabuleiro possui 8x8 quadrados.
# O OpenCV utiliza o número de CANTOS INTERNOS:
# 8 quadrados -> 7 cantos

CHECKERBOARD = (7, 7)      # (largura, altura) em cantos internos

# Cada quadrado mede 2 cm = 20 mm
SQUARE_SIZE = 20.0         # milímetros

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []
gray = None  # Inicializa para evitar erro de tipo

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Carrega as imagens da pasta
images = glob.glob("imagens_calibracao/*.jpg")

if len(images) == 0:
    print(
        "Erro: Nenhuma imagem encontrada na pasta 'imagens_calibracao/'. Verifique o caminho."
    )
    exit()

print(f"✓ {len(images)} imagens encontradas para calibração")
print(
    f"✓ Tabuleiro: 8x8 quadrados "
    f"({CHECKERBOARD[0]}x{CHECKERBOARD[1]} cantos internos)"
)
print(f"✓ Tamanho do quadrado: {SQUARE_SIZE}mm")
print()

print("Processando imagens...")
for idx, fname in enumerate(images, 1):
    img = cv2.imread(fname)
    if img is None:
        print(f"  [{idx}/{len(images)}] ✗ {fname.split('/')[-1]} - Erro ao carregar imagem")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"  [{idx}/{len(images)}] ✓ {fname.split('/')[-1]} - Corners detectados")
    else:
        print(f"  [{idx}/{len(images)}] ✗ {fname.split('/')[-1]} - Falha na detecção")

print(f"\n✓ {len(objpoints)} imagens com corners detectados de {len(images)} total")

if len(objpoints) < 3:
    print("\n✗ Erro: Necessário pelo menos 3 imagens com detecção de corners para calibração")
    exit()

if gray is None:
    print("\n✗ Erro: Nenhuma imagem válida foi processada")
    exit()

cv2.destroyAllWindows()

# 2. EXECUTA A CALIBRAÇÃO
print("\nExecutando calibração da câmera...")
print(f"  - Usando {len(objpoints)} imagens para calibração")
print(f"  - Tamanho da imagem: {gray.shape[1]}x{gray.shape[0]} pixels")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(  # type: ignore
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"✓ Calibração concluída com sucesso!")
print(f"✓ Erro RMS de reprojeção: {ret:.4f} pixels")

# 3. PREPARAÇÃO DOS DADOS PARA O JSON
# .tolist() converte os arrays do Numpy em listas Python nativas
dados_calibracao = {
    "erro_reprojecao_rms": float(ret),
    "parametros_intrinsecos": {
        "matriz_camera_mtx": mtx.tolist(),
        "coeficientes_distorcao_dist": dist.tolist(),
    },
    "parametros_extrinsecos": [],
}

# Organiza os parâmetros extrínsecos por imagem processada
for i in range(len(rvecs)):
    dados_calibracao["parametros_extrinsecos"].append(
        {
            "imagem_indice": i,
            "vetor_rotacao_rvec": rvecs[i].tolist(),
            "vetor_translacao_tvec": tvecs[i].tolist(),
        }
    )

# 4. SALVANDO EM ARQUIVO JSON
nome_arquivo = "calibracao_camera.json"

print(f"\nSalvando parâmetros em '{nome_arquivo}'...")
with open(nome_arquivo, "w", encoding="utf-8") as f:
    json.dump(dados_calibracao, f, indent=4, ensure_ascii=False)

print(f"✓ Arquivo salvo com sucesso!")
print(f"\n{'='*60}")
print(f"CALIBRAÇÃO FINALIZADA")
print(f"{'='*60}")
print(f"Erro RMS: {ret:.4f} pixels")
print(f"Parâmetros salvos em: {nome_arquivo}")
print(f"{'='*60}")
