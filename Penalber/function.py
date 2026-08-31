import cv2
import time
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

def two_viewports(frame, image, black_image, screen_h, screen_w):

    ## Formaliza como vai ficar a proporção da tela.
    ## Depois da resize nelas para poder caber certinho.
    ## E por fim junta elas pelo hconcat.
    h, w = image.shape[:2]
    ## Mesma coisa do codigo de baixo, usa somente os dois primeiros numeros [altura, largural, canal(ñ usa)]

    double_height = w * 2
    ## Valor ajustado para a tela dupla ficar certinha, na metada do meu note, vê depois se ficou bom ai.

    
    scale = min(screen_w / double_height, screen_h / h)
    height = int(h * scale)
    width = int(w * scale)
    black_image_resized = cv2.resize(black_image, (width, height), interpolation = cv2.INTER_AREA)
    image_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    final_image = cv2.hconcat([image_resized, black_image_resized])

    return final_image

def float_viewport(image, background, scale = 0.2):
    ## Função recebendo, image = imagem crua da webcam; background = imagem preta sem nenhuma impressão; scale = escala para multiplicar por 0.3.

    bg_h, bg_w = background.shape[:2]
    ## da nome às variáveis da height e width e marca elas para utilizarem somente 2 valores. [altura, largura, canal (ñ usa esse)]

    float_w = int(bg_w * scale)
    float_h = int(bg_h * scale)
    ## Calcula o valor inteiro de pixels pra proporção da tela.
    floating_resized = cv2.resize(image, (float_w, float_h), interpolation = cv2.INTER_AREA)

    pad = 20
    y1 = bg_h - float_h - pad
    y2 = bg_h - pad
    x1 = bg_w - float_w - pad
    x2 = bg_w - pad

    ## Calcula a posição de onde a imagem flutuante deve ficar, nesse caso, no canto inferior direito.

    result = background.copy()
    result[y1:y2, x1:x2] = floating_resized
    ## Bota a imagemzinha onde ela deve ficar.


    return result