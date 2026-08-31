import cv2
import time
import mediapipe as mp
import numpy as np
import tkinter as tk
from function import two_viewports
from function import float_viewport

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

## Fiz isso pra webcam capturar com qualidade.

root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

## ele abre uma janelinha baseada no tkinker para verificar o tamanho da tela do usuario e depois pega os dados e destroi essa telinha.

screen_mode = 1


with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        valid, frame = webcam.read()

        timestamp_ms = int(time.time() * 1000)
    ## Criei os frames que estarão na tela.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
    ## Criei a variavel da altura e largura para poder passar a tela preta.
        image = frame.copy()

        black_image = np.zeros((h, w, 3), dtype = np.uint8)
        black_image2 = np.zeros((h, w, 3), dtype = np.uint8)
    ## Isso aqui faz a tela ficar totalmente preta.

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)    
    ## Faz tudo num unico if
    
        if result.face_landmarks:   
            h, w, _ = image.shape
            for face_landmarks in result.face_landmarks:
                    
            # 1. Desenha as linhas de contorno do rosto
                for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
                    start_idx = connection.start
                    end_idx = connection.end
                    p1 = face_landmarks[start_idx]
                    p2 = face_landmarks[end_idx]

                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(black_image, pt1, pt2, (255, 255, 255), 1)
                    cv2.line(image, pt1, pt2, (255, 255, 255), 1)
    
                for landmark in face_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(black_image, (cx, cy), 1, (0, 255, 0), -1)
                    cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)
                       
        if  screen_mode == 1:
            final_image = two_viewports(frame, image, black_image, screen_w, screen_h)
        else:
            final_image = float_viewport(image, background = black_image2, scale = 0.2)
            final_image = cv2.resize(final_image, (1920, 1080))

        ## Roda  as duas funções para criação da segunda tela e já adiciona a tela base com as duas viewports.

        cv2.imshow("webcam", final_image)
        key = cv2.waitKey(5) & 0xFF


        if key == 27:  # Tecla ESC para fechar
            break
        elif key == ord('m') or key == ord('M'):
            screen_mode = 2 if screen_mode == 1 else 1

        ## Usa a  tecla M para mudar a interface.

webcam.release()
cv2.destroyAllWindows()

## Essa bosta ta descalibrada/glitchando, ja tentei resolver mas n sei como