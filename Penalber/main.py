import cv2
import time
import mediapipe as mp
import numpy as np
from function import two_viewports

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        valid, frame = webcam.read()

        timestamp_ms = int(time.time() * 1000)
    
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (960, 540))
        h, w, _ = frame.shape

        image = frame.copy()

        black_image = np.zeros((h, w, 3), dtype = np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)    
    
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
                       
        final_image = two_viewports(frame, image, black_image)

        cv2.imshow("webcam", final_image)

        if cv2.waitKey(5) == 27:  # Tecla ESC para fechar
            break

webcam.release()
cv2.destroyAllWindows()