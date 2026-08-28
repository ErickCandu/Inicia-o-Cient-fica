import cv2
import time
import mediapipe as mp

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

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        valid, frame = webcam.read()

        timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in result.face_landmarks:
                
                # 1. Desenha as linhas de contorno do rosto
                for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
                    start_idx = connection.start
                    end_idx = connection.end

                    p1 = face_landmarks[start_idx]
                    p2 = face_landmarks[end_idx]

                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))

                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

                for landmark in face_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        cv2.imshow("webcam", frame)

        if cv2.waitKey(5) == 27:  # Tecla ESC para fechar
            break

webcam.release()
cv2.destroyAllWindows()