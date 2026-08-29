import cv2
import time
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

def two_viewports(frame, image, black_image):

    height = frame.shape[1]
    width = frame.shape[0]
    black_image_resized = cv2.resize(black_image, (height, width))
    image_resized = cv2.resize(image, (height, width))
    final_image = cv2.hconcat([image_resized, black_image_resized])

    return final_image
    