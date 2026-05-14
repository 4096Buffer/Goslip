import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import pygame
import threading

pygame.mixer.init()

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('http://IP:4747/video')


LEFT_EYE_LOW = 472
LEFT_EYE_TOP = 158
RIGHT_EYE_TOP = 385
RIGHT_EYE_LOW = 374

TRIGGERS = 0

THRESHOLD = 0.0078
MIN_VAL = 0.0045

alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True
    pygame.mixer.music.load(r'wakeup.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    alert_playing = False


with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    left_diff = 0
    right_diff = 0
    values = []
    last_mean_calc = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            points = [LEFT_EYE_LOW, LEFT_EYE_TOP, RIGHT_EYE_LOW, RIGHT_EYE_TOP]

            left_low_y = results.multi_face_landmarks[0].landmark[LEFT_EYE_LOW].y
            left_top_y = results.multi_face_landmarks[0].landmark[LEFT_EYE_TOP].y
            right_low_y = results.multi_face_landmarks[0].landmark[RIGHT_EYE_LOW].y
            right_top_y = results.multi_face_landmarks[0].landmark[RIGHT_EYE_TOP].y

            left_diff = abs(left_low_y - left_top_y)
            right_diff = abs(right_low_y - right_top_y)

            for i in points:
                lm = results.multi_face_landmarks[0].landmark[i]

                px, py = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, str(i), (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f'LEFT EYE DIFF - {str(left_diff)}', (10,20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f'RIGHT EYE DIFF - {str(right_diff)}', (10,50), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        now = datetime.now()

        if not last_mean_calc is None and (now - last_mean_calc).total_seconds() > 60:
            values = []

        values.append(left_diff)
        values.append(right_diff)

        if last_mean_calc is None:
            last_mean_calc = now

        median = np.median(values)

        if median < THRESHOLD and median > MIN_VAL:
            cv2.putText(frame, 'WAKE UP! WAKE UP!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            if not alert_playing:
                threading.Thread(target=play_alert, daemon=True).start()

        cv2.putText(frame, f'MEDIAN - {median}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.imshow('Kamera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()