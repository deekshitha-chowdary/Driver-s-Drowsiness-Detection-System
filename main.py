
import cv2
import mediapipe as mp
import math
import time
import winsound

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_ratio(landmarks, eye, width, height):
    points = []
    for i in eye:
        p = landmarks[i]
        points.append((int(p.x * width), int(p.y * height)))

    vertical1 = math.dist(points[1], points[5])
    vertical2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    return (vertical1 + vertical2) / (2 * horizontal)

camera = cv2.VideoCapture(0)
closed_since = None
alarm_played = False

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            print("Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "No face detected"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left = eye_ratio(landmarks, LEFT_EYE, width, height)
            right = eye_ratio(landmarks, RIGHT_EYE, width, height)
            ratio = (left + right) / 2

            if ratio < 0.22:
                if closed_since is None:
                    closed_since = time.time()

                if time.time() - closed_since >= 2:
                    status = "DROWSINESS ALERT!"
                    if not alarm_played:
                        winsound.Beep(2000, 1000)
                        alarm_played = True
                else:
                    status = "Eyes closed"
            else:
                closed_since = None
                alarm_played = False
                status = "Eyes open"

            cv2.putText(
                frame, f"Eye ratio: {ratio:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

        cv2.putText(
            frame, status, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255) if "ALERT" in status
            else (0, 255, 0), 2
        )

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
