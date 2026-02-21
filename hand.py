import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ----------------------------
# Initialize MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Initialize Volume Control
# ----------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_min, vol_max = volume.GetVolumeRange()[:2]

# ----------------------------
# Start Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, c = img.shape

            # Thumb tip (4) and Index tip (8)
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)

            x2 = int(hand_landmarks.landmark[8].x * w)
            y2 = int(hand_landmarks.landmark[8].y * h)

            # Draw circles
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume
            vol = np.interp(length, [30, 200], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol, None)

            # Volume percentage
            vol_per = np.interp(length, [30, 200], [0, 100])

            # Draw volume bar
            vol_bar = np.interp(length, [30, 200], [400, 150])

            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400),
                          (0, 0, 255), cv2.FILLED)

            cv2.putText(img, f'{int(vol_per)} %',
                        (40, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)

            mp_draw.draw_landmarks(img, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()