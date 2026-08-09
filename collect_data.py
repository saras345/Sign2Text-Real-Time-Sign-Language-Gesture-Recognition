import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

os.makedirs('data', exist_ok=True)

cap = cv2.VideoCapture(0)
gesture_name = input("Enter gesture name (e.g., A, B, C): ")
samples = []

print(f"Collecting data for gesture: {gesture_name}")
print("Show your hand gesture to the camera. Press 'q' to quit early.")

while len(samples) < 100:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            samples.append(landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.putText(frame, f"Samples: {len(samples)}/100", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Collect Data', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if len(samples) >= 50:
    np.save(f'data/{gesture_name}.npy', np.array(samples))
    print(f"Saved {len(samples)} samples for gesture: {gesture_name}")
else:
    print(f"Not enough samples collected ({len(samples)}). Need at least 50.")

cap.release()
cv2.destroyAllWindows()