import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('sign_language_model.h5')
gesture_labels = np.load('gesture_labels.npy', allow_pickle=True).item()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print("Starting recognition... Press 'q' to quit")

while True:
    
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
            
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks, verbose=0)
            gesture_idx = np.argmax(prediction)
            confidence = prediction[0][gesture_idx]
            
            if confidence > 0.7:
                gesture_name = gesture_labels[gesture_idx]
                
                # Display the recognized letter/word
                cv2.putText(frame, f"{gesture_name}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Print in terminal too
                print(f"Detected: {gesture_name} ({confidence:.2f})")
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()