import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

print("Loading data...")
X, y = [], []
gesture_labels = {}
label_idx = 0

for filename in os.listdir('data'):
    if filename.endswith('.npy'):
        gesture_name = filename.replace('.npy', '')
        gesture_labels[label_idx] = gesture_name
        
        data = np.load(f'data/{filename}')
        X.extend(data)
        y.extend([label_idx] * len(data))
        print(f"Loaded {len(data)} samples for '{gesture_name}'")
        label_idx += 1

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Total gestures: {len(gesture_labels)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("\nBuilding model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(gesture_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=1)

model.save('sign_language_model.h5')
np.save('gesture_labels.npy', gesture_labels)

print(f"\n✓ Model trained successfully!")
print(f"✓ Validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"✓ Saved as 'sign_language_model.h5'")