 Sign2Text — Real-Time Sign Language Recognition

Sign2Text is a Python-based real-time sign language recognition system that uses Computer Vision and Deep Learning to detect hand gestures through a webcam and convert them into readable text.

The system uses MediaPipe to extract hand landmarks and a trained TensorFlow/Keras neural network to classify different gestures. A Gradio interface provides an interactive web-based application for real-time recognition.

## Features

-  Real-time sign language recognition
-  Webcam-based hand gesture detection
-  21-point hand landmark detection using MediaPipe
-  Neural Network classification using TensorFlow/Keras
-  Gesture-to-text conversion
-  Prediction confidence score
-  Real-time processing
-  Interactive Gradio web interface
-  Dark/Light theme
-  Demo login system
-  Model and performance information
-  Future scope section

##  Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- TensorFlow
- Keras
- Scikit-learn
- Gradio

##  How It Works

The system follows the following pipeline:

Webcam
→ OpenCV
→ MediaPipe Hand Detection
→ 21 Hand Landmarks
→ 63 Numerical Features
→ Neural Network
→ Gesture Classification
→ Confidence Check
→ Recognized Text

Each detected hand contains 21 landmarks, and every landmark contains X, Y and Z coordinates.

Therefore:

21 landmarks × 3 coordinates = 63 features

These 63 features are passed to the trained neural network for gesture classification.

## Model Architecture

The project uses a feed-forward neural network:

Input Layer
→ Dense(128, ReLU)
→ Dropout(0.3)
→ Dense(64, ReLU)
→ Dropout(0.3)
→ Output Layer(Softmax)

The number of output neurons depends on the number of gestures present in the dataset.

##  Project Structure

Sign2Text/
│
├── app.py
├── collect_data.py
├── train_model.py
├── recognize.py
├── camera_test.py
│
├── sign_language_model.h5
├── gesture_labels.npy
│
├── data/
│   ├── A.npy
│   ├── B.npy
│   ├── C.npy
│   └── ...
│
├── .gitignore
└── README.md

##  Dataset Collection

The dataset is created using webcam-based hand landmark extraction.

Run:

python collect_data.py

Enter the gesture name when prompted:

A

The program detects the hand using MediaPipe and collects up to 100 samples for the selected gesture.

The collected data is stored inside the `data/` directory.

Example:

data/A.npy
data/B.npy
data/C.npy

At least 50 samples are required for a gesture dataset to be saved.

## Training the Model

After collecting the gesture data, train the model using:

python train_model.py

The training script:

1. Loads all gesture datasets.
2. Assigns numerical labels to each gesture.
3. Combines the datasets.
4. Splits the data into training and testing sets.
5. Builds the neural network.
6. Trains the model for 50 epochs.
7. Evaluates validation accuracy.
8. Saves the trained model.

The generated files are:

sign_language_model.h5

gesture_labels.npy

##  Standalone Recognition

To test the trained model directly with OpenCV:

python recognize.py

The application opens the webcam and displays:

- Detected gesture
- Confidence score
- Hand landmarks

Press `Q` to exit.

## Gradio Web Application

The main application is built using Gradio.

Run:

python app.py

After starting the application, Gradio provides a local URL such as:

http://127.0.0.1:7860

Open the URL in your browser.

The application contains:

- Home
- Recognition
- How It Works
- Model
- Performance
- Future Scope
- About

##  Login

The application contains a simple demo login system.

Any non-empty username and password can be used to enter the application.

This login is only for demonstration purposes and is not a production authentication system.

##  Recognition Logic

The system uses a confidence threshold of 0.70.

A gesture is accepted only when the prediction confidence is above the threshold.

The application also uses stable-frame detection to reduce repeated or unstable predictions.

A gesture must remain stable for multiple frames before being added to the transcript.

##  Gesture-to-Text Conversion

Once a gesture is confidently detected and remains stable, its corresponding label is added to the transcript.

Example:

A → B → C

Output:

ABC

The transcript can be cleared using the Clear Transcript button.

##  Performance

The actual performance depends on the dataset, number of gestures, number of samples, lighting conditions, camera quality and similarity between gestures.

The training script automatically displays the validation accuracy after training.

Example:

Validation accuracy: XX.XX%

Recommended evaluation metrics for future improvements:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

##  Why MediaPipe Landmarks?

Instead of training directly on complete webcam images, Sign2Text uses hand landmark coordinates.

This approach provides:

- Lower-dimensional input
- Faster processing
- Lightweight dataset
- Real-time performance
- Reduced dependency on background
- Focus on hand structure

## Future Scope

### Sentence-Level Recognition

Extend the system from individual gestures to complete sentences using sequential gesture recognition.

### Voice Output

Convert recognized text into speech using Text-to-Speech technology.

### Larger Gesture Vocabulary

Expand the dataset to include more alphabets, numbers, words and commonly used phrases.

### Multilingual Translation

Translate recognized signs into multiple languages.

### Mobile Deployment

Deploy the trained model on mobile devices using lightweight model formats such as TensorFlow Lite.

### Advanced Deep Learning

Experiment with more advanced architectures and sequence-based models to improve recognition accuracy.

### Continuous Sign Recognition

Develop a system capable of understanding continuous sequences of gestures instead of isolated signs.

##  Current Limitations

- The model can recognize only gestures included in the training dataset.
- Recognition accuracy depends on the quality and diversity of collected data.
- Poor lighting can affect hand detection.
- Occluded or partially visible hands may reduce prediction accuracy.
- The current system focuses mainly on individual gestures.
- The transcript does not yet perform complete natural-language sentence generation.
- The login system is only a demonstration feature.

## Future Improvements

The project can be further improved through:

- Larger and more diverse datasets
- Data augmentation
- Feature normalization
- Improved neural network architectures
- Dynamic gesture recognition
- Sentence-level prediction
- Text-to-Speech integration
- Multilingual translation
- Mobile deployment

## Project Objective

The objective of Sign2Text is to demonstrate how Computer Vision, Machine Learning and Deep Learning can be combined to build a real-time assistive communication system that recognizes hand gestures and converts them into readable text.

## Developer

**Saraswati**

Computer Science / Data Science Project

## Project Highlights

- Python-based Machine Learning project
- Computer Vision
- Deep Learning
- MediaPipe Hand Tracking
- Custom Dataset
- TensorFlow/Keras Model
- OpenCV
- Real-Time Prediction
- Gesture-to-Text Conversion
- Gradio Web Interface

## License

This project is created for educational, academic and portfolio purposes.
