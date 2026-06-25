# 3-Way Emotion Detection System

A deep learning web application that detects human emotions through 
three different input methods — facial expressions, speech, and text.

## 🔍 Features

- **Face Emotion Detection** — uses webcam and CNN (VGG architecture) 
  to detect 7 emotions in real time
- **Speech Emotion Detection** — records voice and analyzes tone using 
  MFCC features to classify emotions
- **Text Emotion Detection** — analyzes typed text using NLP and 
  Logistic Regression to identify emotions
- **AI Chatbot** — integrated Gemini AI chatbot for mental health support
- **Appointment Booking** — users can book doctor appointments
- **Multilingual Support** — text input supports multiple languages 
  via Google Translate API

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras
- **Models:** CNN (VGG), LSTM, Logistic Regression
- **Audio Processing:** Librosa, SoundDevice
- **Computer Vision:** OpenCV, HaarCascade
- **AI Integration:** Google Gemini API
- **Database:** SQLite

## 🧠 Models Used

- `Emotion_little_vgg.h5` — facial emotion detection (7 classes)
- `Emotion_Voice_Detection_Model.h5` — speech emotion detection
- `emotion_classifier_pipe_lr.pkl` — text emotion classification
- `model_weights.h5` — text emotion deep learning weights

## 🚀 How to Run

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python app.py
4. Open browser at: http://localhost:5000

## 👨‍💻 Author

Mohammed Fazlian  
Cybersecurity & AI Enthusiast  
www.linkedin.com/in/mohammed-fazlian-86459b197
mdfazlian30@gmail.com
