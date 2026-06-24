from flask import Flask, render_template, url_for, request, session, redirect
import sqlite3
import os
import secrets
import base64
import google.generativeai as genai
from googletrans import Translator
translator = Translator()

genai.configure(api_key='YOUR_API_KEY_HERE')
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
chat = gemini_model.start_chat(history=[])

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS appointments(id INTEGER PRIMARY KEY AUTOINCREMENT, 
name TEXT, doctor TEXT, 
phone TEXT, symptoms TEXT, date TEXT)"""
cursor.execute(command)


from keras.models import load_model,model_from_json
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
#from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 as cv
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from keras.layers import *
from keras.models import Sequential

mp = {'happy':'https://manybooks.net/categories','sad':'https://www.youtube.com/watch?v=F9wbogYwTVM'}

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model12 = load_model('Emotion_little_vgg.h5')
classes12 = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load the saved model architecture
json_file = open("model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the saved model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")

import secrets,os,sys,librosa,glob,flask
import tensorflow as tf
import pandas as pd
from keras.models import load_model,model_from_json
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import numpy as np
import sounddevice as sd
from keras.layers import *
from keras.models import Sequential
import numpy as np
from keras.models import load_model
import speech_recognition as sr
import sounddevice as sd
import wave
import numpy as np
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
loaded_model = load_model("Emotion_Voice_Detection_Model.h5")
c1 = ['neutral','calm','happy','surprised']
def TTS(text1):
    print(text1)
    myobj = gTTS(text=text1, lang='en-us', tld='com', slow=False)
    myobj.save("voice.mp3")
    print('\n------------Playing--------------\n')
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

def audio():
    duration = 5
    fs = 44100
    channels=2
    filename="test.wav"

    TTS("Speak...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.int16)
    sd.wait()
    print("Recording complete.")

    # Save the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())
        
    X, sample_rate = librosa.load('test.wav')
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    livedf2 = mfccs
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame()
    x = np.expand_dims(livedf2, axis=2)
    x = np.expand_dims(x, axis=0)
    print('gggggggggggggggg....')
   # predictions = loaded_model.predict_classes(x)
    predictions = np.argmax(loaded_model.predict(x), axis=-1)
    mood = convert_class_to_emotion(predictions)
    print(predictions)
    print(mood)
    return mood

def convert_class_to_emotion(pred):        
    label_conversion = {'0': 'neutral',
                        '1': 'calm',
                        '2': 'happy',
                        '3': 'sad',
                        '4': 'angry',
                        '5': 'fearful',
                        '6': 'disgust',
                        '7': 'surprised'}

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label

# Load Model
import joblib 
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl","rb"))

app = Flask(__name__)
chat_history = []
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/appoints')
def appoints():
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    query = "SELECT * FROM appointments"
    cursor.execute(query)
    results = cursor.fetchall()
    return render_template('adminlog.html', results=results)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/appointment', methods=['POST', 'GET'])
def appointment():
    if request.method == 'POST':
        data = request.form
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        print(keys, values)

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        cursor.execute("INSERT INTO appointments VALUES (NULL, ?,?,?,?,?)", values)
        connection.commit()
        return render_template('appointment.html', msg="appointment updated successfully")
    return render_template('appointment.html')

@app.route('/suggestions')
def suggestions():
    return render_template("suggestions.html")

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

@app.route('/emotion')
def emotion():
    return render_template('emotions.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        email = request.form['email']
        password = request.form['password']

        if email == 'doctor@gmail.com' and password == 'doctor123':
            query = "SELECT * FROM appointments"
            cursor.execute(query)
            results = cursor.fetchall()
            return render_template('adminlog.html', results=results)
        else:
            return render_template('admin.html', msg="Entered wrong email or password")
    return render_template('admin.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        email = request.form['email']
        password = request.form['password']

        query = "SELECT * FROM user WHERE email = '"+email+"' AND password= '"+password+"'"
        cursor.execute(query)
        result = cursor.fetchone()
        print(result)
        if result:
            return render_template('home.html')
        else:
            return render_template('signin.html', msg="Entered wrong email or password")
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        query = "SELECT * FROM user WHERE email = '"+email+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            return render_template('index.html', msg='email already exists, try with defferent name')
        else:
            cursor.execute("INSERT INTO user VALUES (NULL, '"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
            connection.commit()
            return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signup.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['query']
        print(user_input)

        gemini_response = chat.send_message(user_input)
        data = gemini_response.text
        data = data.replace("html", "")
        data = data.replace("", "")
        
        chat_history.append([user_input, data])

        return render_template('chatbot.html', chat_history=chat_history)
    
    return render_template('chatbot.html')


@app.route('/faceemotion')
def faceemotion():
    final_label = None
    cap = cv.VideoCapture(0)
    got = False
    while True:
        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)
            
            if(np.sum([roi_gray])!=0):
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
                preds = model12.predict(roi)[0]
                label = classes12[preds.argmax()]
                label_position = (x,y)
                final_label = label
                # got = True
                # break
                cv.putText(frame,label,label_position,cv.FONT_HERSHEY_COMPLEX,2,(0,255,0))
            else:
                cv.putText(frame,'No Face Found',(20,60),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255))
        # if got:
        #     break
        cv.imshow('Emotion Detector',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    # print("Done")
    print(final_label)
    if final_label in ('Happy','Neutral','Surprise'):
        return render_template("happy.html")
    else:
        return render_template("sad.html")


@app.route('/textemotion',methods=['GET','POST'])
def textemotion():
    if request.method == 'POST':
        get_sentence = request.form['query']
        from_lang = request.form['lang']

        to_lang = 'en'

        text_to_translate = translator.translate(get_sentence,src= from_lang,dest= to_lang)
        input_text = text_to_translate.text
        print(input_text)

        emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "shame", "surprise"]

        pred = list(pipe_lr.predict_proba([input_text])[0])
        preds = []
        
        for i in pred:
            ii = int(i*100)
            preds.append(ii)
        text = emotions[preds.index(max(preds))]

        print(text)
        if text in ('joy','love','surprise','happy'):
            return render_template("happy.html")
        else:
            return render_template("sad.html")
    return render_template('emotions.html')

@app.route("/speechemotion")
def speechemotion():
    mood = audio()
    print()
    if mood in c1:
        return render_template("happy.html")
    else:
        return render_template("sad.html")

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
