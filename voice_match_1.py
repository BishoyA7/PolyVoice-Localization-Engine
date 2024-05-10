import numpy as np 
import librosa 
import matplotlib.pyplot as plt 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import os

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to convert WAV file to Mel spectrogram to PNG
def audio_to_mel_to_png(wav_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    start_time = time.time()

    y, sr = librosa.load(wav_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    
    fig.savefig(image_file)
    plt.close(fig)

    end_time = time.time()
    processing_time = end_time - start_time
    # print("Audio conversion and image creation time:", processing_time, "seconds")

    return image_file

def preprocess_image(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return image

# Function to classify gender
def classify_gender(image_file, gender_classifier, base_model):
    start_time = time.time()

    image = preprocess_image(image_file)
    features = base_model.predict(np.expand_dims(image, axis=0))
    gender = gender_classifier.predict(features)

    gender_class_index = np.argmax(gender)

    end_time = time.time()
    processing_time = end_time - start_time
    print("Gender classification time:", processing_time, "seconds")
    print("Gender probablity:", gender)

    return 'Male' if gender_class_index == 1 else 'Female'

# Function to classify emotion based on gender
def classify_emotion(image_file, emotion_classifer, base_model):
    start_time = time.time()

    image = preprocess_image(image_file)
    features = base_model.predict(np.expand_dims(image, axis=0))
    emotion_probs = emotion_classifer.predict(features)
    emotion_label = np.argmax(emotion_probs)
    emotion_labels = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    end_time = time.time()
    processing_time = end_time - start_time
    # print("Emotion classification time:", processing_time, "seconds")

    return emotion_labels[emotion_label]

def get_audio_path(classification_result, target_language):
    gender, emotion = classification_result.split('_')
    base_path = '/content/gdrive/MyDrive/Preset Voice Library 1'
    language_labels = ['English', 'French', 'German', 'Greek', 'Mandarin', 'Persian']

    for language in language_labels:
        if target_language == language:
            language_path = os.path.join(base_path, language)
    
    if gender == 'Male':
        gender_path = os.path.join(language_path, 'Male')
    else: 
        gender_path = os.path.join(language_path, 'Female')

    emotion_path = os.path.join(gender_path, emotion)
    audio_files = os.listdir(emotion_path)
    audio_file = os.path.join(emotion_path, audio_files[0])

    return audio_file


# # Load classifiers 
# gender_classifier = load_model('/content/gdrive/MyDrive/best_model_1.h5')
# gender_classifier.layers[0] = Input(shape=(224, 224, 3))


# male_emotion_classifier = load_model('/content/gdrive/MyDrive/male_model.h5')
# male_emotion_classifier.layers[0] = Input(shape=(224, 224, 3))


# female_emotion_classifier = load_model('/content/gdrive/MyDrive/female_model_3.h5')
# female_emotion_classifier.layers[0] = Input(shape=(224, 224, 3))


def classify_wav(wav_file):
    start_time = time.time()

    mel_spec = audio_to_mel_to_png(wav_file, 'mel_spec.png')

    gender = classify_gender('mel_spec.png', gender_classifier, base_model)

    if gender == 'male':
        emotion = classify_emotion('mel_spec.png', male_emotion_classifier, base_model)
    else:
        emotion = classify_emotion('mel_spec.png', female_emotion_classifier, base_model)

    end_time = time.time()
    total_time = end_time - start_time
    # print("Total processing time:", total_time, "seconds")

    return f'{gender}_{emotion}'


def classify_multilingual_dataset(dataset_path):
    results = {'Correct': 0, 'Total': 0}
    # for gender in os.listdir(dataset_path):
    for speaker in os.listdir(dataset_path):
        # gender_path = os.path.join(dataset_path, gender)
        actor, gender = speaker.split(' - ')
        speaker_path = os.path.join(dataset_path, speaker)
        # for emotion in os.listdir(gender_path):
        for emotion in os.listdir(speaker_path):
            # emotion_path = os.path.join(gender_path, emotion)
            emotion_path = os.path.join(speaker_path, emotion)
            for audio in os.listdir(emotion_path): 
                if audio.endswith(".wav"):
                    wav_file = os.path.join(emotion_path, audio)
                    classification_result = classify_wav(wav_file)
                    actual_label = f'{gender}_{emotion}'
                    # predicted_gender, predicted_emotion = classification_result.split('_')
                    # actual_gender, actual_emotion = actual_label.split('_')
                    if classification_result == actual_label:
                        results['Correct'] += 1
                    results['Total'] += 1 
        print('Language:', os.path.basename(dataset_path))
        print('Results:', results)               
    return results 

def plot_results(results):
    languages = list(results.keys())
    accuracy = [results[lang]['Correct'] / results[lang]['Total'] for lang in languages]

    plt.figure(figsize=(10,6))
    plt.bar(languages, accuracy, color='skyblue')
    plt.xlabel('Languages')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Score of Multilingual Classification')
    plt.ylim(0, 1)
    plt.show()
