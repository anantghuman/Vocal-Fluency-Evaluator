import glob
import os

import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pvrecorder import PvRecorder
import wave, struct

# Preprocessing and feature extraction
def compute_spectrogram(audio_input):
    spectrogram = librosa.feature.melspectrogram(y=audio_input, sr=44100)  # adjust sample rate SR as needed
    return spectrogram

def split_spectrogram_into_frames(spectrogram, frame_length=100):  # adjust frame length as needed
    frames = []
    num_frames = spectrogram.shape[1] // frame_length
    for i in range(num_frames):
        frame = spectrogram[:, i * frame_length : (i + 1) * frame_length]
        frames.append(frame)
    return frames

def get_features(file_name):
    x, sample_rate = librosa.load(file_name)
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=20).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sample_rate).T, axis=0)
    spectrogram = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=x).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=x).T, axis=0)
    sf = np.mean(librosa.feature.spectral_flatness(y=x).T, axis=0)
    #pyin = np.mean(librosa.pyin(y=x, sr=sample_rate, fmin = 65, fmax = 2093), axis=0)
    # print(spectrogram.shape)
    # print(spectrogram)
    # print("mfcc ", mfcc.shape)
    return [mfcc, spec_contrast, zcr, rmse, sf, spectrogram]
    #return [mfcc, spec_contrast, zcr, rmse, sf, pyin, spectrogram]

def get_features_and_labels(file_path):
    # change to mp3
    ext = "*.mp3"
    sub_dir = os.listdir(file_path)
    sub_dir.remove(".ipynb_checkpoints")
    sub_dir.sort()
    #print(sub_dir)
    features, labels = np.empty((0, 158)), np.array([]) # change 30 to 374 if you are using pyin and spectrogram
    for label, sub_dir in enumerate(sub_dir, start = 0):
        for file_name in glob.glob(os.path.join(file_path, sub_dir, ext)):
            feat = get_features(file_name)
            feat = np.hstack(feat)
            #print("feat ",feat.shape)
            features = np.vstack([features, feat])
            labels = np.append(labels, label)
    return features, labels

# SVM classification
# def train_svm(train_data, train_labels):
#     svm_classifier = SVC(kernel='linear')
#     svm_classifier.fit(train_data, train_labels)
#     return svm_classifier

# def svm_prediction(svm_classifier, test_data):
#     svm_predictions = svm_classifier.predict(test_data)
#     return svm_predictions

# LSTM seq modeling
# def train_lstm(train_data, train_labels):
#     lstm_model = Sequential()
#     lstm_model.add(LSTM(units=64, input_shape=train_data.shape[1:]))
#     lstm_model.add(Dense(1, activation='sigmoid'))
#     lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     lstm_model.fit(train_data, train_labels, epochs=2, batch_size=32)  # adjust epoch/batch_size as needed
#     return lstm_model

# def lstm_prediction(lstm_model, svm_predictions):
#     lstm_predictions = lstm_model.predict(svm_predictions)
#     lstm_predictions = (lstm_predictions > 0.5).astype(int)  # conv probability to binary prediction
#     return lstm_predictions

def split_and_label_data(feature_vectors, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=42
    )
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def evaluate_model(predictions, test_labels):
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def train_svm(train_data, train_labels):
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(train_data, train_labels)
    return svm_classifier

def train_lstm(train_data, train_labels, test_data_lstm, test_labels_lstm):
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_shape=(train_data.shape[1], train_data.shape[2])))
    lstm_model.add(Dense(3, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(train_data, train_labels, validation_data=(test_data_lstm, test_labels_lstm), epochs=10, batch_size=32)
    return lstm_model

def prepare_data_for_lstm(spectrogram_frames):
    a,b = spectrogram_frames.shape
    return np.reshape(spectrogram_frames,(a,b,1))

def main():
    file_path = "/content/DATA/"
    feature_vectors, labels = get_features_and_labels(file_path)
    #print(feature_vectors.shape)
    train_data, test_data, train_labels, test_labels = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=42
    )

    # train SVM
    svm_classifier = train_svm(train_data, train_labels)
    svm_predictions = svm_classifier.predict(test_data)

    sequence_length = 10
    lstm_data = prepare_data_for_lstm(train_data)
    print(lstm_data.shape)

    lstm_test = prepare_data_for_lstm(test_data)
    # train LSTM
    lstm_model = train_lstm(lstm_data, train_labels, lstm_test, test_labels)
    lstm_predictions = lstm_model.predict(test_data)

    # averaging
    combined_predictions = (svm_predictions + lstm_predictions) / 2
    final_predictions = np.argmax(combined_predictions, axis=1)

    # convert one-hot encoded test labels to integer
    test_labels = np.argmax(test_labels, axis=1)

    accuracy = accuracy_score(test_labels, final_predictions)
    print("Accuracy:", accuracy)
    return svm_classifier,lstm_model

# call to main
svm_classifier, lstm_model = main()
audio_input = audio_input("user_rec")
pred = svm_classifier, lstm_model

class audio_input:
  def __init__(self, file_name):
    self.file_name = file_name
    for index, device in enumerate(PvRecorder.get_available_devices()):
      print(f"[{index}] {device}")
    input = int(input("Enter a number for the input device"))
    recorder = PvRecorder(device_index=input, frame_length=512) #(32 milliseconds of 16 kHz audio)
    audio = []
    path = 'audio_recording.wav'

    try:
      recorder.start()


      while True:
          frame = recorder.read()
          audio.extend(frame)de
    except KeyboardInterrupt:
      recorder.stop()
      with wave.open(path, 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
    finally:
      recorder.delete()
    def get_feat(svm, lstm):
      features = get_features(self.file_name)
      features = np.hstack(features)
      svm_pred = svm.predict(features)
      lstm_data = prepare_data_for_lstm(features)
      lstm_pred = lstm.predict(lstm_data)
      combined_predictions = (svm_pred + lstm_pred) / 2
      return combined_predictions
