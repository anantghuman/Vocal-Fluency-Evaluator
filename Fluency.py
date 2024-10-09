import pyaudio
import wave
import librosa
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from os import path
import os
from pydub import AudioSegment
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import scipy
from scipy.io import wavfile


def user_input():
  while True:
    while True:
      filename = input("Enter the name of the audio file and its location(.wav or .mp3): ")
      ext = filename.split(".")[-1]
      name = filename.split(".")[0]

      if ext == "wav" or ext == "mp3":
        break
    try:
      with open(filename, 'r') as file:
          if ext == "mp3":
            sound = AudioSegment.from_mp3(filename)
            sound.export(name + ".wav", format="wav")
          return name + ".wav"
    except FileNotFoundError:
      print("File not found.")
    except IOError:
      print("Error while opening the file.")


def get_audio_rec(file_name):
  chunk = 1024  # Record in chunks of 1024 samples
  sample_format = pyaudio.paInt16  # 16 bits per sample
  channels = 2
  sr = 44100  # Record at 44100 samples per second
  seconds = 5
  p = None
  while True:
    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sr,
                    frames_per_buffer=chunk,
                    input=True)

    for i in range(0, int(sr / chunk * seconds)):
      data = stream.read(chunk)
      frames.append(data)



    stream.stop_stream()
    stream.close()
    p.terminate()
  write(file_name, sr, p)
  return file_name

import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
def get_audio_features(audio_path):
  x, sample_rate = librosa.load(audio_path)
  mfcc = np.mean(librosa.feature.mfcc(y = x, sr = sample_rate, n_mfcc = 20).T, axis=0)
  # rmse = np.mean(librosa.feature.rms(y=x).T, axis=0)
  # spectral_flux = np.mean(librosa.onset.onset_strength(y=x, sr=sample_rate).T, axis=0) #Spectral Flux (Stanford's). Returns 1 Value
  #spec_contrast = np.mean(librosa.feature.spectral_contrast(y=x).T, axis=0) #Returns 1 value

  plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()
  #return mfcc #, rmse, spectral_flux, zcr

def show_spectogram(audio_file):
  y, shape = librosa.load(audio_file)
  #ytrim, new_shape = librosa.effects.trim(y)
  X = librosa.stft(y)

  Xdb = librosa.amplitude_to_db(abs(X))

  plt.figure(figsize=(14, 5))
  librosa.display.specshow(Xdb, sr = 44100, x_axis = 'time', y_axis = 'hz')
  plt.colorbar()

def get_spectogram(audio_file):
  y, shape = librosa.load(audio_file)
  ytrim, new_shape = librosa.effects.trim(y)
  X = librosa.stft(ytrim)
  Xdb = librosa.amplitude_to_db(abs(X))
  return Xdb

if __name__=="__main__":
    # hello = user_input()
    # b = show_spectogram(hello)
    a = get_audio_features("/content/air_raid.wav")

# if __name__=="__main__":
#     #hello = user_input()
#     b = get_audio_features("/content/Avalinguo_Manuel_and_Dayana segment 121.mp3")
#     # a = show_spectogram("/content/Med 2.mp3")
    # b = show_spectogram("/content/High.mp3")

# CNN = models.Sequential()
# CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# CNN.add(layers.MaxPooling2D((2, 2)))
# CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
# CNN.add(layers.MaxPooling2D((2, 2)))
# CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
# CNN.add(layers.Flatten())
# CNN.add(layers.Dense(64, activation='relu'))
# CNN.add(layers.Dense(10))

from tensorflow.python.framework.ops import name_from_scope_name
def get_features(file_path):
  #change to mp3
  ext = "*.wav"
  sub_dir = os.listdir(file_path)
  sub_dir.remove(".ipynb_checkpoints")
  sub_dir.sort()
  print(sub_dir)
  features, labels = np.empty((0,20)), np.array([])
  for label, sub_dir in enumerate(sub_dir):
    for file_name in glob.glob(os.path.join(file_path, sub_dir, ext)):
      #mfcc, rmse, spectral_flux, zcr = get_audio_features(file_name)
      mfcc = get_audio_features(file_name)
      #mfcc = np.array(mfcc)
      #print(mfcc)
      #uv = np.hstack([mfcc, rmse, spectral_flux, zcr])
      #print(uv.shape)
      features = np.vstack([features, mfcc])
      labels = np.append(labels, label)
  return features, labels

# def get_spec_values(file_path):
#   ext = "*.wav"
#   sub_dir = os.listdir(file_path)
#   sub_dir.remove(".ipynb_checkpoints")
#   sub_dir.sort()

#   features, labels = np.empty((1025,1)), np.array([])
#   for label, sub_dir in enumerate(sub_dir):
#     for file_name in glob.glob(os.path.join(file_path, sub_dir, ext)):
#       spec = get_spectogram(file_name)
#       spec = np.array(spec)
#       print(spec.shape)
#       #print(features.shape)
#       #features = np.vstack([features, spec])
#       labels = np.append(labels, label)
#   return features, labels

from librosa.core.audio import load
# CNN.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
if __name__=="__main__":
 features, labels = get_features("/content/Data")
 print(labels)

import scipy.io.wavfile
def standardize_duration(audio_path, desired_duration_seconds):
  duration = get_duration(audio_path)
  print(duration)
  audio = AudioSegment.from_mp3(audio_path)
  if duration > desired_duration_seconds:
    dur = desired_duration_seconds * 1000
    audio = audio[:dur]
  else:
    len = desired_duration_seconds - duration
    print(len)
    silence = AudioSegment.silent(duration= len * 1000)
    audio = audio + silence
  file_handle = audio.export("/content/Test/Blank.wav", format="mp3")


def get_duration(audio_path):
  sample_rate, data = wavfile.read(audio_path)
  duration = len(data) / sample_rate
  return duration

if __name__=="__main__":
 standardize_duration("/content/Test/Blank.wav", 20)
