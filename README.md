# VocalFluencyEvaluator
The purpose of this project is to allow english learners to test their pronunciation of commonly used words and sentences, and rating their pronunciation from 0-2, with 0 being a novice speaker and 2 being an expert speaker.

STATUS: In Progress https://levelup.gitconnected.com/5-killer-python-libraries-for-audio-processing-ddef703e3d84

Model will evaluate how fluent a speaker’s speech is
The model will listen to the speaker as they speak into the mic. It will then process the data and based on its training return a value of 0-2, with 0 being a novice speaker and 2 being a fluent speaker. Usually judging vocal fluency is done manually by people and lacks objectivity.	
This algorithm should do the job of grading the audio for the user.
Topic: Vocal Fluency Evaluation Using Machine Learning
The purpose of this project is to allow english learners to test their pronunciation of commonly used words and sentences, and rating their pronunciation on a scale of 0-2 in order to effectively allow them to improve their fluency

Description: 
The purpose of this project is to allow english learners to test their pronunciation of commonly used words and sentences, and rating their pronunciation from 0-2, with 0 being a novice speaker and 2 being an expert speaker
Overview: a machine learning algorithm that allows new English learners to practice their pronunciation by rating their vocal fluency
The project will help learners practice and improve their pronunciation skills. The main objective is to develop a machine learning algorithm that can accurately assess the pronunciation of a particular phrase or sentence against a reference audio and provide feedback to the learners.
The system will utilize a large database of audio recordings from Native English speakers pronouncing these words accurately. The algorithm will analyze the pronunciation of the learner's input by comparing it to the reference audio samples. It will evaluate various aspects such as phonetic accuracy, stress patterns, intonation, and overall fluency.


Baseline - ML algorithm: 
Use a CNN to identify patterns within a spectrogram representation of audio input and extract characteristics such as pitch, tone, raspiness
An RNN (LSTM) can also be used to better evaluate patterns in timing and pronunciation
Inconsequential attributes such as noise, tone (for English or non-tonal languages), pitch, and audio quality and all can be isolated from the “important” attributes (pronunciation, timing, stress) by comparison with sample pronunciation of “confident” words
Determine accuracy and provide a grade
Problems: 
Bad audio quality and other issues caused by different audio input devices may cause issues with 
Have to consider accents and other “complicated” patterns and conditions such as lisp, stuttering, etc
Difficult to fully isolate normally harmless aspects of language (such as tone) from more “harmful” (to vocal fluency) aspects (such as pronunciation)
Harder to factor in emotion or tonal transitions between words
Data: 
We will grab audio from various datasets for commonly used words in the English language and add our data if necessary
https://www.oed.com/ - Different Accents and all
https://speechling.com/dictionary/english/all - Longer phrases, questions, expressions
https://www.dialectsarchive.com/ - Longer phrases in various accents, less specific/rudimentary phrases
https://www.english-corpora.org/coca/ (?)
Steps:
Feature extraction
Classification
Output 

Libraries:
Librosa - feature extraction from audio and allows for spectrogram plotting
Kaldi - Voice Recognition software open-source can be used for speaker characteristics identification (if Librosa not comprehensive enough)
Lium SD - Same as Kaldi
DeepSpeaker - Audio embedding extraction and analysis through Siamese NN (if Librosa not comprehensive enough)
Pydub - play, split, merge, edit our . wav audio files
Keras/TF - Advanced generalized NN operations
SKLearn - Backbone for basic NN operations
Pandas - Matrix List DB operations
Seaborn - Visualization
Matplotlib.pyplot - Visualization
Notes: 
Use SVM/RF? -> Highest % correct



Methodology: 
PRIOR INFORMATION: 
https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d#:~:text=The%20mel%20frequency%20cepstral%20coefficients,shape%20of%20a%20spectral%20envelope.&text=%3D'time')-,.,calculated%20on%20how%20many%20frames. ← ref1
https://ai.googleblog.com/2019/11/spice-self-supervised-pitch-estimation.html ← ref2
https://github.com/agrija9/Avalinguo-Dataset-Speaker-Fluency-Level-Classification-Paper-/blob/master/arXiv%20paper/1808.10556.pdf ← ref3
Speech Emotion Recognition (Sound Classification) | Deep Learning | Python ← ref4
Audio Data Processing in Python ← ref5
Extract Features from Audio File | MFCC | Python ← ref6
https://link.springer.com/chapter/10.1007/978-3-642-25020-0_24 ←ref7
Mel Frequency Cepstral Coefficients (MFCC) Explained ← ref8
https://youtube.com/playlist?list=PLCEzvGfhGbdB_mFduiQMPIMkxf34i4v0y&feature=share8 ←ref1002021398128
Gather input: 
Have the user input optimal audio clip/sample/provide optimal sample save as .wav (?)
Have the user input their input whatever save as .wav
Convert both clips into spectrogram representations -> LIBROSA display.specshow
Use librosa library for audiospectogram feature sxrtracoitn (ASFE) [ref1]
Use a modified CNN to analyze trends/patterns in the spectrogram 	
