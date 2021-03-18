import parselmouth
import numpy as np
from os import listdir
import re
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import python_speech_features
import librosa
import librosa.display


def get_features(path):
    sound = parselmouth.Sound(path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03,
                                          0.45)
    voice_report = voice_report.split('\n')

    index_list = [0, 1, 7, 12, 16, 22, 29, 33]  # Fejlécek

    vr = []
    for index, element in enumerate(voice_report):
        if index not in index_list:
            vr.append(element)

    numbers = []
    for i in vr:
        numbers += ([float(ele) for ele in re.findall(r"[-+]?\d*\.\d+|\d+", i)])

    numbers[7] = numbers[7] * 10 ** ((-1) * numbers[8])
    numbers[9] = numbers[9] * 10 ** ((-1) * numbers[10])
    numbers[19] = numbers[19] * 10 ** ((-1) * numbers[20])

    index_list = [8, 10, 20, 12, 13, 16, 17, 22, 27, 29, 31]  # Szükségtelen számok

    final_vr = []
    for index, element in enumerate(numbers):
        if index not in index_list:
            final_vr.append(element)

    mfcc = get_MFCC(path)

    return final_vr+mfcc


def get_spectrogram_praat(path):
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    snd = parselmouth.Sound(path)
    spectrogram = snd.to_spectrogram()
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 80, cmap='Greys_r')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.axis('off')
    fig.savefig('{}.png'.format(path))
    return "{}.png".format(path)


def get_spectrogram(path):
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    rate, sig = wavfile.read(path)
    frequencies, times, spectrogram = signal.spectrogram(sig, rate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram), vmin=np.log(spectrogram).max() - 23, cmap='Greys_r')
    plt.axis('off')
    fig.tight_layout(pad=0)  # fehér keret eltüntetése
    fig.savefig('{}.png'.format(path))
    return "{}.png".format(path)


def get_melspectrogram(path):
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    y, sr = librosa.load(path)
    sound, _ = librosa.effects.trim(y)
    S = librosa.feature.melspectrogram(y=sound, sr=sr, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap='gray')
    plt.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig('{}.png'.format(path))
    return "{}.png".format(path)


def get_MFCC(path):
    rate, sig = wavfile.read(path)
    mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
    delta = python_speech_features.delta(mfcc, 2)
    delta_delta = python_speech_features.delta(delta, 2)

    m_mfcc = np.mean(mfcc, axis=0)
    m_delta = np.mean(delta, axis=0)
    m_delta_delta = np.mean(delta_delta, axis=0)
    return np.concatenate((m_mfcc, m_delta, m_delta_delta)).tolist()


healthy = {
    'AGNESE P': ["F", 69],
    'ANGELA C': ["F", 68],
    'ANGELA G': ["F", 63],
    'ANTONIETTA P': ["F", 61],
    'ANTONIO C': ["M", 77],
    'BRIGIDA C': ["F", 69],
    'GILDA C': ["F", 65],
    'GIOVANNA G': ["F", 70],
    'GIOVANNI B': ["M", 69],
    'GRAZIA G': ["F", 72],
    'LEONARDA F': ["F", 60],
    'LISCO G': ["M", 60],
    'LUIGI P': ["M", 76],
    'MARIACRISTINA P': ["F", 61],
    'MICHELE G': ["F", 68],
    'NICOLA P': ["M", 75],
    'PORCELLI A': ["M", 68],
    'SUMMO L': ["F", 69],
    'TERESA M': ["F", 63],
    'VITANTONIO D': ["M", 70],
    'VITO A': ["M", 68],
    'VITO L': ["M", 62]}

pd = {
    'Anna B': ["F", 71],
    'Antonia G': ["F", 65],
    'Daria L': ["F", 80],
    'Domenico C': ["M", 50],
    'Felicetta C': ["F", 63],
    'Giovanni M': ["M", 73],
    'Giovanni N': ["M", 70],
    'Giulia L': ["F", 67],
    'Giulia P': ["F", 54],
    'Giustina M': ["F", 78],
    'Leonarda L': ["F", 61],
    'Lucia R': ["F", 40],
    'Luigi B': ["M", 65],
    'Mario B': ["M", 72],
    'Michele C': ["M", 71],
    'Nicola M': ["M", 65],
    'Nicola S': ["M", 73],
    'Nicola S2': ["M", 73],
    'Nicolo C': ["M", 65],
    'Roberto L': ["M", 75],
    'Roberto R': ["M", 68],
    'Roberto R2': ["M", 68],
    'Saverio S': ["M", 56],
    'Ugo B': ["M", 77],
    'Vito L': ["M", 70],
    'Vito S': ["M", 71],
    'Vito S2': ["M", 71],
    'Vito S3': ["M", 70]}


def create_df():
    filename = "sound_features.csv"

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")

        for name in listdir("dataset/healthy"):
            list = listdir("dataset/healthy/{}".format(name))

            features1 = []
            features2 = []
            for vocal in ['A', 'E', 'I', 'O', 'U']:
                sound_file1 = [filename for filename in list if filename.startswith("V{}1".format(vocal))][0]
                current_feature1 = get_features("dataset/healthy/{}/{}".format(name, sound_file1))
                features1 += current_feature1

                location1 = get_melspectrogram("dataset/healthy/{}/{}".format(name, sound_file1))
                features1.append(location1)

                sound_file2 = [filename for filename in list if filename.startswith("V{}2".format(vocal))][0]
                current_feature2 = get_features("dataset/healthy/{}/{}".format(name, sound_file2))
                features2 += current_feature2

                location2 = get_melspectrogram("dataset/healthy/{}/{}".format(name, sound_file2))
                features2.append(location2)

            features1 += (healthy[name][0], healthy[name][1], 0)
            features2 += (healthy[name][0], healthy[name][1], 0)

            csvwriter.writerows([features1, features2])

        for name in listdir("dataset/pd"):
            list = listdir("dataset/pd/{}".format(name))

            features1 = []
            features2 = []
            for vocal in ['A', 'E', 'I', 'O', 'U']:
                sound_file1 = [filename for filename in list if filename.startswith("V{}1".format(vocal))][0]
                current_feature1 = get_features("dataset/pd/{}/{}".format(name, sound_file1))
                features1 += current_feature1

                location1 = get_melspectrogram("dataset/pd/{}/{}".format(name, sound_file1))
                features1.append(location1)

                sound_file2 = [filename for filename in list if filename.startswith("V{}2".format(vocal))][0]
                current_feature2 = get_features("dataset/pd/{}/{}".format(name, sound_file2))
                features2 += current_feature2

                location2 = get_melspectrogram("dataset/pd/{}/{}".format(name, sound_file2))
                features2.append(location2)

            features1 += (pd[name][0], pd[name][1], 1)
            features2 += (pd[name][0], pd[name][1], 1)

            csvwriter.writerows([features1, features2])

        print("Dataset created")


create_df()

#path="C:/Users/Peti/Desktop/DeepLearning/Diploma/PD-detection/dataset/healthy/AGNESE P/VA1APGANRET55F170320171107.wav"
