import tensorflow as tf
import librosa
import numpy as np
from tf_readwrite import dataset_files_labels, write_features


def calc_features(fileName, duration=None):
    y, sr = librosa.load(fileName, sr=22050, mono=True, duration=duration)
    y = y / max(abs(y)) * 0.9

    fftsize = 1024
    stepsize = 315

    minF = 0
    maxF = 11025

    D = np.abs(librosa.stft(y, n_fft=fftsize, hop_length=stepsize))
    freqs = librosa.fft_frequencies(sr=22050, n_fft=fftsize)
    D = D[np.logical_and(freqs >= minF, freqs <= maxF), :]
    freqs = freqs[np.logical_and(freqs >= minF, freqs <= maxF)]
    D = D.transpose()

    return D,freqs,sr/stepsize


def generate_features():

    folders = ['LIST_OF_FOLDERS_TO_PROCESS']
    fl = dataset_files_labels(folders)

    for (fileName, label) in fl:
        features,freqs,sr=calc_features(fileName)
        write_features(features, label, fileName + '.fft.tfr')


def print_features(do_mel=3):

    folders = ['gni']
    # folders = ['gtzan']
    fl = dataset_files_labels(folders)

    fl=[('C:\\Users\\matic\\Research\\Databases\\transcription\\piano\\piano keys normalized\\BOE_LOUD\\A4.BOE_LD.WAV','i')]
    for (fileName, label) in [x for x in fl if x[1]=='i']:
        features,freqs,sr=calc_features(fileName)
        oneframe = np.transpose(features[10:106,:,1])
        np.savetxt('oneframe.csv',oneframe,delimiter='\t')
        break


def main(_):
    generate_features()


if __name__ == '__main__':
    tf.compat.v1.app.run()
