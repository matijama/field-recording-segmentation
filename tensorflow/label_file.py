import tf_slim as slim
import configparser
import math
import glob
import os
from pathlib2 import Path

from tf_readwrite import *
from model import residual_model, residual_parameters
from generate_features import calc_features


def main(_):
    config = configparser.ConfigParser()
    config.read('defaults.ini')

    checkpoint_dir = config['Folders']['trained_model_dir']

    # change folder name to contain files to label
    labels_dir = Path(os.path.join(config['Folders']['db_root']))
    filenames = labels_dir.rglob('*.mp3')

    for filename in filenames:
        file_name = str(filename)
        print(file_name)

        if os.path.isfile(file_name + '.score.csv'):
            continue

        print(file_name)

        tf.compat.v1.reset_default_graph()

        D, sr = get_input_spectrogram(file_name, config, False)
        D=np.transpose(D)

        label_step_size = sr / config.getfloat('Score','label_samplerate')
        block_len = int(config.getfloat('Network','block_length_sec') * sr)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            tf.compat.v1.saved_model.loader.load(sess, ['scoring-tag'], checkpoint_dir)
            predictions = tf.compat.v1.get_default_graph().get_tensor_by_name("predictions:0")

            j = 0
            i = 0.0
            preds = None

            # score batches of 200 frames, add them to preds array
            while i < D.shape[1] - block_len:

                inp=np.zeros([200,D.shape[0],block_len,1])
                for k in range(0,200):
                    inp[k,:,:,0]=D[:, int(i):int(i)+block_len]
                    i = i + label_step_size
                    if i>D.shape[1] - block_len:
                        break

                inp=inp[:k+1,:,:,:]
                all_pred = sess.run(predictions, feed_dict={'xinput:0': inp})
                all_pred  = np.round(all_pred, 2)

                if (preds is None):
                    preds = np.zeros((math.ceil(D.shape[1] / label_step_size), all_pred.shape[1]))
                preds[j:j+k+1, :] = all_pred
                j += k+1

        preds = preds[:j, :]

        with open(file_name + '.score.csv', 'wt') as csvfile:
            for i in range(0,preds.shape[0]):
                csvfile.write(np.array2string(preds[i,:],separator='\t').replace('[','').replace(']','')+'\n')


def get_input_spectrogram(file_name, config, return_mel=True):

    D, freqs, sr = calc_features(file_name)

    if return_mel:
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=config.getint('Spectrogram','mel_bands'),
            num_spectrogram_bins=D.shape[1],
            sample_rate=config.getint('Spectrogram','sample_rate'),
            lower_edge_hertz=config.getfloat('Spectrogram','mel_min'),
            upper_edge_hertz=config.getfloat('Spectrogram','mel_max'))

        D = tf.tensordot(D, linear_to_mel_weight_matrix, 1)
        D = tf.math.log(D + 0.00001)

        with tf.Session() as sess:
            Do=sess.run(D)
    else:
        Do=D
    return Do, sr


if __name__ == '__main__':
    tf.compat.v1.app.run()
