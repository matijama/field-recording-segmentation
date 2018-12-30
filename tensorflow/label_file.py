import tensorflow.contrib.slim as slim
import numpy as np
import configparser
import math
import glob
import os

from tf_readwrite import *
from model import residual_model, residual_parameters
from generate_features import calc_features


def main(_):
    config = configparser.ConfigParser()
    config.read('defaults.ini')

    log_dir = config['Folders']['log_dir']

    checkpoint_dir = os.path.join(log_dir, 'MODEL_FOLDER_NAME')
    do_mel = 0

    filenames=glob.glob(os.path.join(config['Folders']['db_root'],'sessions/*.mp3'))

    for file_name in filenames:

        if (os.path.isfile(file_name + '.score.csv')):
            continue

        print(file_name)

        tf.reset_default_graph()

        D, sr = get_input_spectrogram(file_name, do_mel, config)

        label_step_size = sr / config.getfloat('Score','label_samplerate')

        block_len = int(config.getfloat('Network','block_length_sec') * sr)

        with tf.Graph().as_default():

            x = tf.placeholder(dtype=tf.float32, name="data", shape=(None, D.shape[0], block_len, 1))
            with slim.arg_scope(residual_parameters(weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, reg=3,elu=True)):
                # this should be the same as for training
                logits, predictions = residual_model(x, 4, is_training=False, reuse=False,
                                                     is_music_layer1=True, n_filt=6, n_res_lay=4, layer1_size=10, use_max_pool=[2, 2, 2, 2], use_avg_pool=False, use_atrous=False,
                                                     reg=3, weight_decay=0.00004)

            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            saver = tf.train.Saver()
            with tf.Session() as sess:

                saver.restore(sess, checkpoint_path)

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
                    all_pred = sess.run(predictions, feed_dict={'data:0': inp})

                    if (preds is None):
                        preds = np.zeros((math.ceil(D.shape[1] / label_step_size), all_pred.shape[1]))
                    preds[j:j+k+1, :] = all_pred
                    j += k+1

        preds = preds[:j, :]

        with open(file_name + '.score.csv', 'wt') as csvfile:
            for i in range(0,preds.shape[0]):
                csvfile.write(np.array2string(preds[i,:],separator='\t').replace('[','').replace(']','')+'\n')



def get_input_spectrogram(file_name, do_mel, config):

    D, freqs, sr = calc_features(file_name, do_mel=do_mel)

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.getint('Spectrogram','mel_bands'),
        num_spectrogram_bins=D.shape[1],
        sample_rate=config.getint('Spectrogram','sample_rate'),
        lower_edge_hertz=config.getfloat('Spectrogram','mel_min'),
        upper_edge_hertz=config.getfloat('Spectrogram','mel_max'))

    D = tf.tensordot(D, linear_to_mel_weight_matrix, 1)
    D = tf.math.log(D + 0.00001)
    D = tf.transpose(D,[1,0])

    with tf.Session() as sess:
        Do=sess.run(D)

    return Do, sr

if __name__ == '__main__':
    tf.app.run()
