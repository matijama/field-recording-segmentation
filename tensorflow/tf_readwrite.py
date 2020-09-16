import csv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def dataset_files_labels(folders, db_root):
    """Read all samples and labels from files, return a list of all filename/label pairs."""
    fl = []
    for f in folders:

        fo = open(db_root + '/sample labels ' + f + '.txt', 'r')
        dialect = csv.Sniffer().sniff(fo.read(1024), delimiters="\t ")
        fo.seek(0)
        for x in csv.reader(fo, dialect):
            fl.append([db_root + '/' + f + '/' + x[0], x[1]])
    return fl


def train_test_set(types, testpercent, db_root, block_len_sec, folders, size=None):
    """
    Creates a list of files for the train and test set
    :param types: combination of labels
    :param testpercent: percentage of test cases to produce
    :param folders: list of folders to read files from
    :param size: optional, size of entire (train+test) set, or None if all should be used
    :return: train and test  lists containing tuples [folder, label], and a list of all labels
    """
    fl = dataset_files_labels(folders, db_root)

    fl = [(elem[0] + '.fft.tfr', elem[1]) for elem in fl if elem[1] in types]
    # these values are fixed in generate_features
    num_freqs=513
    step_size=315/22050

    block_len=int(block_len_sec/step_size)

    if size is None:
        size = len(fl)

    trainsize = int(round(size * (1 - testpercent)))
    testsize = size - trainsize

    if (trainsize==0):
        train=[]
        test=fl
    elif (testsize==0):
        train=fl
        test=[]
    else:
        train, test = train_test_split(fl,
                                   train_size=trainsize,
                                   test_size=testsize,
                                   stratify=[x[1] for x in fl],
                                   random_state=42)

    all_labels=list(sorted(set([x[1] for x in fl])))
    return train, test, all_labels, num_freqs, block_len


def dataset_fft_to_mel_single(tfrecord, num_timesteps, num_freqs, linear_to_mel_weight_matrix, all_labels, augment):
    """ Takes a single random sample out of each tfr"""
    context_out, feat_list_out = tf.io.parse_single_sequence_example(
        tfrecord,
        context_features={
            "label": tf.io.FixedLenFeature((1,), dtype=tf.string)
        },
        sequence_features={
            "spectrogram": tf.io.FixedLenSequenceFeature((num_freqs,), tf.float32),
        }
    )
    spectrogram_all = feat_list_out['spectrogram']
    shp = tf.shape(input=spectrogram_all)

    if augment:
        # augment by scaling and shifting
        shift=tf.random.uniform([1,1],1-0.3,1+0.3)
        scale=tf.random.uniform([1,1],1-0.3,1+0.3)
        shift_scale=tf.concat([shift, tf.zeros([1,3]),scale, tf.zeros([1, 3])],axis=1,name='shiftscale')
        spectrogram_aug = tf.cond(pred=tf.random.uniform([],0,1) < 0.5, true_fn=lambda: tf.contrib.image.transform(spectrogram_all,shift_scale,interpolation='BILINEAR',name='augment'), false_fn=lambda: spectrogram_all )
    else:
        spectrogram_aug=spectrogram_all

    # random offset into the file
    frame_offset=tf.cond(  pred=shp[0]>num_timesteps, true_fn=lambda: tf.random.uniform([1],minval=0,maxval=tf.subtract(shp[0],num_timesteps),dtype=tf.int32,seed=42),false_fn=lambda: tf.zeros([1], dtype=tf.int32))
    slice_start = tf.concat([frame_offset,[0]],0)

    slice = tf.slice(spectrogram_aug, slice_start, [num_timesteps, num_freqs])

    mel_slice = tf.tensordot(slice, linear_to_mel_weight_matrix, 1)
    mel_slice = tf.math.log(mel_slice + 0.00001)

    mel_slice = tf.expand_dims(tf.transpose(a=mel_slice), -1)

    lbl = context_out['label']

    _, idx = tf.compat.v1.setdiff1d(tf.constant(all_labels), lbl)
    idx, _ = tf.compat.v1.setdiff1d(tf.range(len(all_labels)), idx)

    return mel_slice , idx[0]


def dataset_fft_to_mel_multi(tfrecord, num_timesteps, num_freqs, linear_to_mel_weight_matrix, all_labels):
    """ Takes consequtive samples with step size out of each tfr"""
    context_out, feat_list_out = tf.io.parse_single_sequence_example(
        tfrecord,
        context_features={
            "label": tf.io.FixedLenFeature((1,), dtype=tf.string)
        },
        sequence_features={
            "spectrogram": tf.io.FixedLenSequenceFeature((num_freqs,), tf.float32),
        }
    )
    spectrogram_all = feat_list_out['spectrogram']

    mel_spect = tf.tensordot(spectrogram_all, linear_to_mel_weight_matrix, 1)
    mel_spect = tf.math.log(mel_spect + 0.00001)

    mel_spect = tf.expand_dims(mel_spect, -1)

    spectrogram_frames = tf.transpose(a=tf.signal.frame(mel_spect, num_timesteps, num_timesteps//4, axis=0),perm=[0,2,1,3])

    lbl = context_out['label']

    _, idx = tf.compat.v1.setdiff1d(tf.constant(all_labels), lbl)
    idx, _ = tf.compat.v1.setdiff1d(tf.range(len(all_labels)), idx)

    lblIndex = tf.fill([tf.shape(input=spectrogram_frames)[0]], tf.cast(idx[0], dtype=tf.int32))
    return spectrogram_frames, lblIndex


def dataset_fft_to_mel_multi_with_files(fn, tfrecord, num_timesteps, num_freqs, linear_to_mel_weight_matrix, all_labels):
    """ Takes consequtive samples with step size out of each tfr"""
    context_out, feat_list_out = tf.io.parse_single_sequence_example(
        tfrecord,
        context_features={
            "label": tf.io.FixedLenFeature((1,), dtype=tf.string)
        },
        sequence_features={
            "spectrogram": tf.io.FixedLenSequenceFeature((num_freqs,), tf.float32),
        }
    )
    spectrogram_all = feat_list_out['spectrogram']

    mel_spect = tf.tensordot(spectrogram_all, linear_to_mel_weight_matrix, 1)
    mel_spect = tf.math.log(mel_spect + 0.00001)

    mel_spect = tf.expand_dims(mel_spect, -1)

    spectrogram_frames = tf.transpose(a=shape_ops.frame(mel_spect, num_timesteps, num_timesteps//4, axis=0),perm=[0,2,1,3])

    lbl = context_out['label']

    _, idx = tf.compat.v1.setdiff1d(tf.constant(all_labels), lbl)
    idx, _ = tf.compat.v1.setdiff1d(tf.range(len(all_labels)), idx)

    lblIndex = tf.fill([tf.shape(input=spectrogram_frames)[0]], tf.cast(idx[0], dtype=tf.int32))
    files = tf.fill([tf.shape(input=spectrogram_frames)[0]], fn)
    return spectrogram_frames, lblIndex, files


def write_features(D, label, file_name):
    '''
    writes features to tensorflow serialized file
    :param D: matrix of features. time dimension should be the first
    :param label: label to write with features
    :param file_name: file name to write in
    '''

    print('Writing', file_name)
    writer = tf.io.TFRecordWriter(file_name)

    if D.ndim==2:
        example = tf.train.SequenceExample(
            context = tf.train.Features(feature = {
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                "spectrogram":
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=d))
                            for d in D
                        ]
                    ),
                }
            )
        )
    else:
        example = tf.train.SequenceExample(
            context = tf.train.Features(feature = {
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                "spectrogram":
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=np.reshape(d,[d.shape[0]*d.shape[1]])))
                            for d in D
                        ]
                    ),
                }
            )
        )


    writer.write(example.SerializeToString())
    writer.close()