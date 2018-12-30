import tensorflow.contrib.slim as slim
import tempfile
import random
import configparser
import os

from functools import reduce
from tf_readwrite import *
from utils import sum_tensor, add_summary_ops
from model import residual_model, residual_parameters
from sklearn.model_selection import StratifiedKFold


def main(_):
    # read configuration files and immediate settings
    config = configparser.ConfigParser()
    config.read('defaults.ini')

    log_dir=config['Folders']['log_dir']
    if not os.path.isabs(log_dir):
        local = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(local, log_dir)
        tf.gfile.MakeDirs(log_dir)

    # setup temp file for logging
    with tempfile.TemporaryDirectory(prefix='run',dir=log_dir) as tmpdirname:
        write_dir=tmpdirname

    # get a list of files
    dataset, x, all_labels, num_freqs, block_len = train_test_set(['1', '2', 'i', 's'], 0, config['Folders']['db_root'], config.getfloat('Network','block_length_sec'), ['effects', 'gtzan', 'mirex2015', 'gni', 'samples2018', 'musan'])

    # setup k-folds
    random.seed(a=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    splits = list(skf.split(np.zeros(len(dataset)), [x[1] for x in dataset]))

    test_acc=[]
    i=1
    for s in splits:
        print('Split: %d' % (i))
        trainset = [dataset[i] for i in s[0]]
        testset = [dataset[i] for i in s[1]]
        acc=trainmodel(trainset, testset, all_labels, num_freqs, block_len, write_dir, config, split_i=i)
        test_acc.append(acc)
        i=i+1

    for a in test_acc:
        print(a)


def trainmodel(trainset, testset, all_labels, num_freqs, block_len, write_dir, config, split_i=None):

    num_categories = len(all_labels)
    shuffle_buffer_size=config.getint('Dataset','shuffle_buffer_size')
    n_epochs = config.getint('Dataset', 'n_epochs')
    batch_size=config.getint('Dataset','batch_size')

    tf.reset_default_graph()

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.getint('Spectrogram','mel_bands'),
        num_spectrogram_bins=num_freqs,
        sample_rate=config.getint('Spectrogram','sample_rate'),
        lower_edge_hertz=config.getfloat('Spectrogram','mel_min'),
        upper_edge_hertz=config.getfloat('Spectrogram','mel_max'))

    # create datasets and iterators
    dataset_train = tf.data.TFRecordDataset([f[0] for f in trainset]). \
        apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size, n_epochs)). \
        map(lambda x: dataset_fft_to_mel_single(x, block_len, num_freqs, linear_to_mel_weight_matrix, all_labels, True), num_parallel_calls=4). \
        batch(batch_size). \
        prefetch(5)

    dataset_test = tf.data.TFRecordDataset([f[0] for f in testset]).\
        map(lambda x: dataset_fft_to_mel_multi(x, block_len, num_freqs, linear_to_mel_weight_matrix, all_labels), num_parallel_calls=4). \
        flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x)). \
        batch(batch_size). \
        prefetch(5)

    handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(
        handle, dataset_train.output_types, dataset_train.output_shapes)
    x, y = iterator.get_next(name='xinput')

    training_iterator = dataset_train.make_initializable_iterator()
    test_iterator = dataset_test.make_initializable_iterator()

    # create model
    with slim.arg_scope(residual_parameters(weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, reg=3, elu=True)):
        logits, predictions = residual_model(x, num_categories, is_training=True, reuse=False,
                                             is_music_layer1=True, n_filt=4, n_res_lay=4, layer1_size=10, use_max_pool=[1, 2, 1, 2], use_atrous=False,
                                             reg=3, weight_decay=0.00004)
        test_logits, test_predictions = residual_model(x, num_categories, is_training=False, reuse=True,
                                         is_music_layer1=True, n_filt=4, n_res_lay=4, layer1_size=10, use_max_pool=[1,2,1,2], use_atrous=False,
                                         reg=3, weight_decay=0.00004)

    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print("Total trainable parameters: ", n)

    one_hot_labels = slim.one_hot_encoding(y, num_categories)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(0.1, global_step, 500, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step)

    predidx = tf.argmax(predictions, axis=1, name='predidx')
    train_measures, train_updates = slim.metrics.aggregate_metric_map({
        'train/Count': tf.contrib.metrics.count(y, name='train_c'),
        'train/Accuracy': tf.metrics.accuracy(y, predidx, name='train_a'),
        'train/PerClass': tf.metrics.mean_per_class_accuracy(y, predidx, num_categories, name='train_ca'),
        'train/ConfusionMT': sum_tensor(tf.confusion_matrix(y, predidx, num_categories, name='train_cm'),name='train_scm')
    })
    train_measures['train/LearningRate'] = learning_rate

    test_predidx = tf.argmax(test_predictions, axis=1, name='test_predidx')
    test_measures, test_updates = slim.metrics.aggregate_metric_map({
        'test/Count': tf.contrib.metrics.count(y, name='test_c'),
        'test/Accuracy': tf.metrics.accuracy(y, test_predidx, name='test_a'),
        'test/PerClass': tf.metrics.mean_per_class_accuracy(y, test_predidx, num_categories, name='test_ca'),
        'test/ConfusionMT': sum_tensor(tf.confusion_matrix(y, test_predidx, num_categories, name='test_cm'), name='test_scm')
    })

    all_m = dict(train_measures)
    all_m.update(test_measures)
    summary = add_summary_ops(all_m, add_variables=True, confmat_size=num_categories)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    sum_dir = write_dir + ('' if split_i is None else '/split' + str(split_i))
    tf.gfile.MakeDirs(sum_dir)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=sum_dir, config=config_proto) as sess:
        training_handle = sess.run(training_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        sess.run(training_iterator.initializer)
        for var in sess.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            sess.run(var.initializer)

        while not sess.should_stop():

            # do training
            [loss, pp, yy] = sess.run([train_op, predidx, y], feed_dict={'iterator_handle:0': training_handle})
            # update training measures
            upd=sess.run(train_updates, feed_dict={'predidx:0':pp, 'xinput:1':yy, handle: training_handle})
            gs = sess.run(global_step)
            if gs%500==0:
                print(gs, loss)
                print(upd['train/Accuracy'])
                if gs%1000==0 or sess.should_stop():
                    # every 1000 iterations, do testing and update test measures
                    acc=test_model(sess, test_iterator, test_handle, test_predidx, y, test_updates)
                    # also reset training stats
                    for var in [x for x in sess.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES) if x.name.startswith('train')]:
                        sess.run(var.initializer)

    print(upd['test/Accuracy'])
    print(upd['test/PerClass'])
    print(upd['test/ConfusionMT'])

    return acc


def test_model(sess, test_iterator, test_handle, predidx, y, updates):
    # init performance counters
    for var in [x for x in sess.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES) if x.name.startswith('test')]:
        sess.run(var.initializer)
    # init test iterator
    sess.run(test_iterator.initializer)
    # do testing
    while True:
        try:
            [pp, yy] = sess.run([predidx, y], feed_dict={'iterator_handle:0': test_handle})
            upd = sess.run(updates, feed_dict={'test_predidx:0': pp, 'xinput:1': yy, 'iterator_handle:0': test_handle})

        except tf.errors.OutOfRangeError:
            break
    print(upd['test/Accuracy'])
    return upd


if __name__ == '__main__':
    tf.app.run()
