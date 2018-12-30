import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import get_regularizer
from mean_zero_constraint import MeanZero


def residual_model(inputs, num_classes, reuse = None, is_training=True, n_filt=8, is_music_layer1=True, n_res_lay=3, layer1_size=8, use_max_pool=None, use_atrous=True, n_modules=1, reg=3, weight_decay=0.0001, scope=''):
    if is_music_layer1:
        wr = get_regularizer(reg, weight_decay)
        net = tf.layers.conv2d(inputs, n_filt, [1, layer1_size], padding='SAME',
                               activation=None, use_bias=False,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=wr,
                               kernel_constraint=MeanZero(1),
                               reuse=reuse, name='conv000')

        net1 = tf.layers.conv2d(inputs, n_filt, [layer1_size, 1], padding='SAME',
                               activation=None, use_bias=False,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=wr,
                               kernel_constraint=MeanZero(0),
                               reuse=reuse, name='conv001')

        net=tf.concat([net, net1], axis=3)

    with slim.variable_scope.variable_scope(scope, 'Model', [inputs, num_classes], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

                if not is_music_layer1:
                    net = slim.conv2d(inputs, n_filt, [layer1_size, layer1_size], normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='conv00')
                    if use_atrous:
                        net = tf.concat([net, slim.conv2d(inputs, n_filt, [layer1_size, layer1_size],rate = 2, normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='conv01')], axis=3)

                n_filt = net.shape[3].value

                if use_max_pool is not None:
                    net = slim.max_pool2d(net,use_max_pool[0:2],stride=use_max_pool[2:4],scope='pool1')

                for i in range(1,n_res_lay+1):
                    for j in range(0,n_modules):
                        net = resnet_block(net, n_filt, 'res' + str(i) + str(j))
                    n_filt *=2


                net = slim.batch_norm(net, scope='postnorm')

                # 1x1 convolution
                net = slim.conv2d(net, 1, [1, 1], normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='gather')

                net = slim.flatten(net, scope='flatten')

                logits = slim.fully_connected(net, num_classes, normalizer_fn=None, activation_fn=None, scope='logits') # restore=restore_logits

                predictions = slim.nn_ops.softmax(logits, name='predictions')

    return logits, predictions



def residual_parameters(weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, reg=3, elu=True):
  """Yields the scope with the default parameters for kaiming_residual.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: Decay for batch norm moving average
    batch_norm_epsilon: Small float added to variance to avoid division by zero
    use_fused_batchnorm: Enable fused batchnorm.
  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': ['moving_vars'],
          'moving_variance': ['moving_vars'],
      }
  }
  wr=get_regularizer(reg, weight_decay)
  with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          weights_regularizer=wr):
    with slim.arg_scope(
        [slim.conv2d, slim.batch_norm],
        activation_fn=slim.nn_ops.elu if elu is True else slim.nn_ops.relu):
      with slim.arg_scope(
          [slim.layers.conv2d],
          weights_initializer=slim.initializers.variance_scaling_initializer(),
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params
            ) as sc:
        return sc


def resnet_block(inputs, n_filt, scope):
    with tf.variable_scope(scope):

        preact = slim.batch_norm(inputs, scope='preact')

        if inputs.shape[3]==n_filt:
                shortcut = inputs
                stride=1
        elif inputs.shape[3] == n_filt / 2:
            stride=2
            shortcut = slim.conv2d(preact, n_filt, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, biases_initializer = None)

        residual = slim.conv2d(preact, n_filt, [3, 3], stride=stride)
        residual = slim.conv2d(residual, n_filt, [3, 3], normalizer_fn=None, activation_fn=None, biases_initializer=None)

        output = shortcut + residual

        return output

