import tensorflow as tf


def sum_tensor(values, name='sum'):

  with tf.name_scope(name):
    total =  tf.Variable(
            lambda: tf.zeros(values.get_shape(), tf.int32),
            trainable=False,
            collections=[ tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES ],
            validate_shape=True,
            name='total')

    update_total_op = tf.assign_add(total, values)

    return total, update_total_op


def get_regularizer(reg, weight_decay):
    wr = None
    if reg == 1:
        wr = tf.contrib.layers.l1_regularizer(weight_decay)
    elif reg == 2:
        wr = tf.contrib.layers.l2_regularizer(weight_decay)
    elif reg == 3:
        wr = tf.contrib.layers.l1_l2_regularizer(weight_decay, weight_decay)
    return wr


def add_summary_ops(names_to_values, add_variables = False, confmat_size = 2):

    so = get_summary_ops(names_to_values, add_variables, confmat_size)
    total_summary = tf.summary.merge(so)

    return total_summary


def get_summary_ops(names_to_values, add_variables = False, confmat_size = 2, init_ops = None):

    summary_ops = []
    if init_ops is not None:
        summary_ops = init_ops
    for metric_name, metric_value in names_to_values.items():
        if not metric_name.endswith('T'):
            op = tf.summary.scalar(metric_name, metric_value)
        else:
            # confusion matrix
            cmi = tf.reshape(tf.cast(metric_value, tf.float32), [1, confmat_size, confmat_size, 1])
            op = tf.summary.image(metric_name, cmi)

        #op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    if add_variables:
        for var in [x for x in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)]:
            summary_ops.append(tf.summary.histogram(var.op.name, var))

    return summary_ops
