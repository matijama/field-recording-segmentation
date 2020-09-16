import tensorflow as tf


def sum_tensor(values, name='sum'):

  with tf.compat.v1.name_scope(name):
    total =  tf.Variable(
            lambda: tf.zeros(values.get_shape(), tf.int32),
            trainable=False,
            collections=[ tf.compat.v1.GraphKeys.LOCAL_VARIABLES, tf.compat.v1.GraphKeys.METRIC_VARIABLES ],
            validate_shape=True,
            name='total')

    update_total_op = tf.compat.v1.assign_add(total, values)

    return total, update_total_op


def get_regularizer(reg, weight_decay):
    wr = None
    if reg == 1:
        wr = tf.keras.regularizers.l1(weight_decay)
    elif reg == 2:
        wr = tf.keras.regularizers.l2(0.5 * (weight_decay))
    elif reg == 3:
        wr = tf.contrib.layers.l1_l2_regularizer(weight_decay, weight_decay)
    return wr


def add_summary_ops(names_to_values, add_variables = False, confmat_size = 2):

    so = get_summary_ops(names_to_values, add_variables, confmat_size)
    total_summary = tf.compat.v1.summary.merge(so)

    return total_summary


def get_summary_ops(names_to_values, add_variables = False, confmat_size = 2, init_ops = None):

    summary_ops = []
    if init_ops is not None:
        summary_ops = init_ops
    for metric_name, metric_value in names_to_values.items():
        if not metric_name.endswith('T'):
            op = tf.compat.v1.summary.scalar(metric_name, metric_value)
        else:
            # confusion matrix
            cmi = tf.reshape(tf.cast(metric_value, tf.float32), [1, confmat_size, confmat_size, 1])
            op = tf.compat.v1.summary.image(metric_name, cmi)

        #op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    if add_variables:
        for var in [x for x in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.MODEL_VARIABLES)]:
            summary_ops.append(tf.compat.v1.summary.histogram(var.op.name, var))

    return summary_ops
