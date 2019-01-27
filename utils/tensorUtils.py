import tensorflow as tf


def initAllSummaries(variablesDict):
    for k in variablesDict.keys():
        v = variablesDict[k]
        if (v.shape == tf.TensorShape(None)):
            # tf.placeholder(tf.float32)
            tf.summary.scalar(name=k, tensor=v)
        elif (len(v.shape) <= 0 or max(v.shape) <= 1):
            # tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))
            tf.summary.scalar(name=k, tensor=v)
        elif (max(v.shape) < 1e5):
            # not is tf.placeholder(tf.float32, [None, 784])
            tf.summary.histogram(name=k, values=v)
