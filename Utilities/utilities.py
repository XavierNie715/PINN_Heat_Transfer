import numpy as np
import tensorflow as tf


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))


def relative_error(pred, exact):
    return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    # if type(pred) is np.ndarray:
    #     return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    # return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))


def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


class neural_net(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y


def condution_2D_error(T, x, y):
    Y = tf.concat(T, 1)

    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    # T = Y[:, 0:1]
    # T_x = Y_x[:, 0:1]
    # T_y = Y_y[:, 0:1]

    T_xx = Y_xx[:, 0:1]
    T_yy = Y_yy[:, 0:1]

    e = T_xx + T_yy

    return e
