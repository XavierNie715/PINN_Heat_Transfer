import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from Utilities.utilities import neural_net, tf_session, mean_squared_error, relative_error, transient_1D_error


class PINN(object):

    def __init__(self, tau_data, x_data, T_data,

                 # B.C. && I.C.
                 tau_U, x_U, T_U,
                 tau_D, x_D, T_D,
                 tau_I, x_I, T_I,

                 layers, batch_size):

        # specs
        self.layers = layers
        self.batch_size = batch_size


        # data
        [self.tau_data, self.x_data, self.T_data] = [tau_data, x_data, T_data]

        [self.tau_U, self.x_U, self.T_U] = [tau_U, x_U, T_U]
        [self.tau_D, self.x_D, self.T_D] = [tau_D, x_D, T_D]
        [self.tau_I, self.x_I, self.T_I] = [tau_I, x_I, T_I]

        # placeholders
        [self.tau_data_tf, self.x_data_tf, self.T_data_tf] = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        [self.tau_U_tf, self.x_U_tf, self.T_U_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.tau_D_tf, self.x_D_tf, self.T_D_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.tau_I_tf, self.x_I_tf, self.T_I_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        # neural networks
        self.net_T = neural_net(self.tau_data, self.x_data, layers=self.layers)

        self.T_data_pred = self.net_T(self.tau_data_tf, self.x_data_tf)

        # neural networks data at the B.C. && I.C.
        self.T_U_pred = self.net_T(self.tau_U_tf, self.x_U_tf)
        self.T_D_pred = self.net_T(self.tau_D_tf, self.x_D_tf)
        self.T_I_pred = self.net_T(self.tau_I_tf, self.x_I_tf)

        self.e_pred = transient_1D_error(self.T_data_pred, self.tau_data_tf, self.x_data_tf)

        # loss
        self.loss = mean_squared_error(self.T_U_pred, self.T_U_tf) + \
                    mean_squared_error(self.T_D_pred, self.T_D_tf) + \
                    mean_squared_error(self.T_I_pred, self.T_I_tf) + \
                    mean_squared_error(self.e_pred, 0.0)

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf_session()

    def train(self, total_time, learning_rate):

        N_data = self.tau_data.shape[0]  # 坐标

        start_time = time.time()
        begin_time = time.time()
        running_time = 0
        it = 0

        while running_time < total_time:

            # for randomly choose data batch
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))

            (tau_data_batch,
             x_data_batch,
             T_data_batch) = (self.tau_data[idx_data, :],
                              self.x_data[idx_data, :],
                              self.T_data[idx_data, :])

            tf_dict = {self.tau_data_tf: tau_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.T_data_tf: T_data_batch,

                       self.tau_U_tf: self.tau_U,
                       self.x_U_tf: self.x_U,
                       self.T_U_tf: self.T_U,

                       self.tau_D_tf: self.tau_D,
                       self.x_D_tf: self.x_D,
                       self.T_D_tf: self.T_D,

                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)

            # Print loss
            if it % 100 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh'
                      % (it, loss_value, elapsed, ((time.time() - begin_time) / 3600)))
                sys.stdout.flush()
                start_time = time.time()
                f = open("./Results/train_1d.txt", "a")  # 记录loss
                f.write("It: {} ".format(it))
                f.write("Loss: {:.3e}\n".format(loss_value))

            # print error && save
            if it % 1000 == 0:
                T_pred = 0 * T_star
                T_pred = model.predict(tau_star, x_star)
                error_T = relative_error(T_pred, T_star)
                print('**************It: %d, Error T: %e**************'
                      % (it, error_T))
                f = open("./Results/error_1d.txt", "a")  # 存error
                f.write("It: {} ".format(it))
                f.write("error_T: {:.3e}\n".format(error_T))

                scipy.io.savemat('./Results/Transient1D_results_%s.mat' % (time.strftime('%d_%m_%Y')),
                                 {'T_pred': T_pred})

            # shutdown train
            if loss_value <= 6e-5:
                break

            it += 1

    def predict(self, tau_star, x_star):

        tf_dict = {self.tau_data_tf: tau_star, self.x_data_tf: x_star}
        T_star = self.sess.run(self.T_data_pred, tf_dict)

        return T_star


if __name__ == "__main__":

    batch_size = 10000

    layers = [2] + 10 * [1 * 50] + [1]  # input * layers * [output * neurons] * output

    # Load Data
    data = scipy.io.loadmat('./Data/1d_transient.mat')

    tau_star = data['tau_star'] # T x 1 (T = N)
    x_star = data['x_star'] # N x 1

    TAU = tau_star.shape[0]
    N = x_star.shape[0]

    T_star = data['T_star']  # N x T

    # Rearrange Data
    TAU_star = np.tile(tau_star, (1, N)).T  # N x T
    X_star = np.tile(x_star, (1, TAU))  # N x T

    tau = TAU_star.flatten()[:, None]  # NT x 1
    x = X_star.flatten()[:, None]  # NT x 1
    T = T_star.flatten()[:, None]  # NT x 1

    ######################################################################
    ######################## Training Data ###############################
    ######################################################################

    TAU_data = TAU  # int(sys.argv[1])
    N_data = N  # int(sys.argv[2])
    idx_tau = np.concatenate([np.array([0]), np.random.choice(TAU - 2, TAU_data - 2, replace=False) + 1, np.array([TAU - 1])])
    idx_x = np.random.choice(N, N_data, replace=False)
    tau_data = tau_star[:, idx_tau][idx_x, :].flatten()[: , None]
    x_data = x_star[:, idx_tau][idx_x, :].flatten()[: , None]
    T_data = T_star[:, idx_tau][idx_x, :].flatten()[: , None]

    # Training Data on B.C. && I.C.
    tau_U = tau[x == x.max()][:, None]
    x_U = x[x == x.max()][:, None]
    T_U = T[x == x.max()][:, None]

    tau_D = tau[x == x.min()][:, None]
    x_D = x[x == x.min()][:, None]
    T_D = T[x == x.min()][:, None]

    tau_I = tau[tau == tau.min()][:, None]
    x_I = x[tau == tau.min()][:, None]
    T_I = T[tau == tau.min()][:, None]


    # Training
    model = PINN(tau_data, x_data, T_data,

                tau_U, x_U, T_U,
                tau_D, x_D, T_D,
                tau_I, x_I, T_I,

                layers, batch_size)

    model.train(total_time=20, learning_rate=1e-3)

    ################# Save Data ###########################

    T_pred = 0 * T_star

    # Prediction
    T_pred = model.predict(tau_star, x_star)

    # Error
    error_T = relative_error(T_pred, T_star)

    print('Error T: %e' % (error_T))

    f = open("./Results/error_1d.txt", "a")  # 存error

    f.write("error_T: {:.3e}".format(error_T))

    scipy.io.savemat('./Results/Transient1D_results_%s.mat' % (time.strftime('%d_%m_%Y')), {'T_pred': T_pred})
