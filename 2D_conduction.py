import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from Utilities.utilities import neural_net, tf_session, mean_squared_error, relative_error, condution_2D_error


class PINN(object):

    def __init__(self, x_data, y_data, T_data,

                 # B.C.
                 x_L, y_L, T_L,
                 x_R, y_R, T_R,
                 x_U, y_U, T_U,
                 x_D, y_D, T_D,

                 layers):
        # specs
        self.layers = layers
        # self.batch_size = batch_size

        # data
        [self.x_data, self.y_data, self.T_data] = [x_data, y_data, T_data]

        [self.x_L, self.y_L, self.T_L] = [x_L, y_L, T_L]
        [self.x_R, self.y_R, self.T_R] = [x_R, y_R, T_R]
        [self.x_U, self.y_U, self.T_U] = [x_U, y_U, T_U]
        [self.x_D, self.y_D, self.T_D] = [x_D, y_D, T_D]

        # placeholders
        [self.x_data_tf, self.y_data_tf, self.T_data_tf] = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        [self.x_L_tf, self.y_L_tf, self.T_L_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.x_R_tf, self.y_R_tf, self.T_R_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.x_U_tf, self.y_U_tf, self.T_U_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.x_D_tf, self.y_D_tf, self.T_D_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        # neural networks
        self.net_T = neural_net(self.x_data, self.y_data, layers=self.layers)

        self.T_data_pred = self.net_T(self.x_data_tf, self.y_data_tf)

        # neural networks data at the B.C.
        self.T_L_pred = self.net_T(self.x_L_tf, self.y_L_tf)
        self.T_R_pred = self.net_T(self.x_R_tf, self.y_R_tf)
        self.T_U_pred = self.net_T(self.x_U_tf, self.y_U_tf)
        self.T_D_pred = self.net_T(self.x_D_tf, self.y_D_tf)

        self.e_pred = condution_2D_error(self.T_data_pred, self.x_data_tf, self.y_data_tf, )

        # loss
        self.loss = mean_squared_error(self.T_L_pred, self.T_L_tf) + \
                    mean_squared_error(self.T_R_pred, self.T_R_tf) + \
                    mean_squared_error(self.T_U_pred, self.T_U_tf) + \
                    mean_squared_error(self.T_D_pred, self.T_D_tf) + \
                    mean_squared_error(self.e_pred, 0.0)

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf_session()

    def train(self, total_time, learning_rate):

        # N_data = self.t_data.shape[0]  # 坐标
        # N_eqns = self.t_eqns.shape[0]  # 坐标

        start_time = time.time()
        begin_time = time.time()
        running_time = 0
        it = 0

        while running_time < total_time:
            # for randomly choose data batch

            # idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            # idx_eqns = np.random.choice(N_eqns, self.batch_size)
            #
            # (x_data_batch,
            #  y_data_batch,
            #  T_data_batch) = (self.x_data[idx_data, :],
            #                   self.y_data[idx_data, :],
            #                   self.T_data[idx_data, :])
            #
            # (x_eqns_batch,
            #  y_eqns_batch) = (self.x_eqns[idx_eqns, :],
            #                   self.y_eqns[idx_eqns, :])

            tf_dict = {self.x_data_tf: self.x_data,
                       self.y_data_tf: self.y_data,
                       self.T_data_tf: self.T_data,

                       self.x_L_tf: self.x_L,
                       self.y_L_tf: self.y_L,
                       self.T_L_tf: self.T_L,

                       self.x_R_tf: self.x_R,
                       self.y_R_tf: self.y_R,
                       self.T_R_tf: self.T_R,

                       self.x_U_tf: self.x_U,
                       self.y_U_tf: self.y_U,
                       self.T_U_tf: self.T_U,

                       self.x_D_tf: self.x_D,
                       self.y_D_tf: self.y_D,
                       self.T_D_tf: self.T_D,

                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh'
                      % (it, loss_value, elapsed, (begin_time - time.time()) / 3600))
                sys.stdout.flush()
                start_time = time.time()
                f = open("./Results/train.txt", "a")  # 记录loss
                f.write("It: {} ".format(it))
                f.write("Loss: {:.3e}\n".format(loss_value))


if __name__ == "__main__":

    layers = [2] + 10 * [1 * 50] + [1]  # input * layers * [output * neurons] * output

    # Load Data
    data = scipy.io.loadmat('./Data/2d_conduction.mat')

    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1

    N = x_star.shape[0]

    T_star = data['T_star']  # N x 1

    ######################################################################
    ######################## Training Data ###############################
    ######################################################################

    N_data = N  # int(sys.argv[2])
    idx_x = np.random.choice(N, N_data, replace=False)
    x_data = x_star[idx_x, :]
    y_data = y_star[idx_x, :]
    T_data = T_star[idx_x, :]

    # Training Data on B.C.
    x_L = x_star[x_star == x_star.min()][:, None]
    y_L = y_star[x_star == x_star.min()][:, None]
    T_L = T_star[x_star == x_star.min()][:, None]

    x_R = x_star[x_star == x_star.max()][:, None]
    y_R = y_star[x_star == x_star.max()][:, None]
    T_R = T_star[x_star == x_star.max()][:, None]

    x_U = x_star[y_star == y_star.max()][:, None]
    y_U = y_star[y_star == y_star.max()][:, None]
    T_U = T_star[y_star == y_star.max()][:, None]

    x_D = x_star[y_star == y_star.min()][:, None]
    y_D = y_star[y_star == y_star.min()][:, None]
    T_D = T_star[y_star == y_star.min()][:, None]


    # Training
    model = PINN(x_data, y_data, T_data,

                x_L, y_L, T_L,
                x_R, y_R, T_R,
                x_U, y_U, T_U,
                x_D, y_D, T_D,

                layers)

    model.train(total_time=20, learning_rate=1e-3)


    # # Test Data
    # snap = np.array([100])
    # t_test = T_star[:,snap]
    # x_test = X_star[:,snap]
    # y_test = Y_star[:,snap]
    #
    # c_test = C_star[:,snap]
    # u_test = U_star[:,snap]
    # v_test = V_star[:,snap]
    # p_test = P_star[:,snap]
    #
    # # Prediction
    # c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    #
    # # Error
    # error_c = relative_error(c_pred, c_test)
    # error_u = relative_error(u_pred, u_test)
    # error_v = relative_error(v_pred, v_test)
    # error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    #
    # print('Error c: %e' % (error_c))
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error p: %e' % (error_p))

    ################# Save Data ###########################

    T_pred = 0 * T_star

    # Prediction
    T_pred = model.predict(x_star, y_star)

    # Error
    error_T = relative_error(T_pred, T_star)

    print('Error c: %e' % (error_T))

    f = open("./Results/error.txt", "a")  # 存error

    f.write("error_T: {:.3e}".format(error_T))

    scipy.io.savemat('./Results/Conduction2D_results_%s.mat' % (time.strftime('%d_%m_%Y')), {'T_pred': T_pred})
