import numpy as np
import tensorflow as tf
from scipy import interpolate

# Assuming Model and default_opt are defined elsewhere
from .model import Model, default_opt
from .layers.subpixel import SubPixel1D, SubPixel1D_v2


class AudioUNet(Model):
    """Generic tensorflow model training code"""

    def __init__(self, from_ckpt=False, n_dim=None, r=2,
                 opt_params=default_opt, log_prefix='./run'):
        # Perform the usual initialization
        self.r = r
        super().__init__(from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                         opt_params=opt_params, log_prefix=log_prefix)

    def create_model(self, n_dim, r):
        # Load inputs
        X, _, _ = self.inputs

        with tf.name_scope('generator'):
            x = X
            L = self.layers
            n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
            n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9]
            downsampling_l = []

            print('Building model...')

            # Downsampling layers
            for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
                with tf.name_scope(f'downsc_conv{l}'):
                    x = tf.keras.layers.Conv1D(filters=nf, kernel_size=fs,
                                               padding='same', kernel_initializer='orthogonal',
                                               strides=2)(x)
                    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                    print('D-Block: ', x.shape)
                    downsampling_l.append(x)

            # Bottleneck layer
            with tf.name_scope('bottleneck_conv'):
                x = tf.keras.layers.Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1],
                                           padding='same', kernel_initializer='orthogonal',
                                           strides=2)(x)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            # Upsampling layers
            for l, nf, fs, l_in in reversed(list(zip(range(L), n_filters, n_filtersizes, downsampling_l))):
                with tf.name_scope(f'upsc_conv{l}'):
                    x = tf.keras.layers.Conv1D(filters=2 * nf, kernel_size=fs,
                                               padding='same', kernel_initializer='orthogonal')(x)
                    x = tf.keras.layers.Dropout(rate=0.5)(x)
                    x = tf.keras.layers.Activation('relu')(x)
                    x = SubPixel1D(x, r=2)
                    x = tf.keras.layers.Concatenate(axis=-1)([x, l_in])
                    print('U-Block: ', x.shape)

            # Final conv layer
            with tf.name_scope('lastconv'):
                x = tf.keras.layers.Conv1D(filters=2, kernel_size=9,
                                           padding='same', kernel_initializer='random_normal')(x)
                x = SubPixel1D(x, r=2)
                print(x.shape)

            g = tf.keras.layers.Add()([x, X])

        return g

    def predict(self, X):
        print("Predicting")
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2 ** (self.layers + 1)))]
        X = x_sp.reshape((1, len(x_sp), 1))
        print((X.shape))
        feed_dict = self.load_batch((X, X), train=False)
        return self.sess.run(self.predictions, feed_dict=feed_dict)


# ----------------------------------------------------------------------------
# helpers

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp
