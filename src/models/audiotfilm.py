import sys
import numpy as np
import tensorflow as tf
from scipy import interpolate

# Assuming Model and default_opt are defined elsewhere
from .model import Model, default_opt
from .layers.subpixel import SubPixel1D

# ----------------------------------------------------------------------------
DRATE = 2


class AudioTfilm(Model):
    def __init__(self, from_ckpt=False, n_dim=None, r=2, pool_size=4, strides=4,
                 opt_params=default_opt, log_prefix='./run'):
        # Perform the usual initialization
        self.r = r
        self.pool_size = pool_size
        self.strides = strides
        super().__init__(from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                         opt_params=opt_params, log_prefix=log_prefix)

    def create_model(self, n_dim, r):
        # Load inputs
        X, _, _ = self.inputs

        with tf.name_scope('generator'):
            x = X
            L = self.layers
            n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
            n_blocks = [128, 64, 32, 16, 8]
            n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
            downsampling_l = []

            print('Building model...')

            def _make_normalizer(x_in, n_filters, n_block):
                """Applies an LSTM layer on top of x_in."""
                x_shape = tf.shape(input=x_in)
                n_steps = x_shape[1] // int(n_block)  # Will be 32 at training

                x_in_down = tf.keras.layers.MaxPooling1D(pool_size=int(n_block), padding='valid')(x_in)
                x_rnn = tf.keras.layers.LSTM(units=n_filters, return_sequences=True)(x_in_down)

                return x_rnn

            def _apply_normalizer(x_in, x_norm, n_filters, n_block):
                x_shape = tf.shape(input=x_in)
                n_steps = x_shape[1] // int(n_block)  # Will be 32 at training

                # Reshape input into blocks
                x_in = tf.reshape(x_in, shape=(-1, n_steps, int(n_block), n_filters))
                x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))

                # Multiply
                x_out = x_norm * x_in

                # Return to original shape
                return tf.reshape(x_out, shape=x_shape)

            # Downsampling layers
            for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
                with tf.name_scope(f'downsc_conv{l}'):
                    x = tf.keras.layers.Conv1D(filters=nf, kernel_size=fs, dilation_rate=DRATE,
                                               padding='same', kernel_initializer='orthogonal')(x)
                    x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, padding='valid', strides=self.strides)(x)
                    x = tf.keras.layers.LeakyReLU(0.2)(x)

                    # Create and apply the normalizer
                    nb = 128 / (2 ** l)
                    x_norm = _make_normalizer(x, nf, nb)
                    x = _apply_normalizer(x, x_norm, nf, nb)

                    print('D-Block: ', x.shape)
                    downsampling_l.append(x)

            # Bottleneck layer
            with tf.name_scope('bottleneck_conv'):
                x = tf.keras.layers.Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate=DRATE,
                                           padding='same', kernel_initializer='orthogonal')(x)
                x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, padding='valid', strides=self.strides)(x)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                x = tf.keras.layers.LeakyReLU(0.2)(x)

                # Create and apply the normalizer
                nb = 128 / (2 ** L)
                x_norm = _make_normalizer(x, n_filters[-1], nb)
                x = _apply_normalizer(x, x_norm, n_filters[-1], nb)

            # Upsampling layers
            for l, nf, fs, l_in in reversed(list(zip(range(L), n_filters, n_filtersizes, downsampling_l))):
                with tf.name_scope(f'upsc_conv{l}'):
                    x = tf.keras.layers.Conv1D(filters=2 * nf, kernel_size=fs, dilation_rate=DRATE,
                                               padding='same', kernel_initializer='orthogonal')(x)

                    x = tf.keras.layers.Dropout(rate=0.5)(x)
                    x = tf.keras.layers.Activation('relu')(x)
                    x = SubPixel1D(x, r=2)

                    # Create and apply the normalizer
                    x_norm = _make_normalizer(x, nf, nb)
                    x = _apply_normalizer(x, x_norm, nf, nb)

                    x = tf.keras.layers.Concatenate()([x, l_in])
                    print('U-Block: ', x.shape)

            # Final conv layer
            with tf.name_scope('lastconv'):
                x = tf.keras.layers.Conv1D(filters=2, kernel_size=9,
                                           padding='same',
                                           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))(x)
                x = SubPixel1D(x, r=2)

            g = tf.keras.layers.Add()([x, X])

        return g

    def predict(self, X):
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2 ** (self.layers + 1)))]
        X = x_sp.reshape((1, len(x_sp), 1))
        feed_dict = self.load_batch((X, X), train=False)
        return self.sess.run(self.predictions, feed_dict=feed_dict)


# ----------------------------------------------------------------------------
# Helpers

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp
