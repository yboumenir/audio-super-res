import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from scipy import interpolate

# Assuming Model and default_opt are defined elsewhere
from .model import Model, default_opt
from .layers.subpixel import SubPixel1D


# ----------------------------------------------------------------------------

class Spline(Model):
    """Generic TensorFlow model training code for spline interpolation"""

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
            x = X  # The model currently just returns the input without any modifications
            return x

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
