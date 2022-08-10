from types import SimpleNamespace
import tensorflow as tf


config = SimpleNamespace(**dict(
    window=2**19,
    conv=dict(
        filters=[16, 32, 64],
        kernels=[32, 64, 128],
        activation=tf.nn.relu,
    ),
    residual=dict(
        filters=[128, 128, 256, 128, 128],
        kernels=64,
        activation=tf.nn.relu,
    ),
    deconv=dict(
        filters=[64, 16, 1],
        kernels=[256, 128, 64],
        activation=tf.nn.relu,
    ),
    generators=dict(
        optimizer=tf.optimizers.Adadelta,
        lr=0.1
    ),
    discriminators=dict(
        optimizer=tf.optimizers.Adadelta,
        lr=0.1
    ),
    data_path=__file__.rsplit("\\")[0]+"\\data",
    save_dir=__file__.rsplit("\\")[0]+"\\models",
    logdir=__file__.rsplit("\\")[0]+"\\logs",
    verbose=1,
    device="GPU",
))