import os
from functools import wraps
import tensorflow as tf

from .decorator import _register as REGISTERED


__all__ = [
    "distribution_strategy",
]


def distribution_strategy(device: str):
    if REGISTERED["strategy"]:
        return REGISTERED["strategy"](device)
    if device.lower() == "cpu":
        return tf.distribute.get_strategy()
    if device.lower() == "gpu":
        return tf.distribute.MirroredStrategy()
    if device.lower() == "tpu":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)


@wraps(open)
def openfile(file: str, *args, **kwargs):
    """
    Opens file, given a path, for i/o operations.
    Creates the intermidiate directories if path doesn't exist.
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    return open(file, *args, **kwargs)
