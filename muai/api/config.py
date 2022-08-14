from .utils import openfile
import tensorflow as tf
from pprint import PrettyPrinter


__all__ = ["config", "Config"]


class Config(dict):
    """
    MuAI Config class. Extends dict to provide keys as object attributes.
    """

    def __init__(self, **config):
        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(**config[key])
        super(Config, self).__init__(**config)

    def pretty(config, to_file: str = None):
        """
        Pretty prints the config to std.out if to_file is None.

        Usage:
        For Config objects - config_obj.pretty(to_file)
            @param config is self
        For other objects - Config.pretty(obj, to_file)
            @param conffig is obj
        """
        stream = openfile(to_file, "w") if to_file else None
        PrettyPrinter(indent=4, sort_dicts=False, compact=False, stream=stream).pprint(
            config
        )
        if stream:
            stream.close()

    @property
    def __dict__(self):
        return self

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = Config(
    **dict(
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
        generator=dict(optimizer=tf.optimizers.Adadelta, lr=0.1),
        discriminator=dict(
            optimizer=tf.optimizers.Adadelta,
            lr=0.001,
            conv=dict(filters=[4, 8, 16, 32], kernels=128, activation=tf.nn.relu),
            residual=dict(
                filters=[4, 4, 4, 16, 4, 4, 4], kernels=16, activation=tf.nn.relu
            ),
        ),
        data_dir="data",
        save_dir="models",
        log_dir="logs",
        verbose=1,
        device="GPU",
    )
)


if __name__ == "__main__":
    config.pretty(to_file=config.log_dir + "/config.txt")
