"""
Defines functions to create models.
"""
from ..api import config
import typing
import tensorflow as tf
from math import log2


def InceptionBlock(filters: int, kernels: typing.List, *, inputs, name, **kwargs):
    kwargs["strides"] = 1
    kwargs["padding"] = "same"
    # inputs = tf.keras.layers.Input(shape=input_shape)
    conv = tf.keras.layers.add(
        [
            tf.keras.layers.Conv1D(filters, kernel, name=f"{name}_{i}", **kwargs)(
                inputs
            )
            for i, kernel in enumerate(kernels)
        ]
    )
    pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same")(conv)
    norm = tf.keras.layers.BatchNormalization()(pool)
    return norm


def ResidualBlock(
    filters: typing.List,
    kernels: typing.Union[typing.List, int],
    *,
    inputs,
    name,
    skip_conections: typing.Optional[typing.List],
    **kwargs,
):
    if skip_conections:
        assert len(filters) == len(
            skip_conections
        ), "skip_connections and residual layers mismatch"
    else:
        skip_conections = [None] * len(filters)

    if not isinstance(kernels, int):
        assert len(filters) == len(kernels), "kernels and residual layers mismatch"
    else:
        kernels = [kernels] * len(filters)

    kwargs["strides"] = 1
    kwargs["padding"] = "same"

    def res_layer(_inputs, filter, kernel, skips):
        # TODO:
        # Add Self-Attention layer after _residue, before _norm
        res_layer.n += 1
        kwargs["name"] = f"{name}_{res_layer.n}"
        # _inputs = tf.keras.layers.Input(shape=shape)
        _conv = tf.keras.layers.Conv1D(filter, kernel, **kwargs)(_inputs)
        channels = [layer.shape[-1] for layer in [_conv, *skips]]
        lengths = [layer.shape[-2] for layer in [_conv, *skips]]
        min_len = min(lengths)
        max_channel = max(channels)
        _add = [_conv, *skips]
        for i in range(len(channels)):
            if channels[i] < max_channel:
                _add[i] = tf.keras.layers.Conv1D(max_channel, 1, 1)(_add[i])
        for i in range(len(lengths)):
            if lengths[i] > min_len:
                factor = int(log2(lengths[i] // min_len))
                for _ in range(factor):
                    _add[i] = tf.keras.layers.Conv1D(max_channel, 1, 2)(_add[i])
        _residue = tf.keras.layers.add(_add)
        _norm = tf.keras.layers.BatchNormalization()(_residue)
        return _norm  # tf.keras.models.Model(
        #     inputs=_inputs,
        #     outputs=_norm,
        #     name=f"ResidualLayer_{res_layer.n}"
        # )

    res_layer.n = -1

    residue = inputs  # = tf.keras.layers.Input(shape=input_shape)
    for i, (filter, kernel, skip) in enumerate(zip(filters, kernels, skip_conections)):
        if i == 0:
            out = res_layer(inputs, filter, kernel, [skip] if skip is not None else [])
        else:
            out, residue = (
                res_layer(
                    out,
                    filter,
                    kernel,
                    [residue, skip] if skip is not None else [residue],
                ),
                out,
            )
    return out


def DeconBlock(filters: int, kernels: int, *, inputs, name, **kwargs):
    kwargs["strides"] = 2
    kwargs["padding"] = "same"
    kwargs["name"] = name
    # inputs = tf.keras.layers.Input(shape=input_shape)
    deconv = tf.keras.layers.Conv1DTranspose(filters, kernels, **kwargs)(inputs)
    norm = tf.keras.layers.BatchNormalization()(deconv)
    return norm


def create_generator(
    name="Generator",
    input_shape=(2**19, 1),
    conv_filters=[16, 32, 64],
    conv_kernels=[32, 64, 128],
    res_filters=[128, 128, 256, 128, 128],
    decov_filters=[64, 16, 1],
    **kwargs,
):
    """
    res_filters length = 2*conv_filters -1
    """
    out = inputs = tf.keras.layers.Input(shape=input_shape)
    skips = []
    for i, filter in enumerate(conv_filters):
        out = InceptionBlock(
            filter,
            conv_kernels,
            inputs=out,
            name=f"InceptionBlock_{i}",
            activation=kwargs.get("activation", config.conv.activation),
        )  # (out)
        skips.append(out)

    out = ResidualBlock(
        res_filters,
        64,
        inputs=out,
        name="ResidualBlock",
        skip_conections=skips[::-1] + skips[1:],
        activation=kwargs.get("activation", config.residual.activation),
    )  # (out)

    for i, filter in enumerate(decov_filters):
        out = DeconBlock(
            filter,
            128,
            inputs=out,
            name=f"Deconvolution_{i}",
            activation=kwargs.get("activation", config.deconv.activation),
        )  # (out)

    out = tf.nn.tanh(out, name="Output")
    return tf.keras.models.Model(inputs=inputs, outputs=out, name=name)


def create_discriminator(
    name="Discriminator",
    input_shape=(2**19, 1),
    conv_filters=[16, 32, 64, 128],
    conv_kernels=[256, 128, 64, 32],
    res_filters=[4, 8, 8, 16, 8, 8, 4],
    **kwargs,
):
    """
    Creates a discriminator. Downsamples the input by
    a factor of len(conv_filters).
    """
    out = inputs = tf.keras.Input(shape=input_shape)
    skips = []
    for i, filter in enumerate(conv_filters):
        out = InceptionBlock(
            filter,
            conv_kernels,
            inputs=out,
            name=f"InceptionBlock_{i}",
            activation=kwargs.get("activation", config.conv.activation),
        )  # (out)
        skips.append(out)

    out = ResidualBlock(
        res_filters,
        64,
        inputs=out,
        name="ResidualBlock",
        skip_conections=skips[::-1] + skips[1:],
        activation=kwargs.get("activation", config.residual.activation),
    )  # (out)

    out = tf.keras.layers.Conv1D(1, 1, name="Output", activation=tf.nn.sigmoid)(out)
    return tf.keras.Model(inputs, out, name=name, **kwargs)
