'''
Defines functions to load and save models.
'''
import typing
import tensorflow as tf
from math import log2


def distribution_strategy(device):
    if device=="cpu":
        return tf.distribute.get_strategy()
    if device=="gpu":
        return tf.distribute.MirroredStrategy()
    if device=="tpu":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)


def ConvBlock(
        inputs,
        filters:typing.List,
        kernel_size:int,
        strides:typing.Union[int,typing.List],
        padding:str,
        activation:typing.Any,
        name:str,
        downsample:bool=True,
        **kwargs
    ):
    '''
    Creates a convolutional block.
    '''
    if isinstance(filters, int):
        filters = [filters]
    if isinstance(strides, int):
        strides = [strides]*len(filters)

    for i in range(len(filters)):
        # downsample
        if downsample:
            inputs = tf.keras.layers.Conv2D(filters[i], kernel_size, strides[i], padding, activation=activation, name=name+"_"+str(i), **kwargs)(inputs)
            inputs = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name=name+"_pool_"+str(i))(inputs)
        # upsample
        else:
            inputs = tf.keras.layers.Conv2DTranspose(filters[i], kernel_size, strides[i], padding, activation=activation, name=name+"_"+str(i), **kwargs)(inputs)
    return inputs


def ResBlock(inputs, filters, kernel_size, strides, padding, activation, name):
    '''
    Creates a residual block.
    '''
    padding = "same"
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation, name=name)(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation, name=name+"_1")(x)
    if x.shape != inputs.shape:
        inputs = tf.keras.layers.Conv2D(filters, 1, 1, "same", name="skip_adjuster_block_"+name)(inputs)
    return tf.keras.layers.Add()([x, inputs])


class Model(tf.keras.Model):
    '''
    Base class for all models.
    '''
    def __init__(
            self,
            name="Model",
            input_shape=(512, None, 1),
            filters=[32, 64, 128, 256],
        ):
        super(Model, self).__init__(name)
        self.i_shape = input_shape
        self.nfilters = filters
        self.build()

    def build(self):
        '''
        Builds the model.
        '''
        inputs = tf.keras.Input(shape=self.i_shape)
        x = ConvBlock(inputs, self.nfilters, 3, 1, "same", tf.nn.relu, "conv1")

        for i in range(4):
            x = ResBlock(x, self.nfilters[i], 3, 1, "same", tf.nn.relu, "res_block_"+str(i))
        
        for i in range(4):
            x = ConvBlock(x, self.nfilters[i], 3, 2, "same", tf.nn.relu, f"deconv_{i}", downsample=False)
        
        x = tf.keras.layers.Conv2D(inputs.shape[-1], 1, 1, "same", activation=tf.nn.relu, name="output")(x)
        self.model = tf.keras.Model(inputs, x)


    def save(self, path):
        '''
        Saves the model to a file.
        '''
        save_model(self, path)

    def load(self, path):
        '''
        Loads a model from a file.
        '''
        self.build()
        load_model(path)

    def freeze(self, layer=None):
        '''
        Freezes a layer.
        '''
        if not layer:
            for layer in self.layers:
                layer.trainable = False
        else:
            layer.trainable = False

    def unfreeze(self, layer=None):
        '''
        Unfreezes a layer.
        '''
        if not layer:
            for layer in self.layers:
                layer.trainable = True
        else:
            layer.trainable = True


def create_generator(
        name="Generator",
        input_shape=(512, None, 1),
        conv_filters=[32, 64, 128, 256],
        res_filters=[128, 128, 256, 256],
        **kwargs
    ):
    '''
    Creates a generator.
    '''
    if input_shape[0] is not None:
        assert input_shape[0] % (2**len(conv_filters)) == 0, f"Input Height must be a multiple of {2**len(conv_filters)} to output same height in Generator {name}."
    if input_shape[1] is not None:
        assert input_shape[1] % (2**len(conv_filters)) == 0, f"Input Width must be a multiple of {2**len(conv_filters)} to output same width in Generator {name}."
    inputs = tf.keras.Input(shape=input_shape)
    x = ConvBlock(inputs, conv_filters, 3, 1, "same", tf.nn.relu, "conv")

    for i in range(len(res_filters)):
        x = ResBlock(x, res_filters[i], 3, 1, "same", tf.nn.relu, "res_block_"+str(i))
    
    for i in range(len(conv_filters)):
        x = ConvBlock(x, conv_filters[-i-1], 3, 2, "same", tf.nn.relu, f"deconv_{i}", downsample=False)
    
    x = tf.keras.layers.Conv2D(inputs.shape[-1], 1, 1, "same", activation=tf.nn.relu, name="output")(x)
    return tf.keras.Model(inputs, x, name=name, **kwargs)


def create_discriminator(
        name="Discriminator",
        input_shape=(512, None, 1),
        conv_filters=[32, 64, 128, 256],
        res_filters=[256, 128, 64, 32],
        **kwargs
    ):
    '''
    Creates a discriminator. Downsamples the input by a factor of len(conv_filters).
    '''
    inputs = tf.keras.Input(shape=input_shape)
    x = ConvBlock(inputs, conv_filters, 3, 1, "same", tf.nn.relu, "conv")

    for i in range(len(res_filters)):
        x = ResBlock(x, res_filters[i], 3, 1, "same", tf.nn.relu, "res_block_"+str(i))
    
    x = tf.keras.layers.Conv2D(1, 1, 1, "same", activation=tf.nn.relu, name="output")(x)
    return tf.keras.Model(inputs, x, name=name, **kwargs)


def InceptionBlock(filters:int, kernels:typing.List, *, inputs, name, **kwargs):
    kwargs["strides"] = 1
    kwargs["padding"] = "same"
    # inputs = tf.keras.layers.Input(shape=input_shape)
    conv = tf.keras.layers.add([
        tf.keras.layers.Conv1D(filters, kernel, **kwargs)(inputs)
        for kernel in kernels
    ])
    pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same")(conv)
    norm = tf.keras.layers.BatchNormalization()(pool)

    return norm#tf.keras.models.Model(inputs=inputs, outputs=norm, name=name)


def ResidualBlock(
    filters:typing.List,
    kernels:typing.Union[typing.List, int],
    *,
    inputs,
    name,
    skip_conections:typing.Optional[typing.List],
    **kwargs
    ):
    if skip_conections:
        assert len(filters)==len(skip_conections), "skip_connections and residual layers mismatch"
    else:
        skip_conections = [None]*len(filters)
    
    if not isinstance(kernels, int):
        assert len(filters)==len(kernels), "kernels and residual layers mismatch"
    else:
        kernels = [kernels]*len(filters)
    
    kwargs["strides"] = 1
    kwargs["padding"] = "same"

    def res_layer(_inputs, filter, kernel, skips):
        # TODO:
        # Add Self-Attention layer after _residue, before _norm
        res_layer.n += 1
        # _inputs = tf.keras.layers.Input(shape=shape)
        _conv = tf.keras.layers.Conv1D(filter, kernel, **kwargs)(_inputs)
        channels = [layer.shape[-1] for layer in [_conv, *skips]]
        lengths = [layer.shape[-2] for layer in [_conv, *skips]]
        min_len = min(lengths)
        max_channel = max(channels)
        _add = [_conv, *skips]
        for i in range(len(channels)):
            if channels[i]<max_channel:
                _add[i] = tf.keras.layers.Conv1D(max_channel, 1, 1)(_add[i])
        for i in range(len(lengths)):
            if lengths[i]>min_len:
                factor = int(log2(lengths[i]//min_len))
                for _ in range(factor):
                    _add[i] = tf.keras.layers.Conv1D(max_channel, 1, 2)(_add[i])
        _residue = tf.keras.layers.add(_add)
        _norm = tf.keras.layers.BatchNormalization()(_residue)
        return _norm#tf.keras.models.Model(
        #     inputs=_inputs,
        #     outputs=_norm,
        #     name=f"ResidualLayer_{res_layer.n}"
        # )

    res_layer.n = -1

    residue = inputs #= tf.keras.layers.Input(shape=input_shape)
    for i, (filter, kernel, skip) in enumerate(zip(filters, kernels, skip_conections)):
        if i==0:
            out = res_layer(
                inputs,
                filter,
                kernel,
                [skip] if skip is not None else []
            )
        else:
            out, residue = res_layer(
                out,
                filter,
                kernel,
                [residue, skip] if skip is not None else [residue]
            ), out
    return out#tf.keras.models.Model(inputs=inputs, outputs=out, name=name)


def DeconBlock(filters:int, kernels:int, *, inputs, name, **kwargs):
    kwargs["strides"] = 2
    kwargs["padding"] = "same"
    # inputs = tf.keras.layers.Input(shape=input_shape)
    deconv = tf.keras.layers.Conv1DTranspose(filters, kernels, **kwargs)(inputs)
    norm = tf.keras.layers.BatchNormalization()(deconv)

    return norm#tf.keras.models.Model(inputs=inputs, outputs=norm, name=name)

def generator1D(
    name="Generator",
    input_shape=(2**19, 1),
    conv_filters=[16, 32, 64],
    conv_kernels=[32, 64, 128],
    res_filters=[128, 128, 256, 128, 128],
    decov_filters=[64, 16, 1],
    **kwargs
    ):
    '''
    res_filters length = 2*conv_filters -1 
    '''
    activation = kwargs.get("activation", tf.nn.relu)
    out = inputs = tf.keras.layers.Input(shape=input_shape)
    skips = []    
    for i, filter in enumerate(conv_filters):
        out = InceptionBlock(
            filter,
            conv_kernels,
            inputs=out,
            name=f"InceptionBlock_{i}",
            activation=activation,
        )#(out)
        skips.append(out)
    
    out = ResidualBlock(
        res_filters,
        64,
        inputs=out,
        name="ResidualBlock",
        skip_conections=skips[::-1]+skips[1:],
        activation=activation,
    )#(out)

    for i, filter in enumerate(decov_filters):
        out = DeconBlock(
            filter,
            128,
            inputs=out,
            name="Deconvolution_{i}",
            activation=activation
        )#(out)
    
    out = tf.nn.tanh(out, name="Output")
    return tf.keras.models.Model(inputs=inputs, outputs=out, name=name)

    
