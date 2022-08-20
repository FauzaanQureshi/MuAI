from typing import Callable

_register = {
    "strategy": None,
    "generator": None,
    "discriminator": None,
}


def distribution_strategy(callable: Callable) -> None:
    """
    Wrapper to register custom distribution strategy.

    The wrapped callable should return tf.distribute.Strategy or it's
    sub-classed object. The wrapped callable should also take only one
    positional argument for device.

    Wrapped callable signature:
    callable(device:str, **kwargs) -> tf.distribute.Strategy
    """
    _register.strategy = callable


def generator_arch(callable: Callable) -> None:
    """
    Wrapper to register custom generator model architecture.

    The wrapped callable should return a tf.keras.models.Model or
    its sub-classed object. The wrapped callable should take at least
    two arguments for 'name' and 'input_shape'. Any additional arguments
    must be passed as keywords in .build() method of muai.Model or
    muai.Trainer object or sub-classed object.

    Wrapped callable signature:
    callable(name:str = "Generator", input_shape:List = [...]) -> Model
    """
    _register.generator = callable


def discriminator_arch(callable: Callable) -> None:
    """
    Wrapper to register custom discriminator model architecture.

    The wrapped callable should return a tf.keras.models.Model or
    its sub-classed object. The wrapped callable should take at least
    two arguments for 'name' and 'input_shape'. Any additional arguments
    must be passed as keywords in .build() method of muai.Model or
    muai.Trainer object or sub-classed object.

    Wrapped callable signature:
    callable(name:str = "Discriminator", input_shape:List = [...]) -> keras.model.Model
    """
    _register.discriminator = callable
