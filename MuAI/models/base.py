import json
from typing import Any, Iterable, Mapping, Optional
from .utils import create_discriminator, create_generator
from ..api import config as CONFIG
from ..api.config import Config
from ..api.utils import distribution_strategy


class BaseModel:
    @property
    def config(self):
        """
        Returns the model configuration.
        """
        return Config(
            **{
                "name": self.name,
                "dir": self.dir,
                "generator": {} if not self.generator else self.generator.get_config(),
                "discriminator": {}
                if not self.discriminator
                else self.discriminator.get_config(),
                "input_shape": self.input_shape,
                "conv_filters": self.conv_filters,
                "conv_kernels": self.conv_kernels,
                "res_filters": self.res_filters,
                "res_kernels": self.res_kernels,
                "glr": self.glr,
                "dlr": self.dlr,
            }
        )

    @config.setter
    def config(self, config_path: str):
        """
        Sets the model configuration.
        """
        config = {}
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")

        self.generator = None
        self.discriminator = None
        self.input_shape = config.get("input_shape", [CONFIG.window, 1])
        self.conv_filters = config.get("conv_filters", CONFIG.conv.filters)
        self.conv_kernels = config.get("conv_kernels", CONFIG.conv.kernels)
        self.res_filters = config.get("res_filters", CONFIG.residual.filters)
        self.res_kernels = config.get("res_kernels", CONFIG.residual.kernels)
        self.glr = config.get("glr", CONFIG.generator.lr)
        self.dlr = config.get("dlr", CONFIG.discriminator.lr)

    def build(
        self, model: Optional[str] = None, load_weights=False, dir=None, **kwargs
    ):
        """
        If model is specified, Builds/Creates the model.
        Else,/Creates both, the Generator and the Discriminator.
        """
        with distribution_strategy(device=CONFIG.device).scope():
            if not model or model.lower() == "generator":
                self.generator = create_generator(
                    name=self.name + "_Generator",
                    input_shape=self.input_shape,
                    conv_filters=self.conv_filters,
                    conv_kernels=self.conv_kernels,
                    res_filters=self.res_filters,
                    res_kernels=self.res_kernels,
                    **kwargs,
                )
                if load_weights:
                    self.load(model="generator", dir=dir)
            if not model or model.lower() == "discriminator":
                self.discriminator = create_discriminator(
                    name=self.name + "_Discriminator",
                    input_shape=self.input_shape,
                    conv_filters=self.conv_filters,
                    res_filters=self.res_filters,
                    **kwargs,
                )
                if load_weights:
                    self.load(model="discriminator", dir=dir)

    def summary(self, model: Optional[str] = None):
        """
        If model is specified, Prints a summary of the model.
        Else, of both, the Generator and the Discriminator.
        """
        if not model or model.lower() == "generator":
            if not self.generator:
                print(f"{self.name}.generator not built.")
            else:
                self.generator.summary()
        if not model or model.lower() == "discriminator":
            if not self.discriminator:
                print(f"{self.name}.discriminator not built.")
            else:
                self.discriminator.summary()

    def freeze(self, model: Optional[str] = None):
        """
        If model is specified, Freezes the model.
        Else, Freezes both, the Generator and the Discriminator.
        """
        if not model or model.lower() == "generator":
            self.generator.trainable = False
        if not model or model.lower() == "discriminator":
            self.discriminator.trainable = False

    def unfreeze(self, model: Optional[str] = None):
        """
        If model is specified, Unfreezes the model.
        Else, Unfreezes both, the Generator and the Discriminator.
        """
        if not model or model.lower() == "generator":
            self.generator.trainable = True
        if not model or model.lower() == "discriminator":
            self.discriminator.trainable = True

    def compile(self, model: Optional[str] = None, **kwargs):
        """
        If model is specified, Compiles the model.
        Else, Compiles both, the Generator and the Discriminator.
        """
        if not self.generator or not self.discriminator:
            print("Model not built.")
            return
        if not model or model.lower() == "generator":
            self.generator.compile(**kwargs)
        if not model or model.lower() == "discriminator":
            self.discriminator.compile(**kwargs)

    @property
    def trainable_variables(self):
        __variables = []
        if self.generator:
            __variables += self.generator.trainable_variables
        if self.discriminator:
            __variables += self.discriminator.trainable_variables
        return __variables

    def load(self, model: Optional[str] = None, dir=None, latest=True):
        """
        If model is specified, Loads the model.
        Else, Loads both, the Generator and the Discriminator.
        """
        if not dir:
            dir = self.dir
        if not model or model.lower() == "generator":
            if not self.generator:
                self.build(model="generator")
            self.generator.load_weights(dir + "/generator", latest=latest)
        if not model or model.lower() == "discriminator":
            if not self.discriminator:
                self.build(model="discriminator")
            self.discriminator.load_weights(dir + "/discriminator", latest=latest)

    def save(self, model: Optional[str] = None, dir=None):
        """
        If model is specified, Saves the model.
        Else, Saves both, the Generator and the Discriminator.
        """
        if not dir:
            dir = self.dir
        try:
            if not model or model.lower() == "generator":
                self.generator.save(dir + "/generator")
            if not model or model.lower() == "discriminator":
                self.discriminator.save(dir + "/discriminator")
        except AttributeError:
            raise ValueError(
                f"{'Models' if not model else model.lower()} for {self.name}"
                " does not exist. Use .build(model) to create."
            )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.generator:
            return self.generator(*args, **kwds)
        raise ValueError(
            f"Generator for {self.name} does not exist. " "Use .build(model) to create."
        )

    def __repr__(self) -> str:
        return f"<{self.name} model>"

    def __getattribute__(self, key):
        try:
            return super(BaseModel, self).__getattribute__(key)
        except AttributeError:
            _vg = _vd = None
            if self.generator:
                _vg = getattr(self.generator, key)
            if self.discriminator:
                _vd = getattr(self.discriminator, key)

            if _vg is None and _vd is None:
                return
            if _vg is None and _vd is not None:
                return _vd
            if _vd is None and _vg is not None:
                return _vg
            # if isinstance(_vg, Mapping) and isinstance(_vg, Mapping):
            #     return (_vg, _vd)
            # if isinstance(_vg, Iterable) and isinstance(_vg, Iterable):
            #     return _vg + _vd
            return [_vg, _vd]
