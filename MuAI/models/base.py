import json
from typing import Any, Optional
from .utils import create_discriminator, create_generator


class BaseModel:
    @property
    def config(self):
        '''
        Returns the model configuration.
        '''
        return {
            "name": self.name,
            "dir": self.dir,
            "generator": {} if not self.generator else self.generator.get_config(),
            "discriminator": {} if not self.discriminator else self.discriminator.get_config(),
            "input_shape": self.input_shape,
            "conv_filters": self.conv_filters,
            "res_filters": self.res_filters,
            "lr": self.lr,
        }
    
    @config.setter
    def config(self, config_path: str):
        '''
        Sets the model configuration.
        '''
        config = {}
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")

        self.generator = None
        self.discriminator = None
        self.input_shape = config.get("input_shape", (64, None, 1))
        self.conv_filters = config.get("conv_filters", [4, 8, 16, 32])
        self.res_filters = config.get("res_filters", [4, 8, 16, 32])
        self.lr = config.get("lr", 0.001)
    
    def build(self, model:Optional[str]=None, **kwargs):
        '''
        If model is specified, Builds/Creates the model.
        Else,/Creates both, the Generator and the Discriminator.
        '''
        if not model or model.lower()=="generator":
            self.generator = create_generator(
                name=self.name+"_Generator",
                input_shape=self.input_shape,
                conv_filters=self.conv_filters,
                res_filters=self.res_filters,
                **kwargs
            )
        if not model or model.lower()=="discriminator":
            self.discriminator = create_discriminator(
                name=self.name+"_Discriminator",
                input_shape=self.input_shape,
                conv_filters=self.conv_filters,
                res_filters=self.res_filters,
                **kwargs
            )
    
    def summary(self, model:Optional[str]=None):
        '''
        If model is specified, Prints a summary of the model.
        Else, of both, the Generator and the Discriminator.
        '''
        if not self.generator or not self.discriminator:
            print("Model not built.")
            return
        if not model or model.lower()=="generator":
            self.generator.summary()
        if not model or model.lower()=="discriminator":
            self.discriminator.summary()

    def freeze(self, model:Optional[str]=None):
        '''
        If model is specified, Freezes the model.
        Else, Freezes both, the Generator and the Discriminator.
        '''
        if not model or model.lower()=="generator":
            self.generator.trainable = False
        if not model or model.lower()=="discriminator":
            self.discriminator.trainable = False
    
    def unfreeze(self, model:Optional[str]=None):
        '''
        If model is specified, Unfreezes the model.
        Else, Unfreezes both, the Generator and the Discriminator.
        '''
        if not model or model.lower()=="generator":
            self.generator.trainable = True
        if not model or model.lower()=="discriminator":
            self.discriminator.trainable = True
    
    def compile(self, model:Optional[str]=None, **kwargs):
        '''
        If model is specified, Compiles the model.
        Else, Compiles both, the Generator and the Discriminator.
        '''
        if not self.generator or not self.discriminator:
            print("Model not built.")
            return
        if not model or model.lower()=="generator":
            self.generator.compile(**kwargs)
        if not model or model.lower()=="discriminator":
            self.discriminator.compile(**kwargs)
    
    def load(self, model:Optional[str]=None, dir=None, latest=True):
        '''
        If model is specified, Loads the model.
        Else, Loads both, the Generator and the Discriminator.
        '''
        if not dir:
            dir = self.dir
        if not model or model.lower()=="generator":
            self.generator.load_weights(dir+"/generator", latest=latest)
        if not model or model.lower()=="discriminator":
            self.discriminator.load_weights(dir+"/discriminator", latest=latest)

    def save(self, model:Optional[str]=None, dir=None):
        '''
        If model is specified, Saves the model.
        Else, Saves both, the Generator and the Discriminator.
        '''
        if not dir:
            dir = self.dir
        if not model or model.lower()=="generator":
            self.generator.save(dir+"/generator")
        if not model or model.lower()=="discriminator":
            self.discriminator.save(dir+"/discriminator")
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.generator:
            return self.generator(*args, **kwds)
