from .base import BaseModel as Model
from ..api import config


class AcapellaModel(Model):
    """
    Acapella model class.
    """

    def __init__(self, dir: str = config.save_dir):
        """
        Initializes the AcapellaModel.
        """
        self.name = "Acapella"
        self.dir = dir
        self.config = f"{dir}/AcapellaModel.config"


class InstrumentalModel(Model):
    """
    Instrumental model class.
    """

    def __init__(self, dir: str = config.save_dir):
        """
        Initializes the InstrumentalModel.
        """
        self.name = "Instrumental"
        self.dir = dir
        self.config = f"{dir}/InstrumentalModel.config"


class IntermixerModel(Model):
    """
    Intermixer model class.
    """

    def __init__(self, dir: str = config.save_dir):
        """
        Initializes the IntermixerModel.
        """
        self.name = "Intermixer"
        self.dir = dir
        self.config = f"{dir}/IntermixerModel.config"
