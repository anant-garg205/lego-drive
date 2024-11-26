from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class Configs:
    PKG: Path = Path(__file__).parents[0]
    ROOT: Path = Path(__file__).parents[1]

    def _load(self, file):
        return OmegaConf.load(file)

    @classmethod
    def carla(cls):
        return cls()._load(f"{cls.PKG}/configs/carla.yaml")

    @classmethod
    def globals(cls):
        return cls()._load(f"{cls.PKG}/configs/globals.yaml")

    @classmethod
    def planner(cls):
        return cls()._load(f"{cls.PKG}/configs/planner.yaml")

    @classmethod
    def glc(cls):
        return cls()._load(f"{cls.PKG}/configs/glc.yaml")

    @classmethod
    def e2e(cls):
        return cls()._load(f"{cls.PKG}/configs/e2e.yaml")
    
    @classmethod
    def sensors(cls):
        return cls()._load(f"{cls.PKG}/configs/sensors.yaml")
