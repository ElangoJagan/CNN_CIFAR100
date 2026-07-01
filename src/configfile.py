from dataclasses import dataclass
from typing import List

@dataclass
class DataLoaderConfig:
    raw_data_path : str = "data/raw/cifar100_data.npz"