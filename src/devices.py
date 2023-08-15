from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
