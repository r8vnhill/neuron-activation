from enum import Enum


class Device(str, Enum):
    """
    Enumeration for specifying computational devices.

    This enumeration categorizes the computational devices that can be used for
    tasks such as neural network training, data processing, and more. The available
    options include CPU, CUDA for Nvidia GPUs, and MPS for CUDA Multi-Process Service.

    :cvar CPU: Represents computations using the central processing unit.
    :cvar CUDA: Represents computations using Nvidia's GPU architecture.
    :cvar MPS: Represents the CUDA Multi-Process Service, which allows for
               concurrent execution of CUDA kernels.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
