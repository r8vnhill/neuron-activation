import torch
from torch.utils.data import DataLoader

from datasets import RandomDataset


def main():
    dataset = RandomDataset(1000, 20, 10)
    data = DataLoader(dataset, batch_size=4)
    for x, y in data:
        print(f"Data::{x.dtype} {x.shape}:")
        print(x)
        print(f"Labels::{y.dtype} {y.shape}:")
        print(y)


if __name__ == "__main__":
    main()
