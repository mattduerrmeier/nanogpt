import tiktoken
from torch.utils.data import Dataset
import torch
import os


class ShakeSpeareLoader:
    def __init__(
        self, B: int = 64, T: int = 1024, dataset_path: str = "data/input.txt"
    ):
        self.B = B
        self.T = T

        with open(dataset_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def __len__(self) -> int:
        return len(self.tokens) // (self.B * self.T)

    def next_batch(self):
        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_pos += self.B * self.T
        if self.current_pos + (self.B * self.T + 1) > len(self.tokens):
            self.current_pos = 0

        return x, y


class TinyStories:
    def __init__(
        self,
        B: int = 64,
        T: int = 1024,
        split: str = "train",
        dataset_root_dir: str = "data/",
    ) -> None:
        self.B = B
        self.T = T

        if split == "train":
            split_path = "tinystories-train.txt"
        elif split in ["val", "test"]:
            split_path = "tinystories-val.txt"
        else:
            raise ValueError("not a valid split")

        dataset_path = os.path.join(dataset_root_dir, split_path)

        with open(dataset_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def __len__(self) -> int:
        return len(self.tokens) // (self.B * self.T)

    def next_batch(self):
        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_pos += self.B * self.T
        if self.current_pos + (self.B * self.T + 1) > len(self.tokens):
            self.current_pos = 0

        return x, y


class ShakeSpeareDataset(Dataset):
    def __init__(self, T: int = 1024, dataset_path: str = "data/input.txt"):
        self.T = T

        with open(dataset_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def __len__(self) -> int:
        return len(self.tokens) // self.T

    def __getitem__(self, idx):
        buf = self.tokens[(idx * self.T) : (idx * self.T) + self.T + 1]
        x = buf[:-1].clone()
        y = buf[1:].clone()
        return x, y
