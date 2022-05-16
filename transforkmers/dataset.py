import os
from typing import Optional, Dict, Union
from io import TextIOWrapper
import logging

log = logging.getLogger("commands")

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding

from .fasta import Fasta
from .tokenizer import DNATokenizer


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: Union[BatchEncoding, Dict[str, torch.Tensor]],
        ids: Optional[np.array] = None,
        y: Optional[torch.Tensor] = None,
    ) -> None:
        self.x = x

        if y is not None:
            self._check_lengths(y, x["input_ids"])
        self.y = y

        if ids is not None:
            self._check_lengths(ids, x["input_ids"])
        self.ids = ids

    def __len__(self) -> int:
        return len(self.x["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.x.items()}
        if self.y is not None:
            item["labels"] = self.y[idx]
        return item

    @staticmethod
    def _check_lengths(a, b):
        assert len(a) == len(b)

    @classmethod
    def from_csv(cls, fd: TextIOWrapper, tokenizer, with_label=True) -> "SequenceDataset":
        log.info(f"Tokenizing {os.path.abspath(fd.name)}")
        assert next(fd).rstrip().split(",") == ["id", "sequence", "label"]  # Verify header

        ids, X, y = [], [], []
        for line in fd:
            line = line.rstrip().split(",")
            ids.append(line[0])
            X.append(line[1])
            if with_label:
                y.append(int(line[2]))
        X = tokenizer(X, padding="max_length")
        log.info(f"Sequences tokenized. {len(ids)} sequences to analyze.")
        if with_label:
            y = torch.tensor(y)
            return cls(x=X, ids=ids, y=y)
        return cls(x=X, ids=np.array(ids))

    @classmethod
    def from_txt(
        cls, fd: TextIOWrapper, tokenizer: DNATokenizer, block_size: int, **kwargs
    ) -> "SequenceDataset":
        log.info(f"Creating features from {fd.name}.")
        lines = [line for line in fd if (len(line) > 0 and not line.isspace())]
        X = tokenizer(
            lines,
            add_special_tokens=True,
            max_length=block_size,
            padding="max_length",
            **kwargs,
        )
        log.info(f"{len(X['input_ids'])} sequences parsed.")
        return cls(x=X)

    @classmethod
    def from_fasta(cls, fd, tokenizer) -> "SequenceDataset":
        log.info(f"Tokenizing {os.path.abspath(fd.name)}")
        ids, X = [], []
        for record in Fasta(fd):
            ids.append(record.id)
            X.append(record.seq)
        X = tokenizer(X)
        log.info(f"Sequences tokenized. {len(ids)} sequences to analyze.")
        return cls(x=X, ids=np.array(ids))
