from typing import Tuple, Dict, Iterable, Optional
import itertools
import os
import logging
from transformers.tokenization_utils import PreTrainedTokenizer
import collections


logger = logging.getLogger("commands")

PAD_TOKEN: str = "[PAD]"
UNK_TOKEN: str = "[UNK]"
SEP_TOKEN: str = "[SEP]"
CLS_TOKEN: str = "[CLS]"
MASK_TOKEN: str = "[MASK]"
ALPHABET: str = "ATCG"
K: int = 6


class DNATokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        k=K,
        alphabet=ALPHABET,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        do_lower_case=False,
        **kwargs,
    ):
        super().__init__(
            k=k,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.k = k
        self.vocab = self.create_vocab(
            alphabet,
            k,
            special_tokens=[
                self.pad_token,
                self.unk_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
            ],
        )
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def _tokenize(self, text):
        k = self.k
        max_len = self.model_max_length
        if k > len(text):
            raise Exception("kmer size should be longer than sequence.")
        yield self.cls_token
        for i in range(len(text) - k):
            # + 3 to compensate with [CLS] and [SEP]
            if max_len is not None and i + 3 > max_len:
                break
            yield text[i : i + k]
        yield self.sep_token

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, "vocab.txt")
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    @classmethod
    def create_vocab(
        cls,
        alphabet: Iterable[str],
        k: int,
        special_tokens=[PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN],
    ) -> Dict[str, int]:
        vocab = {}
        for i, token in enumerate(special_tokens):
            vocab[token] = i
        for i, kmer in enumerate(itertools.product(alphabet, repeat=k), start=len(special_tokens)):
            vocab["".join(kmer)] = i
        return vocab
